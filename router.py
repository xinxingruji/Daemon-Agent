import json
import math
import os
import sys
import urllib.request
import threading
from typing import List, Dict

# 重配 stdout 编码，防止 UTF-8 内容打印到 GBK 终端时 UnicodeEncodeError
# 必须在任何 print() 之前执行，所以放在 router.py 最顶部
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from utterances import SMALL, LARGE

class Claude_Router:
    def __init__(self, threshold: float = 0.45, mistake_threshold: float = 0.75,
                 mistake_file: str = "mistakes.json", seed_file: str = "seed_vectors.json",
                 safe_tokens: int = 3000, penalty_step: int = 4000, max_mistakes: int = 200):
        self.threshold = threshold
        self.mistake_threshold = mistake_threshold
        self.mistake_file = mistake_file
        self.seed_file = seed_file

        self.safe_tokens = safe_tokens
        self.penalty_step = penalty_step
        self.penalty_rate = 0.05

        self.max_mistakes = max_mistakes

        self.model_name = "nomic-embed-text-v2-moe"
        self.api_url = "http://localhost:11434/api/embeddings"
        self._last_alert_query = ""
        self._last_semantic_query = ""
        self._last_intercept_query = ""

        # 原始种子文本（仅在 seed_vectors.json 不存在时用作回退）
        self.routes = {"small": SMALL, "large": LARGE}

        # 记录 utterances.py 中初始 SMALL 种子的长度
        self.base_small_count = len(SMALL)
        # 设置动态种子触发压缩的阈值
        self.max_dynamic_seeds = 100

        # 供外部反馈用的最近一次路由信息
        self._last_query_vector = None
        self._last_route_scores = {"small": 0.0, "large": 0.0}
        self._last_best_route = None

        print(f"[Router] 初始化...")

        # 1. 加载种子向量（优先用预计算缓存）
        self.route_embeddings_text = {"small": [], "large": []}
        self.route_embeddings = {"small": [], "large": []}
        self._load_seed_vectors()

        # 种子库并发控制
        self.seed_lock = threading.RLock()
        self.is_compressing_seeds = False

        # 错题本并发控制
        self.mistake_lock = threading.RLock()
        self.is_compressing_mistakes = False

        # 2. 加载错题本记录
        self.mistake_book = self._load_mistakes()
        if self.mistake_book:
            print(f"[Router] 已加载 {len(self.mistake_book)} 条错题记录。")

    def _load_seed_vectors(self):
        """从 seed_vectors.json 加载预计算向量；不存在则回退到 utterances.py + Ollama"""
        if os.path.exists(self.seed_file):
            try:
                with open(self.seed_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for route_name in ("small", "large"):
                    entries = data.get(route_name, [])
                    for entry in entries:
                        self.route_embeddings_text[route_name].append(entry["text"])
                        self.route_embeddings[route_name].append(entry["vector"])
                s_cnt = len(self.route_embeddings["small"])
                l_cnt = len(self.route_embeddings["large"])
                print(f"[Router] 已从 {self.seed_file} 加载种子向量: small={s_cnt}, large={l_cnt}")
                return
            except Exception as e:
                print(f"[Router] 读取 {self.seed_file} 失败: {e}，回退到 utterances.py")

        # 回退：用 utterances.py + Ollama（保留向后兼容）
        total = sum(len(v) for v in self.routes.values())
        done = 0
        for route_name, utterances in self.routes.items():
            for text in utterances:
                vec = self._get_embedding(text)
                if vec:
                    self.route_embeddings_text[route_name].append(text)
                    self.route_embeddings[route_name].append(vec)
                done += 1
                pct = done * 100 // total
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r\033[K  [Router] 加载向量: |{bar}| {pct}% ({done}/{total})", end="", flush=True)
        print()
        # 顺便写出缓存，下次启动就用它
        self._save_seed_vectors()

    def _save_seed_vectors(self):
        """将当前种子向量写入 seed_vectors.json"""
        data = {"small": [], "large": []}
        for route_name in ("small", "large"):
            for i, vec in enumerate(self.route_embeddings[route_name]):
                text = self.route_embeddings_text[route_name][i]
                data[route_name].append({"text": text, "vector": vec})
        with open(self.seed_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_seed(self, text: str, route_name: str):
        """添加一条新种子，并支持满载自动压缩"""
        if not text.strip():
            return
            
        # 纯字符串完全重复还是过滤一下
        if text in self.route_embeddings_text.get(route_name, []):
            return
            
        vec = self._get_embedding(text)
        if not vec:
            return

        with self.seed_lock:
            if route_name not in self.route_embeddings:
                self.route_embeddings[route_name] = []
                self.route_embeddings_text[route_name] = []
                
            self.route_embeddings_text[route_name].append(text)
            self.route_embeddings[route_name].append(vec)
            self._save_seed_vectors()
            print(f"[Router 📈] 已添加新种子: '{text}' → {route_name}")

            # 触发判断：当前总长度 - 初始保护长度 >= 设定的动态阈值
            if route_name == "small":
                dynamic_count = len(self.route_embeddings["small"]) - self.base_small_count
                if dynamic_count >= self.max_dynamic_seeds and not self.is_compressing_seeds:
                    self._trigger_compression_async(target="seed")

    # main.py中没用，改成压缩机制了
    def remove_most_similar_seed(self, query_vector, route_name: str):
        """删除 route_name 中与 query_vector 最相似的那条种子"""
        if not query_vector or not self.route_embeddings.get(route_name):
            return
        best_idx = -1
        best_score = -1.0
        for i, vec in enumerate(self.route_embeddings[route_name]):
            score = self._cosine_similarity(query_vector, vec)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0:
            removed_text = self.route_embeddings_text[route_name].pop(best_idx)
            self.route_embeddings[route_name].pop(best_idx)
            self._save_seed_vectors()
            print(f"[Router] 已移除种子: '{removed_text}' ← {route_name} (相似度 {best_score:.3f})")

    def reload_seeds(self):
        """热重载 seed_vectors.json"""
        self.route_embeddings = {"small": [], "large": []}
        self.route_embeddings_text = {"small": [], "large": []}
        self._load_seed_vectors()
        print("[Router] 种子库已热更新")

    def _load_mistakes(self) -> List[Dict]:
        """从本地 JSONL 文件逐行加载错题本"""
        mistakes = []
        if os.path.exists(self.mistake_file):
            try:
                with open(self.mistake_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        clean_line = line.strip()
                        if clean_line:  # 确保跳过空行
                            mistakes.append(json.loads(clean_line))
                return mistakes
            except Exception as e:
                print(f"[Error] 错题本加载失败: {e}")
        return []

    def record_mistake(self, query: str):
        if any(m["query"] == query for m in self.mistake_book):
            return
        print(f"正在将翻车任务记入错题本: '{query}'")
        vec = self._get_embedding(query)
        if not vec: return
            
        record = {"query": query, "vector": vec}

        # 加锁追加到内存
        with self.mistake_lock:
            self.mistake_book.append(record)
            # 在未压缩期间，继续追加写入文件，保证极速落盘
            with open(self.mistake_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 判断是否需要触发压缩 (双重检查，确保只有一个后台任务在运行)
            if len(self.mistake_book) >= self.max_mistakes and not self.is_compressing_mistakes:
                self._trigger_compression_async(target="mistake")
        
    def _trigger_compression_async(self, target: str):
        """统一的后台大模型提炼压缩机制 (支持错题本和种子库)"""
        
        # 1. 状态锁定与快照获取
        if target == "mistake":
            self.is_compressing_mistakes = True
            with self.mistake_lock:
                snapshot = self.mistake_book.copy()
        elif target == "seed":
            self.is_compressing_seeds = True
            with self.seed_lock:
                # 切片魔法：只拿 [保护底座N 之后] 的所有动态种子去压缩
                snapshot = self.route_embeddings_text["small"][self.base_small_count:].copy()

        def _compress_task():
            from config import client
            
            try:
                print(f"\n[Router ⚙️] 启动后台 LLM {target} 压缩机制 (处理 {len(snapshot)} 条数据)...")
                
                # 2. 依据 target 动态构建 Prompt
                if target == "mistake":
                    queries_text = "\n".join(f"- {m['query']}" for m in snapshot)
                    prompt = f"""
                    以下是导致小型AI模型失败的指令清单：
                    {queries_text}
                    请将这些指令抽象并合并为 10-20 个涵盖这些核心难点的通用指令。
                    请严格以 JSON 数组的格式输出纯字符串列表（不要有Markdown代码块格式）。
                    例如：["重构复杂的微服务架构代码", "分析并修复底层的内存泄漏"]
                    """
                else: # target == "seed"
                    queries_text = "\n".join(f"- {q}" for q in snapshot)
                    prompt = f"""
                    以下是小型AI模型近期成功处理的 {len(snapshot)} 个具体任务指令：
                    {queries_text}
                    这些指令过于零散。请提取它们背后的“核心意图”，泛化为 5 到 8 个代表性的通用指令。
                    请严格以 JSON 数组的格式输出纯字符串列表（不要有Markdown代码块格式）。
                    例如：["解析并重构前端页面的组件结构", "生成特定业务模块的自动化单元测试"]
                    """

                # 3. 【复用区域】统一调用大模型、洗白与解析
                resp = client.messages.create(
                    model="large", 
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                
                raw_text = resp.content[0].text.strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.strip("`").replace("json\n", "", 1).strip()
                    
                abstract_queries = json.loads(raw_text)
                print(f"[Router ✨] {target} 提炼完成！提取了 {len(abstract_queries)} 条高度概括的经验。")
                
                # 统一向量化大模型的产出
                compressed_records = []
                for q in abstract_queries:
                    vec = self._get_embedding(q)
                    if vec:
                        compressed_records.append({"query": q, "vector": vec})

                # 4. 【分叉区域】统一使用切片算法完成无缝替换
                if target == "mistake":
                    with self.mistake_lock:
                        # 错题本：压缩后的 + 压缩期间新产生的
                        new_arrivals = self.mistake_book[len(snapshot):]
                        self.mistake_book = compressed_records + new_arrivals
                        
                        with open(self.mistake_file, 'w', encoding='utf-8') as f:
                            for r in self.mistake_book:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    print(f"[Router ✅] 错题本已成功净化，当前容量: {len(self.mistake_book)}")

                elif target == "seed":
                    with self.seed_lock:
                        # 种子库切片合并原理：
                        # [0 : N] -> 我们要保护的 utterances.py 出厂基因
                        # [N : N + len(snapshot)] -> 被大模型压缩掉的旧快照 (丢弃)
                        # [N + len(snapshot) : ] -> 压缩这十几秒内，主线程新塞进来的种子
                        
                        N = self.base_small_count
                        S = len(snapshot)
                        
                        base_texts = self.route_embeddings_text["small"][:N]
                        base_vecs = self.route_embeddings["small"][:N]
                        
                        new_arrivals_texts = self.route_embeddings_text["small"][N + S:]
                        new_arrivals_vecs = self.route_embeddings["small"][N + S:]
                        
                        compressed_texts = [r["query"] for r in compressed_records]
                        compressed_vecs = [r["vector"] for r in compressed_records]

                        # 拼接：基础底座 + 刚刚提炼的高级锚点 + 还没来得及提炼的新货
                        self.route_embeddings_text["small"] = base_texts + compressed_texts + new_arrivals_texts
                        self.route_embeddings["small"] = base_vecs + compressed_vecs + new_arrivals_vecs
                        
                        self._save_seed_vectors()
                    print(f"[Router ✅] 种子库已成功净化，当前 small 容量: {len(self.route_embeddings['small'])}")

            except Exception as e:
                print(f"\n[Router ❌] 后台 {target} 压缩失败 (保持原有状态): {e}")
            finally:
                if target == "mistake":
                    self.is_compressing_mistakes = False
                elif target == "seed":
                    self.is_compressing_seeds = False

        # 启动守护线程
        threading.Thread(target=_compress_task, daemon=True).start()

    def _get_embedding(self, text: str) -> List[float]:
        # ... (与之前代码完全一致，调用 Ollama API) ...
        payload = {"model": self.model_name, "prompt": text}
        try:
            req = urllib.request.Request(self.api_url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode('utf-8'))['embedding']
        except:
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        # ... (与之前代码完全一致的数学计算) ...
        if not vec1 or not vec2: return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

    def route(self, query: str, total_tokens: int = 0, force_large: bool = False,
              force_small: bool = False) -> str:
        """核心路由：结合了错题本拦截与动态 Token 惩罚"""
        if force_large:
            return "large"
        if force_small:
            return "small"
            
        if not query.strip():
            return "large"

        query_vector = self._get_embedding(query)
        if not query_vector:
            return "large"

        # 存储供外部反馈使用
        self._last_query_vector = query_vector

        # ==========================================
        # 🚨 第一道防线：检查错题本
        # ==========================================
        for mistake in self.mistake_book:
            sim = self._cosine_similarity(query_vector, mistake["vector"])
            if sim >= self.mistake_threshold:
                if query != self._last_alert_query:
                    print(f"\033[31m[Router 警报] 触发错题拦截！强制拉起大模型！\033[0m")
                    self._last_alert_query = query
                return "large"

        # ==========================================
        # 🟢 第二道防线：常规语义评估
        # ==========================================
        best_route = "large"
        highest_score = 0.0
        self._last_route_scores = {"small": 0.0, "large": 0.0}

        for route_name, embeddings in self.route_embeddings.items():
            for emb in embeddings:
                score = self._cosine_similarity(query_vector, emb)
                if score > self._last_route_scores.get(route_name, 0.0):
                    self._last_route_scores[route_name] = score
                if score > highest_score:
                    highest_score = score
                    best_route = route_name

        # [优化点] 如果判定为大模型任务，或者最高分数连基础及格线都没过，直接扔给大模型
        if best_route == "large" or highest_score < self.threshold:
            if query != self._last_semantic_query:
                print(f"\033[36m[SemanticRouter] 匹配分数: {highest_score:.3f} -> 判定为大型任务或未达基础线，路由至: large\033[0m")
                self._last_semantic_query = query
            return "large"

        # ==========================================
        # 📈 第三道防线：动态 Token 惩罚计算
        # (运行到这里，说明 best_route == "small" 且 highest_score >= self.threshold)
        # ==========================================
        dynamic_threshold = self.threshold

        if total_tokens > self.safe_tokens:
            extra_steps = (total_tokens - self.safe_tokens) // self.penalty_step
            penalty = extra_steps * self.penalty_rate
            dynamic_threshold = min(0.99, self.threshold + penalty)

            if penalty > 0:
                print(f"\033[33m[Router 测算] 上下文较长 ({total_tokens} tokens)，小模型及格线已从 {self.threshold} 动态上调至 {dynamic_threshold:.3f}\033[0m")

        if query != self._last_semantic_query:
            print(f"\033[36m[SemanticRouter] 最终评估: 语义得分 {highest_score:.3f} vs 动态及格线 {dynamic_threshold:.3f}\033[0m")
            self._last_semantic_query = query

        # 终极裁决
        self._last_best_route = best_route
        if highest_score >= dynamic_threshold:
            return "small"

        if query != self._last_intercept_query:
            print(f"\033[35m[Router 拦截] 小模型得分不足以抵抗长文本衰减，升级为大模型！\033[0m")
            self._last_intercept_query = query
        return "large"