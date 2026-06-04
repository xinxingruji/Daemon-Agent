import json
import math
import os
import sys
import urllib.request
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

        # 供外部反馈用的最近一次路由信息
        self._last_query_vector = None
        self._last_route_scores = {"small": 0.0, "large": 0.0}
        self._last_best_route = None

        print(f"[Router] 初始化...")

        # 1. 加载种子向量（优先用预计算缓存）
        self.route_embeddings_text = {"small": [], "large": []}
        self.route_embeddings = {"small": [], "large": []}
        self._load_seed_vectors()

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
        """添加一条新种子，自动调 Ollama 嵌入并持久化"""
        if not text.strip():
            return
        # 检查是否已存在
        if text in self.route_embeddings_text.get(route_name, []):
            return
        vec = self._get_embedding(text)
        if not vec:
            return
        if route_name not in self.route_embeddings:
            self.route_embeddings[route_name] = []
            self.route_embeddings_text[route_name] = []
        self.route_embeddings_text[route_name].append(text)
        self.route_embeddings[route_name].append(vec)
        self._save_seed_vectors()
        print(f"[Router] 已添加种子: '{text}' → {route_name}")

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
        """将导致小模型翻车的 Query 记入错题本"""
        # 如果已在错题本中，跳过（防止同一 query 在循环中重复记录）
        if any(m["query"] == query for m in self.mistake_book):
            return
        print(f"正在将翻车任务记入错题本: '{query}'")
        vec = self._get_embedding(query)
        if not vec:
            return
            
        # 存入内存并写入文件持久化
        record = {"query": query, "vector": vec}

        need_rewrite = False
        if len(self.mistake_book) >= self.max_mistakes:
            self.mistake_book.pop(0)  # 剔除最老的数据
            need_rewrite = True
        # 如果错题本中条目数量超过了max_mistakes，那么就要将最久的那条删掉
        # 此时如果要写入磁盘本地化的话只能整个错题本重新覆写
        
        self.mistake_book.append(record)
        
        # 写入文件持久化
        if need_rewrite:
            # 如果踢掉了老数据，需要覆盖式重写整个文件 ('w' 模式)
            with open(self.mistake_file, 'w', encoding='utf-8') as f:
                for r in self.mistake_book:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            # 否则只需在文件末尾追加 ('a' 模式)，保持最高性能
            with open(self.mistake_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

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