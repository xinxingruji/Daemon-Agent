import json
import os
import sys
import threading
from typing import Dict, List, Mapping, Sequence

# 重配 stdout 编码，防止 UTF-8 内容打印到 GBK 终端时 UnicodeEncodeError
# 必须在任何 print() 之前执行，所以放在 router.py 最顶部
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from utterances import SMALL, LARGE
from seed_cache import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_URL,
    build_seed_cache_document,
    validate_seed_cache,
)
from embedding_service import DEFAULT_EMBEDDING_CACHE_SIZE, EmbeddingProvider
from router_compression import (
    atomic_write_json,
    atomic_write_jsonl,
    build_compressed_records,
    parse_compression_response,
)
from routing_policy import RoutingPolicy, cosine_similarity

class Claude_Router:
    def __init__(self, threshold: float = 0.45, mistake_threshold: float = 0.75,
                 mistake_file: str = "mistakes.json", seed_file: str = "seed_vectors.json",
                 safe_tokens: int = 3000, penalty_step: int = 4000, max_mistakes: int = 200,
                 model_name: str = DEFAULT_EMBEDDING_MODEL,
                 api_url: str = DEFAULT_EMBEDDING_URL,
                 embedding_cache_size: int = DEFAULT_EMBEDDING_CACHE_SIZE,
                 embedding_provider: EmbeddingProvider | None = None):
        self.threshold = threshold
        self.mistake_threshold = mistake_threshold
        self.mistake_file = mistake_file
        self.seed_file = seed_file

        self.safe_tokens = safe_tokens
        self.penalty_step = penalty_step
        self.penalty_rate = 0.05

        self.max_mistakes = max_mistakes

        self.model_name = model_name
        self.api_url = api_url
        self.embedding_provider = embedding_provider or EmbeddingProvider(
            model_name=model_name,
            api_url=api_url,
            cache_size=embedding_cache_size,
        )
        self.routing_policy = RoutingPolicy(
            threshold=threshold,
            mistake_threshold=mistake_threshold,
            safe_tokens=safe_tokens,
            penalty_step=penalty_step,
            penalty_rate=self.penalty_rate,
        )
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
                report = validate_seed_cache(
                    data,
                    expected_model=self.model_name,
                    source_routes=self.routes,
                )
                if not report.valid:
                    details = "; ".join(report.errors)
                    raise ValueError(
                        f"种子缓存不匹配: {details}. "
                        "请运行 python precompute_seeds.py 重新生成",
                    )
                for warning in report.warnings:
                    print(f"[Router 警告] {warning}")
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
                print(f"[Router] 读取 {self.seed_file} 失败: {e}，尝试从 Ollama 重建")

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
        expected = total
        loaded = sum(len(items) for items in self.route_embeddings.values())
        if loaded == expected:
            # 仅在全部种子成功时写缓存，避免用不完整结果覆盖旧文件。
            self._save_seed_vectors()
        else:
            self.route_embeddings_text = {"small": [], "large": []}
            self.route_embeddings = {"small": [], "large": []}
            print(
                f"[Router 错误] 仅生成 {loaded}/{expected} 条种子向量，"
                "未写入缓存；自动路由将保守使用 large。",
            )

    def _build_seed_document(
        self,
        texts: Mapping[str, Sequence[str]],
        embeddings: Mapping[str, Sequence[Sequence[float]]],
    ) -> dict:
        entries = {"small": [], "large": []}
        for route_name in ("small", "large"):
            route_texts = texts.get(route_name, ())
            route_vectors = embeddings.get(route_name, ())
            if len(route_texts) != len(route_vectors):
                raise ValueError(f"{route_name} seed text/vector counts do not match")
            for text, vec in zip(route_texts, route_vectors):
                entries[route_name].append({"text": text, "vector": vec})
        return build_seed_cache_document(
            entries,
            embedding_model=self.model_name,
            source_routes=self.routes,
        )

    def _save_seed_vectors_data(
        self,
        texts: Mapping[str, Sequence[str]],
        embeddings: Mapping[str, Sequence[Sequence[float]]],
    ) -> None:
        document = self._build_seed_document(texts, embeddings)
        atomic_write_json(self.seed_file, document)

    def _save_seed_vectors(self):
        """Atomically persist the current validated seed vectors."""
        self._save_seed_vectors_data(
            self.route_embeddings_text,
            self.route_embeddings,
        )

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
        with self.seed_lock:
            self.route_embeddings = {"small": [], "large": []}
            self.route_embeddings_text = {"small": [], "large": []}
            self._load_seed_vectors()
        print("[Router] 种子库已热更新")

    def embedding_cache_stats(self) -> dict[str, int | float]:
        return self.embedding_provider.stats().to_dict()

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
        
    def _embedding_dimension(self) -> int:
        for route_name in ("small", "large"):
            for vector in self.route_embeddings.get(route_name, ()):
                if vector:
                    return len(vector)
        for record in self.mistake_book:
            vector = record.get("vector", [])
            if vector:
                return len(vector)
        raise ValueError("cannot compress without a known embedding dimension")

    @staticmethod
    def _compression_prompt(target: str, snapshot: list) -> str:
        if target == "mistake":
            queries_text = "\n".join(f"- {item['query']}" for item in snapshot)
            return f"""
            以下是导致小型AI模型失败的指令清单：
            {queries_text}
            请将这些指令抽象并合并为 10-20 个涵盖这些核心难点的通用指令。
            请严格以 JSON 数组的格式输出纯字符串列表（不要有Markdown代码块格式）。
            """
        queries_text = "\n".join(f"- {query}" for query in snapshot)
        return f"""
        以下是小型AI模型近期成功处理的 {len(snapshot)} 个具体任务指令：
        {queries_text}
        请提取它们背后的核心意图，泛化为 5 到 8 个代表性的通用指令。
        请严格以 JSON 数组的格式输出纯字符串列表（不要有Markdown代码块格式）。
        """

    def _apply_compressed_mistakes(self, snapshot: list, records: list[dict]) -> None:
        with self.mistake_lock:
            if self.mistake_book[:len(snapshot)] != snapshot:
                raise RuntimeError("mistake snapshot changed during compression")
            new_arrivals = self.mistake_book[len(snapshot):]
            candidate = records + new_arrivals
            atomic_write_jsonl(self.mistake_file, candidate)
            self.mistake_book = candidate

    def _apply_compressed_seeds(self, snapshot: list, records: list[dict]) -> None:
        with self.seed_lock:
            base_count = self.base_small_count
            snapshot_count = len(snapshot)
            current_snapshot = self.route_embeddings_text["small"][
                base_count:base_count + snapshot_count
            ]
            if current_snapshot != snapshot:
                raise RuntimeError("seed snapshot changed during compression")
            candidate_texts = {
                name: list(values)
                for name, values in self.route_embeddings_text.items()
            }
            candidate_vectors = {
                name: [list(vector) for vector in values]
                for name, values in self.route_embeddings.items()
            }
            candidate_texts["small"] = (
                candidate_texts["small"][:base_count]
                + [record["query"] for record in records]
                + candidate_texts["small"][base_count + snapshot_count:]
            )
            candidate_vectors["small"] = (
                candidate_vectors["small"][:base_count]
                + [record["vector"] for record in records]
                + candidate_vectors["small"][base_count + snapshot_count:]
            )
            self._save_seed_vectors_data(candidate_texts, candidate_vectors)
            self.route_embeddings_text = candidate_texts
            self.route_embeddings = candidate_vectors

    def _trigger_compression_async(self, target: str):
        """Compress one consistent snapshot and replace it only after validation."""
        if target == "mistake":
            with self.mistake_lock:
                if self.is_compressing_mistakes:
                    return
                self.is_compressing_mistakes = True
                snapshot = self.mistake_book.copy()
        elif target == "seed":
            with self.seed_lock:
                if self.is_compressing_seeds:
                    return
                self.is_compressing_seeds = True
                snapshot = self.route_embeddings_text["small"][self.base_small_count:].copy()
        else:
            raise ValueError(f"unsupported compression target: {target}")

        def _compress_task():
            try:
                from config import get_client

                print(
                    f"\n[Router ⚙️] 启动后台 LLM {target} 压缩机制 "
                    f"(处理 {len(snapshot)} 条数据)...",
                )
                client = get_client()
                response = client.messages.create(
                    model="large",
                    messages=[{
                        "role": "user",
                        "content": self._compression_prompt(target, snapshot),
                    }],
                    max_tokens=1000,
                )
                if not response.content or not hasattr(response.content[0], "text"):
                    raise ValueError("compression model returned no text content")
                abstract_queries = parse_compression_response(
                    response.content[0].text,
                    target,
                )
                compressed_records = build_compressed_records(
                    abstract_queries,
                    embedding_getter=self._get_embedding,
                    expected_dimension=self._embedding_dimension(),
                )

                if target == "mistake":
                    self._apply_compressed_mistakes(snapshot, compressed_records)
                    current_size = len(self.mistake_book)
                else:
                    self._apply_compressed_seeds(snapshot, compressed_records)
                    current_size = len(self.route_embeddings["small"])
                print(
                    f"[Router ✅] {target} 压缩通过完整校验并原子写回，"
                    f"当前容量: {current_size}",
                )
            except Exception as e:
                print(f"\n[Router ❌] 后台 {target} 压缩失败 (保持原有状态): {e}")
            finally:
                if target == "mistake":
                    with self.mistake_lock:
                        self.is_compressing_mistakes = False
                else:
                    with self.seed_lock:
                        self.is_compressing_seeds = False

        try:
            threading.Thread(target=_compress_task, daemon=True).start()
        except Exception:
            if target == "mistake":
                with self.mistake_lock:
                    self.is_compressing_mistakes = False
            else:
                with self.seed_lock:
                    self.is_compressing_seeds = False
            raise

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedding_provider.get(text)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        return cosine_similarity(vec1, vec2)

    def _sync_routing_policy(self) -> None:
        """Preserve compatibility with callers that tune Router attributes."""
        self.routing_policy.threshold = self.threshold
        self.routing_policy.mistake_threshold = self.mistake_threshold
        self.routing_policy.safe_tokens = self.safe_tokens
        self.routing_policy.penalty_step = self.penalty_step
        self.routing_policy.penalty_rate = self.penalty_rate

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

        with self.mistake_lock:
            mistake_vectors = [
                item.get("vector", [])
                for item in self.mistake_book
                if isinstance(item, dict)
            ]
        with self.seed_lock:
            route_embeddings = {
                name: list(vectors)
                for name, vectors in self.route_embeddings.items()
            }
        self._sync_routing_policy()
        decision = self.routing_policy.decide(
            query_vector,
            route_embeddings=route_embeddings,
            mistake_vectors=mistake_vectors,
            total_tokens=total_tokens,
        )
        self._last_route_scores = decision.route_scores
        self._last_best_route = decision.best_route

        if decision.intercepted_by_mistake:
            if query != self._last_alert_query:
                print("\033[31m[Router 警报] 触发错题拦截！强制拉起大模型！\033[0m")
                self._last_alert_query = query
            return "large"

        if decision.best_route == "large" or decision.highest_score < self.threshold:
            if query != self._last_semantic_query:
                print(
                    f"\033[36m[SemanticRouter] 匹配分数: "
                    f"{decision.highest_score:.3f} -> 判定为大型任务或未达基础线，"
                    "路由至: large\033[0m",
                )
                self._last_semantic_query = query
            return "large"

        if decision.dynamic_threshold > self.threshold:
            print(
                f"\033[33m[Router 测算] 上下文较长 ({total_tokens} tokens)，"
                f"小模型及格线已从 {self.threshold} 动态上调至 "
                f"{decision.dynamic_threshold:.3f}\033[0m",
            )

        if query != self._last_semantic_query:
            print(
                f"\033[36m[SemanticRouter] 最终评估: 语义得分 "
                f"{decision.highest_score:.3f} vs 动态及格线 "
                f"{decision.dynamic_threshold:.3f}\033[0m",
            )
            self._last_semantic_query = query

        if decision.route == "small":
            return "small"

        if decision.intercepted_by_context and query != self._last_intercept_query:
            print("\033[35m[Router 拦截] 小模型得分不足以抵抗长文本衰减，升级为大模型！\033[0m")
            self._last_intercept_query = query
        return "large"
