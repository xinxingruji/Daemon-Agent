import json
import math
import os
import urllib.request
from typing import List, Dict
from utterances import SMALL, LARGE

class Claude_Router:
    def __init__(self, threshold: float = 0.45, mistake_threshold: float = 0.85, mistake_file: str = "mistakes.json", safe_tokens: int = 3000, penalty_step: int = 4000, max_mistakes: int = 200):
        self.threshold = threshold
        self.mistake_threshold = mistake_threshold # 错题拦截的相似度要求更严苛一点
        self.mistake_file = mistake_file

        self.safe_tokens = safe_tokens      # 安全区：在这之下不作惩罚
        self.penalty_step = penalty_step    # 惩罚步长：每超这个数，要求就变严苛
        self.penalty_rate = 0.05

        self.max_mistakes = max_mistakes

        self.model_name = "nomic-embed-text-v2-moe"
        self.api_url = "http://localhost:11434/api/embeddings"
        
        print(f"[Router] 初始化零依赖向量大脑...")
        
        # 1. 正常的路由航线（种子语句在 utterances.py 中维护）
        self.routes = {"small": SMALL, "large": LARGE}

        # 2. 加载错题本记录
        self.mistake_book = self._load_mistakes()
        if self.mistake_book:
            print(f"[Router] 📚 已加载 {len(self.mistake_book)} 条错题记录。")

        # 3. 预计算常规种子向量
        self.route_embeddings = {"small": [], "large": []}
        total = sum(len(v) for v in self.routes.values())
        done = 0
        for route_name, utterances in self.routes.items():
            for text in utterances:
                vec = self._get_embedding(text)
                if vec:
                    self.route_embeddings[route_name].append(vec)
                done += 1
                pct = done * 100 // total
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r\033[K  [Router] 加载向量: |{bar}| {pct}% ({done}/{total})", end="", flush=True)
        print()

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

    def route(self, query: str, total_tokens: int = 0, force_large: bool = False) -> str:
        """核心路由：结合了错题本拦截与动态 Token 惩罚"""
        if force_large:
            return "large"
            
        if not query.strip():
            return "large"

        query_vector = self._get_embedding(query)
        if not query_vector:
            return "large"

        # ==========================================
        # 🚨 第一道防线：检查错题本
        # ==========================================
        for mistake in self.mistake_book:
            sim = self._cosine_similarity(query_vector, mistake["vector"])
            if sim >= self.mistake_threshold:
                print(f"\033[31m[Router 警报] 触发错题拦截！强制拉起大模型！\033[0m")
                return "large"

        # ==========================================
        # 🟢 第二道防线：常规语义评估
        # ==========================================
        best_route = "large"
        highest_score = 0.0

        for route_name, embeddings in self.route_embeddings.items():
            for emb in embeddings:
                score = self._cosine_similarity(query_vector, emb)
                if score > highest_score:
                    highest_score = score
                    best_route = route_name

        # [优化点] 如果判定为大模型任务，或者最高分数连基础及格线都没过，直接扔给大模型
        if best_route == "large" or highest_score < self.threshold:
            print(f"\033[36m[SemanticRouter] 匹配分数: {highest_score:.3f} -> 判定为大型任务或未达基础线，路由至: large\033[0m")
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
            
            print(f"\033[33m[Router 测算] 上下文较长 ({total_tokens} tokens)，小模型及格线已从 {self.threshold} 动态上调至 {dynamic_threshold:.3f}\033[0m")

        print(f"\033[36m[SemanticRouter] 最终评估: 语义得分 {highest_score:.3f} vs 动态及格线 {dynamic_threshold:.3f}\033[0m")
        
        # 终极裁决
        if highest_score >= dynamic_threshold:
            return "small"
            
        print(f"\033[35m[Router 拦截] 小模型得分不足以抵抗长文本衰减，升级为大模型！\033[0m")
        return "large"