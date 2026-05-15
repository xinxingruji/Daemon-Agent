"""
语义路由器：基于文本相似度的智能复杂度预测
- 无外部依赖
- 完全本地运行
- 支持动态种子更新
"""

from typing import Dict, List


class SemanticRouter:
    """
    基于语义向量的智能路由
    
    核心逻辑：
    1. 准备"简单"和"复杂"的种子句子
    2. 将用户查询向量化
    3. 计算与种子句子的相似度
    4. 根据相似度分布预测复杂度
    """

    def __init__(self):
        # 种子句子库：代表"简单任务"的语义空间
        self.easy_seeds = [
            "What is Python?",
            "How do I print hello world?",
            "How to define a function?",
            "What is a variable?",
            "How to use a loop?",
            "Explain a list in Python",
            "What is a string?",
            "How to create a dictionary?",
            "Basic syntax of Python",
            "How to read a file?",
        ]

        # 种子句子库：代表"复杂任务"的语义空间
        self.hard_seeds = [
            "How to optimize distributed system garbage collection?",
            "Explain consensus algorithms in distributed systems",
            "How to implement a custom memory allocator?",
            "Design a high-performance key-value store",
            "How to do CPU cache optimization?",
            "Explain the CAP theorem and its implications",
            "Implement a thread-safe concurrent data structure",
            "Design a machine learning pipeline for production",
            "How to scale a relational database?",
            "Implement Byzantine fault tolerance protocol",
        ]

        # 预计算种子的特征向量（缓存）
        self.easy_vectors = [self._text_to_vector(s) for s in self.easy_seeds]
        self.hard_vectors = [self._text_to_vector(s) for s in self.hard_seeds]

    def _text_to_vector(self, text: str) -> Dict[str, float]:
        """
        简单的向量化：词频-反向文档频率的简化版本
        
        算法：
        1. 分词并转小写
        2. 计算词频
        3. L2归一化
        
        完全本地实现，无外部依赖
        """
        words = text.lower().split()
        vector = {}

        for word in words:
            # 去除标点符号
            word = "".join(c for c in word if c.isalnum())
            if word:
                vector[word] = vector.get(word, 0) + 1

        # L2 归一化
        norm = sum(v**2 for v in vector.values()) ** 0.5
        if norm > 0:
            vector = {k: v / norm for k, v in vector.items()}

        return vector

    def _cosine_similarity(
        self, vec1: Dict[str, float], vec2: Dict[str, float]
    ) -> float:
        """
        计算两个向量的余弦相似度
        
        公式：similarity = (A · B) / (||A|| * ||B||)
        由于向量已归一化，直接计算点积即可
        """
        if not vec1 or not vec2:
            return 0.0

        # 计算点积
        dot_product = sum(
            vec1.get(word, 0) * vec2.get(word, 0)
            for word in set(vec1.keys()) | set(vec2.keys())
        )

        return dot_product

    def predict_complexity(self, query: str) -> float:
        """
        预测查询的复杂度（0.0 - 1.0）

        算法：
        1. 将查询向量化
        2. 计算与"简单"种子的平均相似度 → easy_score
        3. 计算与"复杂"种子的平均相似度 → hard_score
        4. 返回: (hard_score - easy_score) / 2 + 0.5

        结果解释：
        - 0.0-0.3: 明确简单
        - 0.3-0.7: 中等难度
        - 0.7-1.0: 明确复杂
        """
        query_vector = self._text_to_vector(query)

        if not query_vector:
            # 空查询，返回中等难度
            return 0.5

        # 与简单任务的相似度
        easy_similarities = [
            self._cosine_similarity(query_vector, seed_vec)
            for seed_vec in self.easy_vectors
        ]
        easy_score = (
            sum(easy_similarities) / len(easy_similarities)
            if easy_similarities
            else 0.0
        )

        # 与复杂任务的相似度
        hard_similarities = [
            self._cosine_similarity(query_vector, seed_vec)
            for seed_vec in self.hard_vectors
        ]
        hard_score = (
            sum(hard_similarities) / len(hard_similarities)
            if hard_similarities
            else 0.0
        )

        # 标准化到 0-1
        complexity = (hard_score - easy_score) / 2 + 0.5
        return max(0.0, min(1.0, complexity))

    def add_easy_seed(self, text: str) -> None:
        """动态添加简单任务的种子"""
        if text not in self.easy_seeds:
            self.easy_seeds.append(text)
            self.easy_vectors.append(self._text_to_vector(text))

    def add_hard_seed(self, text: str) -> None:
        """动态添加复杂任务的种子"""
        if text not in self.hard_seeds:
            self.hard_seeds.append(text)
            self.hard_vectors.append(self._text_to_vector(text))

    def debug_similarity(self, query: str) -> Dict:
        """调试：查看查询与各种子的相似度分布"""
        query_vector = self._text_to_vector(query)

        easy_sims = [
            (seed, self._cosine_similarity(query_vector, vec))
            for seed, vec in zip(self.easy_seeds, self.easy_vectors)
        ]
        hard_sims = [
            (seed, self._cosine_similarity(query_vector, vec))
            for seed, vec in zip(self.hard_seeds, self.hard_vectors)
        ]

        return {
            "complexity_score": self.predict_complexity(query),
            "easy_similarities": sorted(easy_sims, key=lambda x: x[1], reverse=True)[:3],
            "hard_similarities": sorted(hard_sims, key=lambda x: x[1], reverse=True)[:3],
        }
