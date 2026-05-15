from typing import Dict, List, Optional

from runtime.memory import FailureMemory
from runtime.trace import Trace
from runtime.semantic_router import SemanticRouter


class Router:
    TASK_TYPE_KEYWORDS = {
        "coding": ["code", "debug", "refactor", "implement", "function", "class", "api", "python", "javascript"],
        "architecture": ["architecture", "design", "system", "tradeoff", "pattern", "distributed"],
        "optimization": ["optimize", "performance", "benchmark", "efficiency", "faster"],
        "analysis": ["analysis", "analyze", "explain", "compare", "pros", "cons"],
        "workflow": ["step", "plan", "how", "process", "workflow", "guide"],
    }

    COMPLEXITY_MARKERS = [
        "architecture", "tradeoff", "benchmark", "optimize", "analysis", "design",
        "distributed", "security", "algorithm", "compare", "evaluate",
    ]

    def __init__(self, memory: FailureMemory, trace: Trace, use_semantic_routing: bool = True) -> None:
        self.memory = memory
        self.trace = trace
        self.use_semantic_routing = use_semantic_routing
        self.semantic_router = SemanticRouter() if use_semantic_routing else None

    def route(self, query: str, memory_stats: Optional[Dict] = None) -> Dict[str, str]:
        """
        多因子智能路由决策
        
        因子权重：
        - 失败率 (Task-specific): 0.3
        - 全局失败率: 0.2
        - 语义复杂度: 0.3
        - 长度/标记复杂度: 0.2
        
        决策阈值：总分 >= 0.55 → 选择 "large"
        """
        q = query.strip()
        q_lower = q.lower()

        # 提取任务类型
        task_type = self.extract_task_type(q)
        task_failure_rate = self.get_failure_rate(task_type)
        global_failure_rate = self.memory.get_failure_rate(q)

        # 获取语义复杂度分数
        if self.use_semantic_routing and self.semantic_router:
            semantic_score = self.semantic_router.predict_complexity(q)
            self.trace.log(
                "ROUTER",
                f"semantic_complexity={semantic_score:.3f}",
                {"task_type": task_type},
            )
        else:
            semantic_score = 0.5

        # 长度和复杂标记检查
        is_complex = len(q) > 160 or any(k in q_lower for k in self.COMPLEXITY_MARKERS)
        is_long = len(q) > 260

        # 多因子加权决策
        decision_score = 0.0
        decision_factors = []

        # 因子1：任务特定失败率（权重 0.3）
        if task_failure_rate >= 0.50:
            decision_score += 0.3
            decision_factors.append(f"task_failure={task_failure_rate:.0%}")

        # 因子2：全局失败率（权重 0.2）
        if global_failure_rate >= 0.40:
            decision_score += 0.2
            decision_factors.append(f"global_failure={global_failure_rate:.0%}")

        # 因子3：语义复杂度（权重 0.3）
        decision_score += semantic_score * 0.3
        if semantic_score > 0.7:
            decision_factors.append(f"semantic_hard={semantic_score:.0%}")
        elif semantic_score < 0.3:
            decision_factors.append(f"semantic_easy={semantic_score:.0%}")

        # 因子4：长度和结构复杂度（权重 0.2）
        if is_long and is_complex:
            decision_score += 0.2
            decision_factors.append("very_long_complex")
        elif is_complex:
            decision_score += 0.1
            decision_factors.append("complex_markers")

        # 最终路由决策
        decision = "large" if decision_score >= 0.55 else "small"
        reason = (
            " | ".join(decision_factors) if decision_factors else "default_cost_first"
        )

        self.trace.log(
            "ROUTER",
            f"selected_model={decision}",
            {
                "task_type": task_type,
                "decision_score": round(decision_score, 3),
                "semantic_score": round(semantic_score, 3),
                "task_failure_rate": round(task_failure_rate, 3),
                "reason": reason,
            },
        )

        return {
            "target_model": decision,
            "reason": reason,
            "semantic_score": semantic_score,
            "decision_score": decision_score,
        }

    def extract_task_type(self, query: str) -> str:
        q_lower = query.lower()
        scores = {}

        for task_type, keywords in self.TASK_TYPE_KEYWORDS.items():
            match_count = sum(1 for kw in keywords if kw in q_lower)
            scores[task_type] = match_count

        if not scores or max(scores.values()) == 0:
            return "generic"

        return max(scores, key=scores.get)

    def get_failure_rate(self, task_type: str) -> float:
        if task_type == "generic":
            return 0.0

        data = self.memory.load_memory()
        failures = data.get("failures", [])
        if not failures:
            return 0.0

        task_failures = 0
        for record in failures:
            tags = set(record.get("query_tags", []))
            task_keywords = set(self.TASK_TYPE_KEYWORDS.get(task_type, []))
            if tags.intersection(task_keywords):
                task_failures += 1

        if task_failures == 0:
            return 0.0

        return min(1.0, task_failures / max(2, len(failures)))
