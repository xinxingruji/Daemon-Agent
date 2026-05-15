import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple


class FailureMemory:
    STOPWORDS = {
        "the", "and", "for", "that", "with", "from", "this", "what", "when", "where",
        "which", "your", "about", "into", "have", "will", "would", "could", "should",
        "how", "why", "are", "is", "was", "were", "can", "please", "show", "give",
    }

    def __init__(self, path: str = "data/failure_memory.json") -> None:
        self.path = path
        self._ensure_file()

    def _ensure_file(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"failures": [], "metadata": {"version": 2}}, f, indent=2)

    def load_memory(self) -> Dict:
        self._ensure_file()
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "failures" not in data:
            data["failures"] = []
        if "metadata" not in data:
            data["metadata"] = {"version": 2}
        return data

    def save_failure(
        self,
        query: str,
        task_type: str,
        selected_model: str,
        reason: List[str],
        small_answer: Optional[str] = None,
        final_answer: Optional[str] = None,
    ) -> None:
        data = self.load_memory()
        query_preview = query[:100] if len(query) > 100 else query
        record = {
            "timestamp": int(time.time()),
            "task_type": task_type,
            "selected_model": selected_model,
            "query_preview": query_preview,
            "reason": reason,
        }
        if small_answer:
            record["small_answer"] = small_answer
        if final_answer:
            record["final_answer"] = final_answer

        data["failures"].append(record)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)

    def get_failure_rate(
        self,
        query: str = "",
        task_type: Optional[str] = None,
        model: Optional[str] = None,
    ) -> float:
        data = self.load_memory()
        failures = data.get("failures", [])
        if not failures:
            return 0.0

        matching = []
        for record in failures:
            if task_type and record.get("task_type") != task_type:
                continue
            if model and record.get("selected_model") != model:
                continue
            matching.append(record)

        if not matching:
            return 0.0

        return min(1.0, len(matching) / max(2, len(failures)))

    def summarize_memory(self) -> Dict:
        data = self.load_memory()
        failures = data.get("failures", [])

        if not failures:
            return {
                "total_failures": 0,
                "by_task_type": {},
                "by_model": {},
                "by_task_and_model": {},
            }

        total = len(failures)
        by_task_type: Dict[str, Dict[str, int]] = {}
        by_model: Dict[str, int] = {}
        by_task_and_model: Dict[str, Dict[str, int]] = {}

        for record in failures:
            task_type = record.get("task_type", "unknown")
            model = record.get("selected_model", "unknown")
            reason = record.get("reason", [])

            if task_type not in by_task_type:
                by_task_type[task_type] = {"count": 0, "rate": 0.0}
            by_task_type[task_type]["count"] += 1

            if model not in by_model:
                by_model[model] = 0
            by_model[model] += 1

            key = f"{task_type}_{model}"
            if key not in by_task_and_model:
                by_task_and_model[key] = {"count": 0, "rate": 0.0}
            by_task_and_model[key]["count"] += 1

        for task_type in by_task_type:
            by_task_type[task_type]["rate"] = round(
                by_task_type[task_type]["count"] / total, 3
            )

        for key in by_task_and_model:
            by_task_and_model[key]["rate"] = round(
                by_task_and_model[key]["count"] / total, 3
            )

        return {
            "total_failures": total,
            "by_task_type": by_task_type,
            "by_model": by_model,
            "by_task_and_model": by_task_and_model,
        }

    def save_routing_feedback(
        self,
        query: str,
        predicted_complexity: float,
        selected_model: str,
        final_quality: str,
    ) -> None:
        """
        记录路由决策的反馈，用于学习和改进
        
        Args:
            query: 用户查询
            predicted_complexity: 路由器预测的复杂度（0-1）
            selected_model: 选择的模型（"small"或"large"）
            final_quality: 最终答案质量（"GOOD"或"BAD"）
        """
        data = self.load_memory()

        if "routing_feedback" not in data:
            data["routing_feedback"] = []

        # 判断路由决策是否正确
        was_correct = False
        if selected_model == "small" and final_quality == "GOOD":
            was_correct = True  # 小模型成功，节省成本
        elif selected_model == "large" and final_quality == "GOOD":
            was_correct = True  # 大模型成功，质量保证
        elif selected_model == "small" and final_quality == "BAD":
            was_correct = False  # 小模型失败，应该用大模型
        elif selected_model == "large" and final_quality == "BAD":
            was_correct = False  # 大模型失败（极少见）

        record = {
            "timestamp": int(time.time()),
            "query_preview": query[:50],
            "predicted_complexity": round(predicted_complexity, 3),
            "selected_model": selected_model,
            "final_quality": final_quality,
            "was_correct": was_correct,
        }

        data["routing_feedback"].append(record)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)

    def get_routing_stats(self) -> Dict:
        """
        获取路由器的准确度统计
        用于评估路由决策的质量
        """
        data = self.load_memory()
        feedback = data.get("routing_feedback", [])

        if not feedback:
            return {
                "accuracy": 0.0,
                "total_decisions": 0,
                "by_model": {},
                "trend": "insufficient_data",
            }

        # 总体准确度
        correct = sum(1 for f in feedback if f.get("was_correct"))
        total = len(feedback)
        accuracy = correct / total if total > 0 else 0.0

        # 按模型统计
        small_correct = sum(
            1 for f in feedback
            if f.get("selected_model") == "small" and f.get("was_correct")
        )
        small_total = sum(
            1 for f in feedback if f.get("selected_model") == "small"
        )
        large_correct = sum(
            1 for f in feedback
            if f.get("selected_model") == "large" and f.get("was_correct")
        )
        large_total = sum(
            1 for f in feedback if f.get("selected_model") == "large"
        )

        # 计算趋势（最近10条）
        recent = feedback[-10:]
        recent_correct = sum(1 for f in recent if f.get("was_correct"))
        trend = "improving" if recent_correct / len(recent) > accuracy else "stable"

        return {
            "accuracy": round(accuracy, 3),
            "total_decisions": total,
            "recent_trend": trend,
            "by_model": {
                "small": {
                    "accuracy": round(small_correct / small_total, 3) if small_total > 0 else 0.0,
                    "count": small_total,
                },
                "large": {
                    "accuracy": round(large_correct / large_total, 3) if large_total > 0 else 0.0,
                    "count": large_total,
                },
            },
        }
