from typing import Dict

from models.model_manager import ModelManager
from runtime.escalation import EscalationManager
from runtime.evaluator import Evaluator
from runtime.memory import FailureMemory
from runtime.router import Router
from runtime.trace import Trace


class AdaptiveInferenceRuntime:
    def __init__(self, memory_path: str = "data/failure_memory.json", enable_trace: bool = True) -> None:
        self.trace = Trace(enabled=enable_trace)
        self.memory = FailureMemory(path=memory_path)
        self.model_manager = ModelManager()
        self.router = Router(memory=self.memory, trace=self.trace)
        self.evaluator = Evaluator(trace=self.trace)
        self.escalation = EscalationManager(model_manager=self.model_manager, trace=self.trace)

    def run(self, query: str) -> Dict:
        self.trace.log("RUNTIME", "received query", {"query": query})

        # ROUTER stage
        memory_stats = self.memory.summarize_memory()
        route = self.router.route(query, memory_stats=memory_stats)
        target = route["target_model"]
        task_type = self.router.extract_task_type(query)

        # EXECUTION stage: run the selected model
        self.trace.log("EXECUTION", f"invoking model={target}", {"route_reason": route["reason"]})
        answer = self.model_manager.generate(target, query)

        # If routed directly to large model, return immediately
        if target == "large":
            self.trace.log("EXECUTION", "served by large model", {"task_type": task_type})
            self.trace.log("RUNTIME", "returning final answer")
            
            # Record routing feedback
            semantic_score = route.get("semantic_score", 0.5)
            self.memory.save_routing_feedback(
                query=query,
                predicted_complexity=semantic_score,
                selected_model="large",
                final_quality="GOOD",  # 大模型直接返回认为GOOD
            )
            
            self.trace.print_runtime_summary()
            return {
                "answer": answer,
                "meta": {
                    "path": "router->large",
                    "escalated": False,
                    "route_reason": route["reason"],
                    "task_type": task_type,
                },
            }

        # EVALUATOR stage
        self.trace.log("EVALUATOR", "evaluating small-model output", {"task_type": task_type})
        eval_label = self.evaluator.evaluate(query, answer)

        if eval_label == "GOOD":
            self.trace.log("RUNTIME", "small model answer accepted", {"task_type": task_type})
            
            # Record routing feedback: small model succeeded
            semantic_score = route.get("semantic_score", 0.5)
            self.memory.save_routing_feedback(
                query=query,
                predicted_complexity=semantic_score,
                selected_model="small",
                final_quality="GOOD",
            )
            
            self.trace.print_runtime_summary()
            return {
                "answer": answer,
                "meta": {
                    "path": "router->small->evaluator",
                    "escalated": False,
                    "quality": eval_label,
                    "task_type": task_type,
                },
            }

        # ESCALATION stage: retry with large model
        self.trace.log("ESCALATION", "escalating to large model", {"task_type": task_type})
        failure_reasons = ["judge_bad"]
        final_answer = self.escalation.retry_with_large_model(
            query=query,
            small_answer=answer,
            reasons=failure_reasons,
        )

        # MEMORY stage: record failure experience
        self.memory.save_failure(
            query=query,
            task_type=task_type,
            selected_model=target,
            reason=failure_reasons,
            small_answer=answer,
            final_answer=final_answer,
        )
        self.trace.log("MEMORY", "failure case saved", {"task_type": task_type, "reason": failure_reasons})
        
        # Record routing feedback: small model failed, needed escalation
        semantic_score = route.get("semantic_score", 0.5)
        routing_stats = self.memory.get_routing_stats()
        self.memory.save_routing_feedback(
            query=query,
            predicted_complexity=semantic_score,
            selected_model="small",
            final_quality="BAD",  # 评估为BAD，触发了升级
        )

        self.trace.print_runtime_summary()
        return {
            "answer": final_answer,
            "meta": {
                "path": "router->small->evaluator->escalation->large",
                "escalated": True,
                "quality": eval_label,
                "task_type": task_type,
                "failure_saved": True,
                "routing_accuracy": routing_stats["accuracy"],
            },
        }
