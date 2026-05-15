from typing import List

from models.model_manager import ModelManager
from runtime.trace import Trace


class EscalationManager:
    def __init__(self, model_manager: ModelManager, trace: Trace) -> None:
        self.model_manager = model_manager
        self.trace = trace

    def retry_with_large_model(self, query: str, small_answer: str, reasons: List[str]) -> str:
        """
        Escalate failed small-model answer to large model with full repair context.
        
        Args:
            query: Original user query
            small_answer: Failed response from small model
            reasons: List of reasons why the answer was judged BAD
        
        Returns:
            Repaired answer from large model
        """
        # Build repair prompt with full context
        repair_prompt = (
            "You are improving a weak draft answer.\n"
            f"User Query: {query}\n"
            f"Weak Draft: {small_answer}\n"
            f"Quality Issues: {', '.join(reasons) if reasons else 'none'}\n"
            "Return an improved, concrete, and complete answer."
        )

        self.trace.log(
            "ESCALATION",
            "escalating to large model with repair context",
            {"query_len": len(query), "reasons": reasons},
        )
        
        # Invoke large model with repair context
        repaired_answer = self.model_manager.generate("large", repair_prompt)
        
        # Log successful repair completion
        self.trace.log(
            "ESCALATION",
            "large model repair completed",
            {"repaired_len": len(repaired_answer)},
        )
        
        return repaired_answer
