from models.model_manager import ModelManager

from runtime.trace import Trace


class Evaluator:
    JUDGE_PROMPT_TEMPLATE = """You are a strict LLM judge.
Evaluate the RESPONSE for the USER_QUERY on four dimensions:
1) task completion
2) correctness
3) hallucination risk
4) missing critical information

If the response is sufficient and reliable, output GOOD.
If it is incomplete, incorrect, hallucinated, or missing key info, output BAD.

Return only one token: GOOD or BAD.

USER_QUERY:
{query}

RESPONSE:
{response}
""".strip()

    LOW_CONFIDENCE_PHRASES = (
        "not sure",
        "i might be wrong",
        "cannot guarantee",
        "uncertain",
        "maybe",
    )

    def __init__(
        self,
        trace: Trace,
        model_manager: ModelManager | None = None,
        judge_model: str = "small",
    ) -> None:
        self.trace = trace
        self.model_manager = model_manager or ModelManager()
        self.judge_model = judge_model

    def evaluate(self, query: str, response: str) -> str:
        """
        评估答案质量，检测多种不自信信号
        返回 "GOOD" 或 "BAD"
        """
        # 1. 检测模型自己标记的不自信
        if self._has_model_uncertainty_marker(response):
            self.trace.log("EVALUATOR", "BAD", {"reason": "model_expressed_uncertainty"})
            return "BAD"
        
        # 2. 检测答案太短或为空
        if self._is_too_short_or_empty(response):
            self.trace.log("EVALUATOR", "BAD", {"reason": "response_too_short_or_empty"})
            return "BAD"
        
        # 3. 使用 LLM-as-Judge 进行最终评估
        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            query=query.strip(),
            response=response.strip(),
        )
        raw = self.model_manager.generate(self.judge_model, judge_prompt)
        label = self._normalize_label(raw, response)

        self.trace.log("EVALUATOR", label)
        return label

    def _has_model_uncertainty_marker(self, response: str) -> bool:
        """检测模型自己标记的不自信信号"""
        uncertainty_markers = [
            "[UNCERTAIN_MODEL_OUTPUT]",
            "[SHORT_RESPONSE]",
            "[LOW_CONFIDENCE]",
        ]
        response_upper = (response or "").upper()
        for marker in uncertainty_markers:
            if marker in response_upper:
                return True
        return False

    def _is_too_short_or_empty(self, response: str) -> bool:
        text = (response or "").strip()
        if not text:
            return True
        # 移除模型标记后检查长度
        clean_text = text
        for marker in ["[UNCERTAIN_MODEL_OUTPUT]", "[SHORT_RESPONSE]", "[LOW_CONFIDENCE]"]:
            clean_text = clean_text.replace(marker, "").strip()
        return len(clean_text.split()) < 8

    def _normalize_label(self, judge_output: str, response: str) -> str:
        normalized = (judge_output or "").strip().upper()
        if normalized.startswith("GOOD"):
            return "GOOD"
        if normalized.startswith("BAD"):
            return "BAD"

        response_lower = response.lower()
        if any(p in response_lower for p in self.LOW_CONFIDENCE_PHRASES):
            return "BAD"

        return "GOOD"
