import json
import os
import urllib.request
from typing import Optional


def _resolve_config_path() -> str:
    """查找项目根目录的 config.json"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "config.json")


def _read_config(key: str, default: str = "") -> str:
    """从 config.json 读取配置项"""
    try:
        path = _resolve_config_path()
        with open(path, "r") as f:
            cfg = json.load(f)
        return cfg.get(key, "") or ""
    except Exception:
        return default


class SmallModel:
    """小型本地模型：优先 Ollama (qwen2.5:0.5b)，失败降级 Mock"""

    HARD_KEYWORDS = {
        "debug", "code", "python", "javascript", "architecture",
        "distributed", "optimization", "security", "benchmark"
    }

    UNCERTAINTY_MARKERS = [
        "i am not sure", "i'm not sure", "as an ai", "i don't have enough context",
        "i cannot fulfill", "i apologize", "unable to", "not qualified",
        "抱歉", "我不确定", "作为一个AI", "无法确定"
    ]

    def __init__(
        self,
        model_name: str = "qwen2.5:0.5b",
        base_url: str = "http://localhost:11434/api/chat",
        fallback_to_mock: bool = True,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.fallback_to_mock = fallback_to_mock

        # 连接测试：Ollama → Mock 自动降级
        if self._test_connection():
            self.use_ollama = True
            self.name = f"ollama-{model_name}"
            print(f"[SmallModel] Ollama connected: {model_name}")
        else:
            self.use_ollama = False
            self.name = "mock-small-model"
            if fallback_to_mock:
                print(f"[SmallModel] Ollama not responding. Falling back to mock mode.")
            else:
                print(f"[SmallModel] WARNING: Ollama unavailable and mock is disabled.")

    def _test_connection(self) -> bool:
        """极简请求测试 Ollama 是否在线"""
        try:
            return self._call_ollama("hello", max_tokens=5) is not None
        except Exception:
            return False

    def _call_ollama(self, query: str, temperature: float = 0.3, max_tokens: int = 512) -> Optional[str]:
        """原生 HTTP 调用 Ollama API"""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": query}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["message"]["content"]
        except Exception as e:
            print(f"[SmallModel Error] Ollama request failed: {e}")
            return None

    def generate(self, query: str) -> str:
        if self.use_ollama:
            return self._generate_with_ollama(query)
        else:
            return self._generate_with_mock(query)

    def _generate_with_ollama(self, query: str) -> str:
        """Ollama 推理 + 自我校验"""
        answer = self._call_ollama(query)

        if answer is None:
            if self.fallback_to_mock:
                return self._generate_with_mock(query)
            return "[System Error] Ollama failed to respond."

        # 校验 1：模型是否表达了不自信？
        if self._detect_uncertainty(answer):
            return f"[UNCERTAIN_MODEL_OUTPUT] {answer}"

        # 校验 2：长问题的回答过短
        if len(answer) < 20 and len(query) > 50:
            return f"[SHORT_RESPONSE] {answer}"

        return answer

    def _generate_with_mock(self, query: str) -> str:
        """Ollama 不可用时的退化策略"""
        q_lower = query.lower()
        if any(kw in q_lower for kw in self.HARD_KEYWORDS):
            return self._generate_low_confidence_response("hard_keyword_detected")
        if len(query) > 100:
            return self._generate_low_confidence_response("query_too_long")
        if query.count("?") > 1:
            return self._generate_low_confidence_response("multiple_questions")
        return self._generate_high_confidence_response()

    def _detect_uncertainty(self, response: str) -> bool:
        response_lower = response.lower()
        return any(marker in response_lower for marker in self.UNCERTAINTY_MARKERS)

    def _generate_low_confidence_response(self, reason: str) -> str:
        responses = {
            "hard_keyword_detected": "I'm not entirely sure about this... This might require more specialized knowledge.",
            "query_too_long": "This is quite complex. I might be missing important context.",
            "multiple_questions": "There are multiple aspects here. I might not cover everything correctly.",
        }
        return f"[UNCERTAIN_MODEL_OUTPUT] {responses.get(reason, 'I am not confident.')}"

    def _generate_high_confidence_response(self) -> str:
        return (
            "Use a lightweight runtime with adaptive routing, quality evaluation, "
            "escalation for failures, and memory-based learning."
        )
