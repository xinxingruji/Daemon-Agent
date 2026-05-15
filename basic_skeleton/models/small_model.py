import json
import urllib.request
from typing import Dict, Any

class SmallModel:
    """轻量级模型：原生本地推理 + 降级检测 + 零依赖"""
    
    # 已知难以处理的任务关键词（作为第一道防线）
    HARD_KEYWORDS = {
        "debug", "code", "python", "javascript", "architecture", 
        "distributed", "optimization", "security", "benchmark"
    }

    # 模型不自信的表现特征（作为生成后的检测）
    UNCERTAINTY_MARKERS = [
        "i am not sure", "i'm not sure", "as an ai", "i don't have enough context",
        "i cannot fulfill", "i apologize", "unable to", "not qualified",
        "抱歉", "我不确定", "作为一个AI", "无法确定"
    ]

    def __init__(
        self, 
        model_name: str = "qwen2.5:0.5b",
        base_url: str = "http://localhost:11434/api/chat",
        fallback_to_mock: bool = True
    ):
        """
        Args:
            model_name: Ollama 中你截图里显示的实际模型名称
            base_url: Ollama 原生 API 地址
            fallback_to_mock: 如果没开 Ollama，是否退回 Mock 模式
        """
        self.model_name = model_name
        self.base_url = base_url
        self.fallback_to_mock = fallback_to_mock
        
        # 尝试连一下 Ollama 看看活着没
        if self._test_connection():
            self.use_ollama = True
            self.name = f"ollama-{model_name}"
            print(f"[SmallModel] Ollama connected: {model_name} (Direct API)")
        else:
            self.use_ollama = False
            self.name = "mock-small-model"
            if fallback_to_mock:
                print(f"[SmallModel] Ollama not responding. Falling back to mock mode.")
            else:
                print(f"[SmallModel] WARNING: Ollama connection failed and mock is disabled.")

    def _test_connection(self) -> bool:
        """用极简请求测试 Ollama 服务是否可用"""
        try:
            return self._call_ollama("hello", max_tokens=5) is not None
        except Exception:
            return False

    def _call_ollama(self, query: str, temperature: float = 0.3, max_tokens: int = 512) -> str | None:
        """底层请求：用 Python 原生库调用 Ollama"""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": query}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            req = urllib.request.Request(
                self.base_url, 
                data=json.dumps(payload).encode('utf-8'), 
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['message']['content']
        except Exception as e:
            print(f"[SmallModel Error] HTTP Request failed: {e}")
            return None

    def generate(self, query: str) -> str:
        """对外暴露的生成接口"""
        if self.use_ollama:
            return self._generate_with_ollama(query)
        else:
            return self._generate_with_mock(query)

    def _generate_with_ollama(self, query: str) -> str:
        """执行真实的推理与自我校验"""
        answer = self._call_ollama(query)
        
        if answer is None:
            if self.fallback_to_mock:
                return self._generate_with_mock(query)
            return "[System Error] Ollama failed to respond."

        # 校验 1：是否表达了不自信？
        if self._detect_uncertainty(answer):
            return f"[UNCERTAIN_MODEL_OUTPUT] {answer}"
        
        # 校验 2：对于长问题，回答是不是太短了？
        if len(answer) < 20 and len(query) > 50:
            return f"[SHORT_RESPONSE] {answer}"
        
        return answer

    def _generate_with_mock(self, query: str) -> str:
        """完全离线时的退化策略：依靠字数和关键词判断"""
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