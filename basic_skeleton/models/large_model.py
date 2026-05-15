import json
import os
import urllib.request


def _resolve_config_path() -> str:
    """查找项目根目录的 config.json"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "config.json")


def _read_api_key_from_config() -> str:
    """从 config.json 读取 DeepSeek API Key"""
    try:
        path = _resolve_config_path()
        with open(path, "r") as f:
            cfg = json.load(f)
        return cfg.get("deepseek_api_key", "") or ""
    except Exception:
        return ""


class LargeModel:
    name = "deepseek-reasoner"

    def __init__(self, model_name: str = "deepseek-reasoner"):
        """
        Args:
            model_name: DeepSeek API 模型名（默认 deepseek-reasoner = Pro）
        """
        self.model_name = model_name
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "") or _read_api_key_from_config()
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

        if self.api_key:
            self.use_api = True
            print(f"[LargeModel] DeepSeek API ready: {model_name}")
        else:
            self.use_api = False
            print(f"[LargeModel] WARNING: DEEPSEEK_API_KEY not set.")

    def _call_deepseek(self, messages: list, temperature: float = 0.5) -> str:
        """调用 DeepSeek Pro API"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        try:
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"\n[LargeModel Error] DeepSeek API request failed: {e}")
            return f"[System Error] DeepSeek API failed: {e}"

    def generate(self, query: str) -> str:
        if not self.use_api:
            return "[System Error] DEEPSEEK_API_KEY not set. Cannot call large model."

        q_lower = query.lower()

        # 判断是否是修复请求（包含 escalation 传入的修复上下文）
        is_repair = (
            "improving" in q_lower or "weak draft" in q_lower
            or "quality issues" in q_lower or "user query:" in q_lower
        )

        if is_repair:
            return self._handle_repair(query)

        return self._handle_regular(query)

    def _handle_repair(self, repair_context: str) -> str:
        """处理小模型翻车后的修复请求"""
        system_prompt = (
            "You are an expert senior system architect. "
            "Review the user's original query, the weak draft, and the identified quality issues. "
            "Provide a comprehensive, highly accurate, and concrete response that completely resolves the issues. "
            "Do not apologize. Just provide the final improved answer directly."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": repair_context},
        ]

        return self._call_deepseek(messages, temperature=0.6)

    def _handle_regular(self, query: str) -> str:
        """处理直达大模型的常规高难度请求"""
        system_prompt = (
            "You are an expert technical assistant specializing in software architecture "
            "and deep technical analysis. Provide concrete, structured, and thorough answers."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        return self._call_deepseek(messages, temperature=0.5)
