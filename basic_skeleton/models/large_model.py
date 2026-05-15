import json
import urllib.request

class LargeModel:
    name = "qwen3.5:9b-fallback"

    def __init__(self, model_name: str = "qwen3.5:9b"):
        # 直接使用你截图里的模型名
        self.model_name = model_name
        print(f"[LargeModel] Initialized with {model_name} via Direct API")

    def _call_ollama(self, messages: list, temperature: float) -> str:
        """底层方法：使用 Python 自带库直接调用 Ollama 原生 API"""
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            # 发送 HTTP POST 请求给本地运行的小羊驼
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode('utf-8'), 
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['message']['content']
        except Exception as e:
            print(f"\n[LargeModel Error] 无法连接到本地 Ollama: {e}")
            return f"[System Error] Fallback model failed: {e}"

    def generate(self, query: str) -> str:
        q_lower = query.lower()
        
        # 完美保留你的路由解析判断
        is_repair = ("improving" in q_lower or "weak draft" in q_lower or 
                     "quality issues" in q_lower or "user query:" in q_lower)
        
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
            {"role": "user", "content": repair_context}
        ]
        
        return self._call_ollama(messages, temperature=0.6)
    
    def _handle_regular(self, query: str) -> str:
        """处理直达大模型的常规高难度请求"""
        system_prompt = (
            "You are an expert technical assistant specializing in software architecture "
            "and deep technical analysis. Provide concrete, structured, and thorough answers."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        return self._call_ollama(messages, temperature=0.5)