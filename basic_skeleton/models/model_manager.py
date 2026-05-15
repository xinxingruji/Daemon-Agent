from models.large_model import LargeModel
from models.small_model import SmallModel

class ModelManager:
    """统管本地零依赖大小模型的管理器"""
    
    def __init__(self) -> None:
        # 1. 实例化小模型 (去掉了 use_ollama，更新了原生 API 地址)
        self.small = SmallModel(
            model_name="qwen2.5:0.5b",
            base_url="http://localhost:11434/api/chat", 
            fallback_to_mock=False  
        )
        
        # 2. 实例化大模型
        self.large = LargeModel(
            model_name="qwen3.5:9b"
        )

    def generate(self, model_name: str, query: str) -> str:
        try:
            if model_name == "small":
                return self.small.generate(query)
            if model_name == "large":
                return self.large.generate(query)
            raise ValueError(f"Unknown model_name: {model_name}")
        except Exception as e:
            return f"[System Error] Model {model_name} execution failed: {e}"