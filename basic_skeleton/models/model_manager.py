from models.large_model import LargeModel
from models.small_model import SmallModel


class ModelManager:
    """混合模型管理器：小模型(Ollama本地) + 大模型(DeepSeek Pro API)"""

    def __init__(self) -> None:
        # 小模型：本地 Ollama 小 Qwen（弱、免费、经常失败触发升级）
        self.small = SmallModel(
            model_name="qwen2.5:1.5b",
            fallback_to_mock=True,
        )

        # 大模型：DeepSeek Pro（强、兜底修复）
        self.large = LargeModel(
            model_name="deepseek-reasoner"
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
