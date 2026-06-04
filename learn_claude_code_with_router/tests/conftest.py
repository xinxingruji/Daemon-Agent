from unittest.mock import patch

import pytest

from router import Claude_Router


def pytest_addoption(parser):
<<<<<<< Updated upstream
    parser.addoption("--run-api", action="store_true", default=False,
                     help="Run API latency tests (requires LiteLLM)")
=======
    parser.addoption(
        "--run-api", action="store_true", default=False,
        help="包含需要 LiteLLM 代理在线的测试",
    )

DIM = 768  # 匹配真实 Ollama 输出


def _fake_load_seeds(self):
    """注入测试种子向量，不读磁盘"""
    self.route_embeddings = {
        "small": [[0.5] * DIM for _ in range(5)],
        "large": [[0.5] * DIM for _ in range(3)],
    }
    self.route_embeddings_text = {
        "small": [f"seed_s{i}" for i in range(5)],
        "large": [f"seed_l{i}" for i in range(3)],
    }
>>>>>>> Stashed changes


def make_router(**kwargs):
    """创建一个不调用 Ollama、不读写磁盘的 Router 实例"""
    with patch.object(Claude_Router, "_get_embedding", return_value=[0.5] * DIM):
        with patch.object(Claude_Router, "_load_mistakes", return_value=[]):
            with patch.object(Claude_Router, "_load_seed_vectors",
                              side_effect=_fake_load_seeds, autospec=True):
                with patch.object(Claude_Router, "_save_seed_vectors"):
                    r = Claude_Router(**kwargs)
    return r


@pytest.fixture
def router():
    return make_router()
