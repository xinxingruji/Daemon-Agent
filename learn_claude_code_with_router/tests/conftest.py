from unittest.mock import patch

import pytest

from router import Claude_Router


def pytest_addoption(parser):
    parser.addoption("--run-api", action="store_true", default=False,
                     help="Run API latency tests (requires LiteLLM)")


def make_router(**kwargs):
    """创建一个不调用 Ollama、不读写 mistake 文件的 Router 实例"""
    with patch.object(Claude_Router, "_get_embedding", return_value=[0.5, 0.5, 0.5]):
        with patch.object(Claude_Router, "_load_mistakes", return_value=[]):
            r = Claude_Router(**kwargs)
    return r


@pytest.fixture
def router():
    return make_router()
