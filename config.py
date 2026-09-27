# 提取所有的系统变量、路径和 Anthropic 以及 Router 的初始化逻辑。

import os
import threading
from pathlib import Path
from typing import Callable

from router import Claude_Router
from seed_cache import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_URL

WORKDIR = Path.cwd()

TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOKEN_THRESHOLD = 100000
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

VALID_MSG_TYPES = {"message", "broadcast", "shutdown_request",
                   "shutdown_response", "plan_approval_response"}


_env_loaded = False
_env_lock = threading.Lock()
_client_instance = None
_client_lock = threading.Lock()
_router_instance = None
_router_lock = threading.Lock()


def load_runtime_env() -> None:
    """Load .env once, only when a runtime dependency is first requested."""
    global _env_loaded
    if _env_loaded:
        return
    with _env_lock:
        if _env_loaded:
            return
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError(
                "python-dotenv is required; run pip install -r requirements.txt",
            ) from exc
        load_dotenv(override=True)
        if os.getenv("ANTHROPIC_BASE_URL"):
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        _env_loaded = True


def get_client():
    """Create the Anthropic-compatible client on first use."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance
    with _client_lock:
        if _client_instance is None:
            load_runtime_env()
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic is required; run pip install -r requirements.txt",
                ) from exc
            _client_instance = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
    return _client_instance


def get_router() -> Claude_Router:
    """Create the semantic router on first use."""
    global _router_instance
    if _router_instance is not None:
        return _router_instance
    with _router_lock:
        if _router_instance is None:
            load_runtime_env()
            _router_instance = Claude_Router(
                model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                api_url=os.getenv("OLLAMA_EMBEDDING_URL", DEFAULT_EMBEDDING_URL),
            )
    return _router_instance


class LazyResource:
    """Compatibility proxy that preserves existing client/ROUTER imports."""

    def __init__(self, factory: Callable):
        self._factory = factory

    def __getattr__(self, name):
        return getattr(self._factory(), name)

    def __repr__(self) -> str:
        return f"<LazyResource factory={self._factory.__name__}>"


def runtime_initialization_status() -> dict[str, bool]:
    """Expose initialization state for diagnostics and regression tests."""
    return {
        "environment": _env_loaded,
        "client": _client_instance is not None,
        "router": _router_instance is not None,
    }


client = LazyResource(get_client)
ROUTER = LazyResource(get_router)
