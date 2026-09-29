"""Read-only startup diagnostics for Daemon-Agent."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from seed_cache import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_URL,
    validate_seed_cache,
)
from embedding_service import DEFAULT_EMBEDDING_CACHE_SIZE
from utterances import LARGE, SMALL


@dataclass(frozen=True)
class DiagnosticResult:
    name: str
    status: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def read_env_file(path: Path) -> dict[str, str]:
    """Read simple dotenv assignments without changing process environment."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _setting(name: str, env_file: dict[str, str], default: str = "") -> str:
    return os.getenv(name) or env_file.get(name) or default


def safe_endpoint(url: str) -> str:
    """Return an endpoint safe for logs by removing credentials and query data."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "<invalid endpoint>"


def check_python() -> DiagnosticResult:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 11):
        return DiagnosticResult("python", "error", f"Python {version}; requires 3.11+")
    return DiagnosticResult("python", "ok", f"Python {version}")


def check_dependencies() -> list[DiagnosticResult]:
    modules = {
        "anthropic": "anthropic",
        "python-dotenv": "dotenv",
        "PyYAML": "yaml",
        "LiteLLM": "litellm",
        "pytest": "pytest",
    }
    results = []
    for label, module in modules.items():
        installed = importlib.util.find_spec(module) is not None
        results.append(DiagnosticResult(
            f"dependency:{label}",
            "ok" if installed else "error",
            "installed" if installed else "missing; run pip install -r requirements.txt",
        ))
    return results


def _load_model_config(path: Path) -> tuple[Any, str | None]:
    if not path.exists():
        return None, f"missing configuration file: {path.name}"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"cannot read {path.name}: {exc}"

    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    try:
        import yaml
    except ImportError:
        return None, "PyYAML is required to inspect litellm_config.yaml"
    try:
        return yaml.safe_load(text), None
    except Exception:
        return None, f"invalid YAML in {path.name}; fix its formatting"


def check_model_mapping(path: Path) -> DiagnosticResult:
    data, error = _load_model_config(path)
    if error:
        return DiagnosticResult("model-mapping", "error", error)
    if not isinstance(data, dict) or not isinstance(data.get("model_list"), list):
        return DiagnosticResult("model-mapping", "error", "model_list must be a list")

    names = {
        item.get("model_name")
        for item in data["model_list"]
        if isinstance(item, dict)
    }
    missing = sorted({"small", "large"} - names)
    if missing:
        return DiagnosticResult(
            "model-mapping", "error", f"missing model aliases: {', '.join(missing)}",
        )
    return DiagnosticResult("model-mapping", "ok", "small and large aliases are configured")


def check_environment(env_file: dict[str, str], env_path: Path) -> list[DiagnosticResult]:
    results = []
    if env_path.exists():
        results.append(DiagnosticResult("env-file", "ok", f"found {env_path.name}"))
    else:
        results.append(DiagnosticResult(
            "env-file", "error", f"missing {env_path.name}; copy .env.example",
        ))

    base_url = _setting("ANTHROPIC_BASE_URL", env_file)
    results.append(DiagnosticResult(
        "anthropic-base-url",
        "ok" if base_url else "error",
        f"configured: {safe_endpoint(base_url)}" if base_url
        else "ANTHROPIC_BASE_URL is not configured",
    ))

    api_key_present = bool(
        _setting("ANTHROPIC_API_KEY", env_file)
        or _setting("ANTHROPIC_AUTH_TOKEN", env_file)
    )
    results.append(DiagnosticResult(
        "anthropic-credential",
        "ok" if api_key_present else "error",
        "configured (value hidden)" if api_key_present
        else "ANTHROPIC_API_KEY is not configured",
    ))
    return results


def check_seed_cache(path: Path, expected_model: str) -> DiagnosticResult:
    if not path.exists():
        return DiagnosticResult(
            "seed-cache", "error", "missing seed_vectors.json; run precompute_seeds.py",
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return DiagnosticResult("seed-cache", "error", f"cannot read cache: {exc}")

    report = validate_seed_cache(
        data,
        expected_model=expected_model,
        source_routes={"small": SMALL, "large": LARGE},
    )
    counts = f"small={report.counts['small']}, large={report.counts['large']}"
    if not report.valid:
        return DiagnosticResult(
            "seed-cache", "error",
            f"{'; '.join(report.errors)}; run precompute_seeds.py",
        )
    if report.legacy:
        return DiagnosticResult(
            "seed-cache", "warning",
            f"legacy cache ({counts}, dimension={report.dimension}); "
            "run precompute_seeds.py to add metadata",
        )
    return DiagnosticResult(
        "seed-cache", "ok", f"schema valid ({counts}, dimension={report.dimension})",
    )


def check_embedding_cache_size(value: str) -> DiagnosticResult:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return DiagnosticResult(
            "embedding-cache", "error", "EMBEDDING_CACHE_SIZE must be an integer",
        )
    if size < 0:
        return DiagnosticResult(
            "embedding-cache", "error", "EMBEDDING_CACHE_SIZE must be non-negative",
        )
    message = "disabled" if size == 0 else f"capacity={size}"
    return DiagnosticResult("embedding-cache", "ok", message)


def check_ollama(api_url: str, model_name: str, timeout: float) -> DiagnosticResult:
    parsed = urlsplit(api_url)
    tags_url = urlunsplit((parsed.scheme, parsed.netloc, "/api/tags", "", ""))
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return DiagnosticResult(
            "ollama", "error",
            f"unreachable at {safe_endpoint(tags_url)}: {type(exc).__name__}",
        )

    installed = {
        item.get("name") or item.get("model")
        for item in payload.get("models", [])
        if isinstance(item, dict)
    }
    expected_base = model_name.split(":", 1)[0]
    found = any(
        isinstance(name, str) and name.split(":", 1)[0] == expected_base
        for name in installed
    )
    if not found:
        return DiagnosticResult(
            "ollama", "error",
            f"service reachable but embedding model {model_name!r} is not installed",
        )
    return DiagnosticResult(
        "ollama", "ok", f"service reachable; embedding model {model_name!r} found",
    )


def check_litellm(base_url: str, timeout: float) -> DiagnosticResult:
    if not base_url:
        return DiagnosticResult("litellm", "error", "ANTHROPIC_BASE_URL is not configured")
    try:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("expected an http(s) URL")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            pass
    except Exception as exc:
        return DiagnosticResult(
            "litellm", "error",
            f"unreachable at {safe_endpoint(base_url)}: {type(exc).__name__}",
        )
    return DiagnosticResult(
        "litellm", "ok", f"TCP endpoint reachable at {safe_endpoint(base_url)}",
    )


def run_diagnostics(
    workdir: Path,
    *,
    offline: bool = False,
    timeout: float = 2.0,
) -> list[DiagnosticResult]:
    workdir = workdir.resolve()
    env_path = workdir / ".env"
    env_file = read_env_file(env_path)
    embedding_model = _setting(
        "EMBEDDING_MODEL", env_file, DEFAULT_EMBEDDING_MODEL,
    )
    embedding_url = _setting(
        "OLLAMA_EMBEDDING_URL", env_file, DEFAULT_EMBEDDING_URL,
    )
    embedding_cache_size = _setting(
        "EMBEDDING_CACHE_SIZE", env_file, str(DEFAULT_EMBEDDING_CACHE_SIZE),
    )
    base_url = _setting("ANTHROPIC_BASE_URL", env_file)

    results = [check_python(), *check_dependencies()]
    results.extend(check_environment(env_file, env_path))
    results.append(check_model_mapping(workdir / "litellm_config.yaml"))
    results.append(check_seed_cache(workdir / "seed_vectors.json", embedding_model))
    results.append(check_embedding_cache_size(embedding_cache_size))
    results.append(DiagnosticResult(
        "workspace", "ok" if os.access(workdir, os.W_OK) else "error",
        "writable" if os.access(workdir, os.W_OK) else "not writable",
    ))

    if offline:
        results.extend([
            DiagnosticResult("ollama", "skipped", "offline mode"),
            DiagnosticResult("litellm", "skipped", "offline mode"),
        ])
    else:
        results.append(check_ollama(embedding_url, embedding_model, timeout))
        results.append(check_litellm(base_url, timeout))
    return results


def render_diagnostics(results: list[DiagnosticResult]) -> str:
    labels = {"ok": "PASS", "warning": "WARN", "error": "FAIL", "skipped": "SKIP"}
    return "\n".join(
        f"[{labels.get(item.status, item.status.upper()):4}] "
        f"{item.name}: {item.message}"
        for item in results
    )


def diagnostics_exit_code(results: list[DiagnosticResult]) -> int:
    return 1 if any(item.status == "error" for item in results) else 0
