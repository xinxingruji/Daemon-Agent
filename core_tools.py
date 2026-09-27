# 存放纯粹的执行逻辑，不依赖于任何复杂的 Agent 状态。

import json
import os
from pathlib import Path
import stat
import tempfile

from command_executor import CommandExecutor
from config import WORKDIR
from tool_result import ErrorKind, ToolResult


COMMAND_EXECUTOR = CommandExecutor(WORKDIR)

# === SECTION: base_tools ===
def get_command_executor() -> CommandExecutor:
    return COMMAND_EXECUTOR


def safe_path(p: str) -> Path:
    root = WORKDIR.resolve()
    path = (root / p).resolve()
    if not path.is_relative_to(root):
        raise PermissionError(f"Path escapes workspace: {p}")
    relative_parts = tuple(part.lower() for part in path.relative_to(root).parts)
    if ".git" in relative_parts:
        raise PermissionError("Direct access to .git is blocked; use approved Git commands")
    if relative_parts:
        filename = relative_parts[-1]
        if filename == "litellm_config.yaml":
            raise PermissionError("Direct access to the real LiteLLM configuration is blocked")
        if filename == ".env" or (
            filename.startswith(".env.") and filename != ".env.example"
        ):
            raise PermissionError("Direct access to runtime environment files is blocked")
    return path

def run_bash(command: str, timeout: int = 120, requester: str = "agent") -> ToolResult:
    return COMMAND_EXECUTOR.execute(command, timeout=timeout, requester=requester)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        if previous_mode is not None:
            os.chmod(temporary_name, previous_mode)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _file_failure(exc: Exception) -> ToolResult:
    kind = ErrorKind.PERMISSION if isinstance(exc, PermissionError) else ErrorKind.TOOL
    return ToolResult.failure(str(exc), error_kind=kind)


def run_read(path: str, limit: int = None) -> ToolResult:
    if not isinstance(path, str) or not path.strip():
        return ToolResult.failure("Path must be a non-empty string", error_kind=ErrorKind.MODEL)
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        return ToolResult.failure("Limit must be a positive integer", error_kind=ErrorKind.MODEL)
    try:
        lines = safe_path(path).read_text(encoding='utf-8').splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return ToolResult.success("\n".join(lines)[:50000], exit_code=None)
    except Exception as e:
        return _file_failure(e)

def run_write(path: str, content: str) -> ToolResult:
    if not isinstance(path, str) or not path.strip():
        return ToolResult.failure("Path must be a non-empty string", error_kind=ErrorKind.MODEL)
    if not isinstance(content, str):
        return ToolResult.failure("Content must be a string", error_kind=ErrorKind.MODEL)
    try:
        fp = safe_path(path)
        _atomic_write_text(fp, content)
        byte_count = len(content.encode("utf-8"))
        return ToolResult.success(f"Wrote {byte_count} bytes to {path}", exit_code=None)
    except Exception as e:
        return _file_failure(e)

def run_edit(path: str, old_text: str, new_text: str) -> ToolResult:
    if not isinstance(path, str) or not path.strip():
        return ToolResult.failure("Path must be a non-empty string", error_kind=ErrorKind.MODEL)
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return ToolResult.failure("old_text and new_text must be strings", error_kind=ErrorKind.MODEL)
    if not old_text:
        return ToolResult.failure("old_text must not be empty", error_kind=ErrorKind.MODEL)
    try:
        fp = safe_path(path)
        c = fp.read_text(encoding='utf-8')
        if old_text not in c:
            return ToolResult.failure(f"Text not found in {path}", error_kind=ErrorKind.TOOL)
        _atomic_write_text(fp, c.replace(old_text, new_text, 1))
        return ToolResult.success(f"Edited {path}", exit_code=None)
    except Exception as e:
        return _file_failure(e)

# === SECTION: compression (s06) ===
def estimate_tokens(messages: list) -> int:
    return len(json.dumps(messages, default=str)) // 4

def microcompact(messages: list):
    indices = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    indices.append(part)
    if len(indices) <= 3:
        return
    for part in indices[:-3]:
        if isinstance(part.get("content"), str) and len(part["content"]) > 100:
            part["content"] = "[cleared]"

def is_tool_error(output: object) -> bool:
    return isinstance(output, ToolResult) and not output.ok


def should_record_mistake(output: object) -> bool:
    return isinstance(output, ToolResult) and output.should_record_mistake
