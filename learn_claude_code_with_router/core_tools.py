# 存放纯粹的执行逻辑，不依赖于任何复杂的 Agent 状态。

import subprocess
import json
from pathlib import Path
from config import WORKDIR

# === SECTION: base_tools ===
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

# run_bash返回命令行的输出以及退出码returncode，如果returncode不为0则表示出错
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()

        if r.returncode != 0:
            # 1. 提取命令的基础名 (去掉参数)
            # 例如 "grep -r 'todo' ." -> "grep"
            base_cmd = command.strip().split()[0] if command.strip() else ""
            
            # 2. 定义豁免规则：
            # grep 找不到内容时返回 1
            # diff 发现差异时返回 1
            if r.returncode == 1 and base_cmd in ("grep", "diff", "cmp"):
                # 这是正常探测结果，不视为 Error
                return out[:50000] if out else "(no matches / differences found)"
                
            # 3. 其他非 0 退出码，老老实实打上 Error 标签
            return f"Error: Bash exit code {r.returncode}\n{out}"
            
        return out[:50000] if out else "(no output)"
        
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

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

def is_tool_error(output: str) -> bool:
    out_str = str(output).strip()
    # 这是针对run_bash的，如果返回码非0，那一定出错了
    # 绝对确定的系统级错误前缀 
    error_prefixes = (
        "Error:", 
        "error:", 
        "ERROR:",
        "Unknown",        # 捕获 "Unknown tool: xxx"
        "KeyError",       # 防御性捕获
        "Exception",      # 防御性捕获
        "Traceback",      # 捕获未格式化的 Python 崩溃堆栈
        "Fatal:",         # 捕获一些底层库的致命错误
        "Trace/BPT trap"  # 捕获 C 级别底层崩溃
    )
    if out_str.startswith(error_prefixes):
        return True
        
    # 特定工具的隐式失败标志
    if out_str in ("(subagent failed)", "Unknown tool", "(no summary)"):
        return True
        
    return False