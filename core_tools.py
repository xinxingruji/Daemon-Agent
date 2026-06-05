# 存放纯粹的执行逻辑，不依赖于任何复杂的 Agent 状态。

import subprocess
import json
import sys
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
    # Windows 上自动转换常见 Unix 命令，避免模型习惯性写 ls / pwd 时报错
    if sys.platform == "win32":
        unix_to_win = {
            "ls": "dir", "ls -l": "dir", "ls -la": "dir /a",
            "ls -lh": "dir", "ls -lha": "dir /a", "ls -al": "dir /a",
            "pwd": "cd", "cat": "type", "clear": "cls",
        }
        cmd_stripped = command.strip()
        if cmd_stripped in unix_to_win:
            command = unix_to_win[cmd_stripped]
    # 检测疑似文件内容被错误地当作命令执行
    # 超长内容且无 shell 操作符 → 很可能是文件内容而不是命令
    SHELL_OPERATORS = (";", "|", "&&", "||", "`", "$(", ">", "<")
    if len(command) > 500 and not any(op in command for op in SHELL_OPERATORS):
        return ("Error: 输入内容过长且不含 shell 操作符，看起来像是文件内容被误当作命令执行。"
                "如想读取文件请使用 read_file 工具。")
    try:
        # 先以二进制模式捕获输出，手动解码
        # 先试 UTF-8（API/网页内容多是 UTF-8），失败再回退到系统编码（GBK，系统消息）
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, timeout=120)
        out_bytes = r.stdout + r.stderr
        out = ""
        for enc in ['utf-8', 'gbk']:
            try:
                out = out_bytes.decode(enc).strip()
                break
            except UnicodeDecodeError:
                continue
        if not out:
            out = out_bytes.decode('utf-8', errors='replace').strip()

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
    except (UnicodeDecodeError, UnicodeError) as e:
        return f"Error: 编码解码失败 - {e}"

def run_read(path: str, limit: int = None) -> str:
    try:
        # 显式指定 UTF-8 编码，避免中文 Windows 默认用 GBK 解码 UTF-8 文件时崩溃
        lines = safe_path(path).read_text(encoding='utf-8').splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        # 显式指定 UTF-8 编码，避免中文 Windows 默认用 GBK 写入时丢字符或崩溃
        fp.write_text(content, encoding='utf-8')
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        # 显式指定 UTF-8 编码，与 run_read / run_write 保持一致
        c = fp.read_text(encoding='utf-8')
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1), encoding='utf-8')
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