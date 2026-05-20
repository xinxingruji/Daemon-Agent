#!/usr/bin/env python3
"""
agent.py - Integrated Agent Loop

融合 learn-claude-code s_full 的完整框架和 basic_skeleton 的自适应推理。

Features:
- 完整的多轮对话循环（s01）
- 工具分派系统（s02）
- Todo 管理（s03）
- 上下文自动压缩（s06）
- 自适应推理运行时集成
"""

import os
import sys
import json
import time
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Any
import re

from dotenv import load_dotenv

from runtime.runtime import AdaptiveInferenceRuntime

load_dotenv(override=True)

# ============================================================================
# CONFIG
# ============================================================================

WORKDIR = Path.cwd()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/chat")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "qwen3.5:9b")
COMPRESSION_MODEL = os.environ.get("COMPRESSION_MODEL", "qwen2.5:0.5b")
MAIN_MODEL = AGENT_MODEL

# Context limits
TOKEN_THRESHOLD = 100000
CONTEXT_WINDOW = 200000

# Conversation history
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TRANSCRIPT_DIR.mkdir(exist_ok=True)


def call_ollama(model: str, messages: list, temperature: float = 0.2, max_tokens: int = 4096) -> str:
    """调用本地 Ollama Chat API。"""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    request = urllib.request.Request(
        OLLAMA_BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["message"]["content"]


def extract_json_block(text: str) -> Optional[dict]:
    """从模型输出中提取 JSON 对象。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def format_adaptive_context(adaptive_result: dict) -> str:
    """把自适应推理结果格式化为可注入的上下文。"""
    answer = str(adaptive_result.get("answer", "")).strip()
    meta = adaptive_result.get("meta", {})
    meta_text = json.dumps(meta, ensure_ascii=False, indent=2)
    return (
        "Adaptive runtime result:\n"
        f"Answer:\n{answer}\n\n"
        f"Meta:\n{meta_text}\n\n"
        "Use this result as the default answer when it is sufficient. "
        "If more work is needed, continue with tools or refine it."
    )


# ============================================================================
# SECTION: TodoManager (s03)
# ============================================================================

class TodoManager:
    """轻量级 Todo 管理"""
    
    def __init__(self):
        self.items = []
    
    def update(self, items: list) -> str:
        """更新 todo 列表"""
        validated = []
        in_progress = 0
        
        for item in items:
            if not isinstance(item, dict):
                continue
            status = item.get("status", "not-started")
            if status == "in-progress":
                in_progress += 1
            
            validated.append({
                "id": len(validated) + 1,
                "title": item.get("title", ""),
                "status": status
            })
        
        if len(validated) > 20:
            return f"Error: Too many todos ({len(validated)} > 20)"
        if in_progress > 1:
            return f"Error: Too many in-progress todos ({in_progress} > 1)"
        
        self.items = validated
        return self.render()
    
    def render(self) -> str:
        """渲染 todo 列表"""
        if not self.items:
            return "(no todos)"
        
        lines = []
        for item in self.items:
            status_mark = {
                "not-started": "[ ]",
                "in-progress": "[~]",
                "completed": "[x]"
            }.get(item.get("status"), "[-]")
            
            lines.append(f"{status_mark} {item.get('title', '')}")
        
        done = sum(1 for t in self.items if t.get("status") == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)
    
    def has_open_items(self) -> bool:
        """是否有未完成的任务"""
        return any(item.get("status") != "completed" for item in self.items)


# ============================================================================
# SECTION: Context Compression (s06)
# ============================================================================

def estimate_tokens(messages: list) -> int:
    """粗略估算 token 数"""
    return len(json.dumps(messages, default=str)) // 4


def auto_compact(messages: list) -> list:
    """自动压缩长对话"""
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    
    # 保存原始记录
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    
    # 提取最后的对话用于压缩
    conv_text = json.dumps(messages[-20:], default=str)
    
    try:
        summary = call_ollama(
            COMPRESSION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize this conversation for continuity. Keep key facts, tools used, files changed, and next steps.\n\n{conv_text}",
                }
            ],
            temperature=0.1,
            max_tokens=1200,
        )
    except Exception as e:
        summary = f"(compression failed: {str(e)[:100]})"
    
    return [
        {"role": "user", "content": f"[Previous context compressed. Transcript: {path}]\n{summary}"},
    ]


# ============================================================================
# SECTION: Tool Handlers
# ============================================================================

def safe_path(p: str) -> Path:
    """确保路径不会逃逸工作目录"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 shell 命令"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                          capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except Exception as e:
        return f"Error: {str(e)[:200]}"


def run_read(path: str, limit: Optional[int] = None) -> str:
    """读取文件"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit]
            lines.append(f"... ({len(lines)} more lines omitted)")
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """编辑文件（精确替换）"""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: '{old_text[:50]}...' not found in {path}"
        
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# SECTION: Adaptive Inference Integration
# ============================================================================

def run_inference_query(query: str) -> dict:
    """调用自适应推理运行时"""
    try:
        runtime = AdaptiveInferenceRuntime()
        result = runtime.run(query)
        return result
    except Exception as e:
        return {
            "answer": f"Inference error: {str(e)[:200]}",
            "meta": {"error": str(e)}
        }


# ============================================================================
# SECTION: Tool Dispatch (s02)
# ============================================================================

TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "TodoWrite": lambda **kw: TODO.update(kw["items"]),
    "inference": lambda **kw: run_inference_query(kw["query"]),
}

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "description": "Max lines to read"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file (1st occurrence only).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"}
            },
            "required": ["path", "old_text", "new_text"]
        }
    },
    {
        "name": "TodoWrite",
        "description": "Update a todo list for tracking work.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "status": {"type": "string", "enum": ["not-started", "in-progress", "completed"]}
                        },
                        "required": ["title", "status"]
                    }
                }
            },
            "required": ["items"]
        }
    },
    {
        "name": "inference",
        "description": "Run an adaptive inference query (routes to small/large model based on complexity).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to process adaptively"}
            },
            "required": ["query"]
        }
    },
]


# ============================================================================
# SECTION: System Prompt
# ============================================================================

SYSTEM_PROMPT = f"""You are an intelligent coding assistant at {WORKDIR}.

You can use tools to read/write files, run commands, manage tasks, and invoke adaptive inference.

Key behaviors:
1. Use 'inference' tool for complex technical questions or when you need intelligent routing.
2. For file operations, always use read_file/write_file/edit_file instead of bash.
3. For task tracking, use TodoWrite to maintain a clear checklist.
4. Be direct and concise. Explain decisions clearly.
5. When working on multi-step tasks, create a todo list first.
6. A local adaptive runtime result will be provided for every user turn. Treat it as the default routed answer and refine it only if needed.

Available tools: bash, read_file, write_file, edit_file, TodoWrite, inference

Work directory: {WORKDIR}
"""


TOOL_SCHEMA_PROMPT = """
You are a local coding agent running on Ollama.

Respond in one of two valid JSON formats only:

1. Final response:
{"type":"final","content":"..."}

2. Tool calls:
{"type":"tool_calls","tool_calls":[{"name":"tool_name","input":{...}}, ...]}

Rules:
- Use tool_calls only when you need a tool.
- Do not include markdown fences.
- Do not output any text outside JSON.
- After tool results are provided, continue the task or return a final response.
"""


# ============================================================================
# SECTION: Global Instances
# ============================================================================

TODO = TodoManager()


# ============================================================================
# SECTION: Agent Loop (s01)
# ============================================================================

class Agent:
    """完整的 Agent 循环"""
    
    def __init__(self):
        self.messages = []
        self.rounds = 0
        self.max_rounds = 100
        self.turn_context = ""

    def _call_model(self) -> str:
        turn_messages = list(self.messages)
        if self.turn_context:
            turn_messages.insert(
                0,
                {
                    "role": "system",
                    "content": self.turn_context,
                },
            )
        return call_ollama(
            MAIN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + TOOL_SCHEMA_PROMPT},
                *turn_messages,
            ],
            temperature=0.2,
            max_tokens=4096,
        )

    def _append_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def _handle_model_output(self, output: str) -> bool:
        parsed = extract_json_block(output)
        if not parsed:
            print(f"\n{output}")
            self._append_assistant(output)
            return False

        if parsed.get("type") == "final":
            content = str(parsed.get("content", ""))
            print(f"\n{content}")
            self._append_assistant(content)
            return False

        if parsed.get("type") != "tool_calls":
            print(f"\n{output}")
            self._append_assistant(output)
            return False

        tool_results = []
        for tool_call in parsed.get("tool_calls", []):
            tool_name = tool_call.get("name")
            tool_input = tool_call.get("input", {})

            print(f"\n[TOOL] {tool_name}")

            if tool_name in TOOL_HANDLERS:
                try:
                    result = TOOL_HANDLERS[tool_name](**tool_input)
                except Exception as e:
                    result = f"Error: {str(e)[:500]}"
            else:
                result = f"Error: Unknown tool '{tool_name}'"

            result_text = str(result)
            print(f"[RESULT] {result_text[:200]}..." if len(result_text) > 200 else f"[RESULT] {result_text}")

            tool_results.append({
                "tool_name": tool_name,
                "input": tool_input,
                "output": result_text[:10000],
            })

        self._append_assistant(output)
        self.messages.append({
            "role": "user",
            "content": "Tool results:\n" + json.dumps(tool_results, ensure_ascii=False, indent=2)
        })
        return True
    
    def run(self, user_input: str) -> None:
        """单轮对话"""
        adaptive_result = run_inference_query(user_input)
        self.turn_context = format_adaptive_context(adaptive_result)

        # 检查是否需要压缩
        token_count = estimate_tokens(self.messages)
        if token_count > TOKEN_THRESHOLD:
            print(f"\n[COMPRESS] Context too large ({token_count} tokens), compacting...")
            self.messages = auto_compact(self.messages)
        
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_rounds):
            try:
                output = self._call_model()
            except Exception as e:
                print(f"Error: API call failed: {e}")
                return

            if not self._handle_model_output(output):
                return

            self.rounds += 1
            if self.rounds >= self.max_rounds:
                print("\n[WARN] Reached max rounds without final response.")
                return
    
    def _continue_loop(self) -> None:
        """继续处理工具结果的循环"""
        for _ in range(self.max_rounds):
            try:
                output = self._call_model()
            except Exception as e:
                print(f"Error: API call failed: {e}")
                return

            if not self._handle_model_output(output):
                return

            self.rounds += 1
            if self.rounds >= self.max_rounds:
                print("\n[WARN] Reached max rounds without final response.")
                return


# ============================================================================
# SECTION: REPL Main Loop
# ============================================================================

def main():
    """交互式 REPL"""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║          Adaptive Agent with s_full Framework                  ║")
    print("║  (type 'help' for commands, 'exit' to quit, 'clear' to reset)  ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    agent = Agent()
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except KeyboardInterrupt:
            print("\nExit.")
            break
        except EOFError:
            break
        
        if not user_input:
            continue
        
        # 特殊命令
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        elif user_input.lower() == "clear":
            agent = Agent()
            print("Conversation cleared.")
            continue
        elif user_input.lower() == "help":
            print("""
Commands:
  exit           - Exit the agent
  clear          - Clear conversation history
  
Tools available:
  bash           - Run shell commands
  read_file      - Read file
  write_file     - Write file
  edit_file      - Edit file (replace text)
  TodoWrite      - Manage task list
  inference      - Adaptive inference (routes to small/large model)
""")
            continue
        elif user_input.lower() == "todo":
            print(TODO.render())
            continue
        
        # 普通对话
        agent.run(user_input)


if __name__ == "__main__":
    main()
