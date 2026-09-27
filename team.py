# 负责生成子智能体 (Subagent) 和管理持久化的队友 (Teammate)，并且容纳压缩上下文的 LLM 调用。


import json
import sys
import threading
import time
from config import client, WORKDIR, TEAM_DIR, TASKS_DIR, TRANSCRIPT_DIR, ROUTER, POLL_INTERVAL, IDLE_TIMEOUT
from core_tools import run_bash, run_read, run_write, run_edit, estimate_tokens
from managers import MessageBus, TaskManager
from tool_result import ErrorKind, ToolResult, ensure_tool_result

# 重配 stdout 编码，防止 UTF-8 内容打印到 GBK 终端时 UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    conv_text = json.dumps(messages, default=str)[-80000:]
    resp = client.messages.create(
        model="large",  # 上下文压缩默认调用大模型
        messages=[{"role": "user", "content": f"Summarize for continuity:\n{conv_text}"}],
        max_tokens=2000,
    )
    summary = resp.content[0].text
    return [
        {"role": "user", "content": f"[Compressed. Transcript: {path}]\n{summary}"},
    ]

def _invoke_agent_tool(name: str, arguments: dict, requester: str) -> ToolResult:
    dispatch = {
        "bash": lambda **kw: run_bash(kw["command"], requester=requester),
        "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
        "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
        "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    }
    handler = dispatch.get(name)
    if handler is None:
        return ToolResult.failure(f"Unknown tool: {name}", error_kind=ErrorKind.MODEL)
    try:
        return ensure_tool_result(handler(**arguments))
    except (KeyError, TypeError, ValueError) as exc:
        return ToolResult.failure(
            f"Invalid arguments for {name}: {exc}",
            error_kind=ErrorKind.MODEL,
        )
    except Exception as exc:
        return ToolResult.failure(
            f"Tool {name} failed internally: {exc}",
            error_kind=ErrorKind.INTERNAL,
        )


# === SECTION: subagent (s04) ===
def run_subagent(prompt: str, agent_type: str = "Explore") -> ToolResult:
    sub_tools = [
        {"name": "bash", "description": "Run a direct command; high-risk operations require human approval.",
         "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        {"name": "read_file", "description": "Read file.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    ]
    if agent_type != "Explore":
        sub_tools += [
            {"name": "write_file", "description": "Write file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Edit file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
        ]
    sub_msgs = [{"role": "user", "content": prompt}]
    resp = None
    for _ in range(30):

        # 路由：
        current_tokens = estimate_tokens(sub_msgs)
        current_model = ROUTER.route(query=prompt, total_tokens=current_tokens, force_large=False)

        resp = client.messages.create(model=current_model, messages=sub_msgs, tools=sub_tools, max_tokens=8000)
        sub_msgs.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                output = _invoke_agent_tool(b.name, b.input, f"subagent:{agent_type}")
                if output.should_record_mistake and current_model == "small":
                    ROUTER.record_mistake(prompt)

                results.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": output.to_model_content()[:50000],
                })
        sub_msgs.append({"role": "user", "content": results})
    if resp:
        summary = "".join(b.text for b in resp.content if hasattr(b, "text"))
        if summary:
            return ToolResult.success(summary, exit_code=None)
        return ToolResult.failure("Subagent returned no summary", error_kind=ErrorKind.TOOL)
    return ToolResult.failure("Subagent failed to produce a response", error_kind=ErrorKind.ENVIRONMENT)

# === SECTION: team (s09/s11) ===
class TeammateManager:
    def __init__(self, bus: MessageBus, task_mgr: TaskManager):
        TEAM_DIR.mkdir(exist_ok=True)
        self.bus = bus
        self.task_mgr = task_mgr
        self.config_path = TEAM_DIR / "config.json"
        self.config = self._load()
        self.threads = {}

    def _load(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name: return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> ToolResult:
        member = self._find(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return ToolResult.failure(
                    f"'{name}' is currently {member['status']}",
                    error_kind=ErrorKind.TOOL,
                )
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save()
        threading.Thread(target=self._loop, args=(name, role, prompt), daemon=True).start()
        return ToolResult.success(f"Spawned '{name}' (role: {role})", exit_code=None)

    def _set_status(self, name: str, status: str):
        member = self._find(name)
        if member:
            member["status"] = status
            self._save()

    def _loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        sys_prompt = (f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
                      f"Use idle when done with current work. You may auto-claim tasks.")
        messages = [{"role": "user", "content": prompt}]
        tools = [
            {"name": "bash", "description": "Run a direct command; high-risk operations require human approval.", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Edit file.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            {"name": "send_message", "description": "Send message.", "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}}, "required": ["to", "content"]}},
            {"name": "idle", "description": "Signal no more work.", "input_schema": {"type": "object", "properties": {}}},
            {"name": "claim_task", "description": "Claim task by ID.", "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
        ]
        while True:
            # -- WORK PHASE --
            for _ in range(50):
                inbox = self.bus.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": f"<inbox>{json.dumps(msg)}</inbox>"})

                # 路由
                current_tokens = estimate_tokens(messages)
                current_mission = prompt  # 兜底宏观目标
                for msg in reversed(messages):
                    if msg["role"] == "user" and isinstance(msg["content"], str):
                        content_str = msg["content"]
                        
                        # 嗅探 1：任务板认领
                        if "<auto-claimed>" in content_str:
                            current_mission = content_str
                            break
                            
                        # 嗅探 2：Inbox 私信
                        elif "<inbox>" in content_str:
                            # 既然我们确定它是 JSON 格式的字符串，直接用正则或简单的截取即可
                            # 但最安全的还是把它当作整段字符串喂给路由器，或者精细化提取：
                            try:
                                # 剥离外层的 <inbox> 标签还原 JSON
                                raw_json = content_str.replace("<inbox>", "").replace("</inbox>", "")
                                msg_dict = json.loads(raw_json)
                                current_mission = msg_dict.get("content", content_str)
                            except:
                                current_mission = content_str # 兜底
                            break
                current_model = ROUTER.route(query=current_mission, total_tokens=current_tokens, force_large=False)


                try:
                    response = client.messages.create(
                        model=current_model, system=sys_prompt, messages=messages,
                        tools=tools, max_tokens=8000)
                except Exception:
                    self._set_status(name, "shutdown")
                    return
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break
                results = []
                idle_requested = False
                for block in response.content:
                    if block.type == "tool_use":
                        try:
                            if block.name == "idle":
                                idle_requested = True
                                output = ToolResult.success("Entering idle phase.", exit_code=None)
                            elif block.name == "claim_task":
                                output = ensure_tool_result(
                                    self.task_mgr.claim(block.input["task_id"], name)
                                )
                            elif block.name == "send_message":
                                output = ensure_tool_result(
                                    self.bus.send(name, block.input["to"], block.input["content"])
                                )
                            else:
                                output = _invoke_agent_tool(
                                    block.name,
                                    block.input,
                                    f"teammate:{name}",
                                )
                        except (KeyError, TypeError, ValueError) as exc:
                            output = ToolResult.failure(
                                f"Invalid arguments for {block.name}: {exc}",
                                error_kind=ErrorKind.MODEL,
                            )
                        except Exception as exc:
                            output = ToolResult.failure(
                                f"Tool {block.name} failed internally: {exc}",
                                error_kind=ErrorKind.INTERNAL,
                            )
                        out_str = output.content
                        # 按工具类型定制显示，和 main.py 保持一致的风格
                        path = ""
                        if hasattr(block, 'input') and 'path' in block.input:
                            path = block.input['path'].replace(str(WORKDIR), ".")
                        if block.name == "bash":
                            print(f"  \033[36m[{name}] > bash:\033[0m")
                            print(f"  {out_str[:120]}")
                        elif block.name == "read_file":
                            lines = out_str.count('\n') if output.ok else 0
                            if not output.ok:
                                print(f"  \033[31m[{name}] ⚠ read_file: {out_str[:120]}\033[0m")
                            else:
                                print(f"  \033[34m[{name}] 📄 read_file: {path} ({lines} 行)\033[0m")
                        elif block.name in ("write_file", "edit_file"):
                            print(f"  \033[32m[{name}] ✏️ {block.name}: {path}\033[0m")
                        else:
                            print(f"  \033[33m[{name}] 🔧 {block.name}: {out_str[:120]}\033[0m")

                        # 错题本机制
                        if output.should_record_mistake and current_model == "small":
                            ROUTER.record_mistake(current_mission)

                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output.to_model_content(),
                        })
                messages.append({"role": "user", "content": results})
                if idle_requested:
                    break
            # -- IDLE PHASE: poll for messages and unclaimed tasks --
            self._set_status(name, "idle")
            resume = False
            for _ in range(IDLE_TIMEOUT // max(POLL_INTERVAL, 1)):
                time.sleep(POLL_INTERVAL)
                inbox = self.bus.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break
                unclaimed = []
                for f in sorted(TASKS_DIR.glob("task_*.json")):
                    t = json.loads(f.read_text())
                    if t.get("status") == "pending" and not t.get("owner") and not t.get("blockedBy"):
                        unclaimed.append(t)
                if unclaimed:
                    task = unclaimed[0]
                    self.task_mgr.claim(task["id"], name)
                    # Identity re-injection for compressed contexts
                    if len(messages) <= 3:
                        messages.insert(0, {"role": "user", "content":
                            f"<identity>You are '{name}', role: {role}, team: {team_name}.</identity>"})
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})
                    messages.append({"role": "user", "content":
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n{task.get('description', '')}</auto-claimed>"})
                    messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})
                    resume = True
                    break
            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    def list_all(self) -> str:
        if not self.config["members"]: return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]
