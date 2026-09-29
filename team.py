# 负责生成子智能体 (Subagent) 和管理持久化的队友 (Teammate)，并且容纳压缩上下文的 LLM 调用。


import json
from pathlib import Path
import sys
import threading
import time
from config import client, WORKDIR, TEAM_DIR, TRANSCRIPT_DIR, ROUTER, POLL_INTERVAL, IDLE_TIMEOUT
from core_tools import run_bash, run_read, run_write, run_edit, estimate_tokens
from managers import MessageBus, TaskManager
from state_persistence import atomic_write_json, shared_path_lock
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
def run_subagent(
    prompt: str,
    agent_type: str = "Explore",
    max_rounds: int = 30,
) -> ToolResult:
    if max_rounds <= 0:
        return ToolResult.failure(
            "Subagent max_rounds must be positive",
            error_kind=ErrorKind.MODEL,
        )
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
    for _ in range(max_rounds):

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
    else:
        return ToolResult.failure(
            f"Subagent round budget exhausted ({max_rounds})",
            error_kind=ErrorKind.TOOL,
        )
    if resp:
        summary = "".join(b.text for b in resp.content if hasattr(b, "text"))
        if summary:
            return ToolResult.success(summary, exit_code=None)
        return ToolResult.failure("Subagent returned no summary", error_kind=ErrorKind.TOOL)
    return ToolResult.failure("Subagent failed to produce a response", error_kind=ErrorKind.ENVIRONMENT)

# === SECTION: team (s09/s11) ===
class TeammateManager:
    LIVE_STATES = {"starting", "working", "idle", "stopping"}
    TERMINAL_STATES = {"shutdown", "failed", "timed_out", "interrupted"}
    VALID_STATES = LIVE_STATES | TERMINAL_STATES
    ALLOWED_TRANSITIONS = {
        "starting": {"working", "stopping", "shutdown", "failed"},
        "working": {"idle", "stopping", "shutdown", "failed"},
        "idle": {"working", "stopping", "shutdown", "failed", "timed_out"},
        "stopping": {"shutdown", "failed"},
        "shutdown": {"starting"},
        "failed": {"starting"},
        "timed_out": {"starting"},
        "interrupted": {"starting"},
    }

    def __init__(
        self,
        bus: MessageBus,
        task_mgr: TaskManager,
        *,
        team_dir: str | Path | None = None,
        max_concurrency: int = 4,
        max_rounds: int = 50,
    ):
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        self.team_dir = TEAM_DIR if team_dir is None else Path(team_dir)
        self.team_dir.mkdir(parents=True, exist_ok=True)
        self.bus = bus
        self.task_mgr = task_mgr
        self.max_concurrency = max_concurrency
        self.max_rounds = max_rounds
        self.config_path = self.team_dir / "config.json"
        self._lock = shared_path_lock(self.config_path)
        self.threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._closing = False
        self.config, recovered = self._load()
        if recovered:
            self._save()

    def _load(self) -> tuple[dict, bool]:
        with self._lock:
            if not self.config_path.exists():
                return {"team_name": "default", "members": []}, False
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("members"), list):
            raise ValueError("team config must contain a members list")
        data.setdefault("team_name", "default")
        recovered = False
        for index, member in enumerate(data["members"]):
            if not isinstance(member, dict) or not member.get("name"):
                raise ValueError(f"team member {index} must be an object with a name")
            member.setdefault("role", "unspecified")
            if member.get("status") in self.LIVE_STATES:
                member["status"] = "interrupted"
                member["statusReason"] = (
                    "process restarted; execution context was not persisted"
                )
                member["updatedAt"] = time.time()
                recovered = True
        return data, recovered

    def _save(self) -> None:
        with self._lock:
            atomic_write_json(self.config_path, self.config)

    def _find_unlocked(self, name: str) -> dict | None:
        for member in self.config["members"]:
            if member.get("name") == name:
                return member
        return None

    def _transition_unlocked(self, member: dict, status: str, reason: str = "") -> None:
        if status not in self.VALID_STATES:
            raise ValueError(f"invalid teammate status: {status}")
        current = member.get("status")
        if (
            current in self.VALID_STATES
            and status != current
            and status not in self.ALLOWED_TRANSITIONS[current]
        ):
            raise RuntimeError(f"invalid teammate transition: {current} -> {status}")
        member["status"] = status
        member["statusReason"] = reason or None
        member["updatedAt"] = time.time()

    def _set_status(self, name: str, status: str, reason: str = "") -> None:
        with self._lock:
            member = self._find_unlocked(name)
            if member:
                self._transition_unlocked(member, status, reason)
                atomic_write_json(self.config_path, self.config)

    def _set_active_status(
        self,
        name: str,
        status: str,
        stop_event: threading.Event,
    ) -> bool:
        """Transition to a live state unless shutdown won the same lock."""
        with self._lock:
            member = self._find_unlocked(name)
            if (
                member is None
                or stop_event.is_set()
                or self._closing
                or member.get("status") == "stopping"
            ):
                return False
            self._transition_unlocked(member, status)
            atomic_write_json(self.config_path, self.config)
            return True

    def spawn(self, name: str, role: str, prompt: str) -> ToolResult:
        name = str(name).strip()
        role = str(role).strip()
        if not name or not role:
            return ToolResult.failure("name and role are required", error_kind=ErrorKind.MODEL)
        with self._lock:
            if self._closing:
                return ToolResult.denied("Teammate manager is shutting down")
            live_threads = sum(thread.is_alive() for thread in self.threads.values())
            if live_threads >= self.max_concurrency:
                return ToolResult.failure(
                    f"teammate concurrency limit reached ({self.max_concurrency})",
                    error_kind=ErrorKind.TOOL,
                )
            existing_thread = self.threads.get(name)
            if existing_thread and existing_thread.is_alive():
                return ToolResult.failure(
                    f"'{name}' already has a live execution context",
                    error_kind=ErrorKind.TOOL,
                )
            member = self._find_unlocked(name)
            now = time.time()
            if member:
                if member.get("status") not in self.TERMINAL_STATES:
                    return ToolResult.failure(
                        f"'{name}' is currently {member.get('status')}",
                        error_kind=ErrorKind.TOOL,
                    )
                member["role"] = role
                member["startedAt"] = now
                self._transition_unlocked(member, "starting")
            else:
                member = {
                    "name": name,
                    "role": role,
                    "status": "starting",
                    "statusReason": None,
                    "startedAt": now,
                    "updatedAt": now,
                }
                self.config["members"].append(member)
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run_member,
                args=(name, role, prompt, stop_event),
                daemon=True,
                name=f"teammate-{name}",
            )
            self._stop_events[name] = stop_event
            self.threads[name] = thread
            atomic_write_json(self.config_path, self.config)
            try:
                thread.start()
            except Exception as exc:
                self._set_status(name, "failed", f"thread start failed: {exc}")
                raise
        return ToolResult.success(f"Spawned '{name}' (role: {role})", exit_code=None)

    def _run_member(
        self,
        name: str,
        role: str,
        prompt: str,
        stop_event: threading.Event,
    ) -> None:
        status = "shutdown"
        reason = "work completed"
        try:
            if not self._set_active_status(name, "working", stop_event):
                status, reason = "shutdown", "shutdown requested before start"
            else:
                status, reason = self._loop(name, role, prompt, stop_event)
        except Exception as exc:
            status = "failed"
            reason = f"unhandled {type(exc).__name__}: {exc}"
        finally:
            try:
                self.task_mgr.release_owner(name, reason)
            except Exception as exc:
                status = "failed"
                reason = f"{reason}; task release failed: {type(exc).__name__}: {exc}"
            self._set_status(name, status, reason)

    def _loop(
        self,
        name: str,
        role: str,
        prompt: str,
        stop_event: threading.Event,
    ) -> tuple[str, str]:
        team_name = self.config["team_name"]
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            "Use idle when done with current work. You may auto-claim tasks."
        )
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
            idle_requested = False
            rounds_this_activation = 0
            while rounds_this_activation < self.max_rounds:
                if stop_event.is_set():
                    return "shutdown", "shutdown requested"
                inbox = self.bus.read_inbox(name)
                for message in inbox:
                    if message.get("type") == "shutdown_request":
                        return "shutdown", "shutdown requested by message"
                    messages.append({
                        "role": "user",
                        "content": f"<inbox>{json.dumps(message)}</inbox>",
                    })

                current_tokens = estimate_tokens(messages)
                current_mission = prompt
                for message in reversed(messages):
                    if message["role"] != "user" or not isinstance(message["content"], str):
                        continue
                    content = message["content"]
                    if "<auto-claimed>" in content:
                        current_mission = content
                        break
                    if "<inbox>" in content:
                        try:
                            raw = content.replace("<inbox>", "").replace("</inbox>", "")
                            current_mission = json.loads(raw).get("content", content)
                        except (json.JSONDecodeError, AttributeError):
                            current_mission = content
                        break
                current_model = ROUTER.route(
                    query=current_mission,
                    total_tokens=current_tokens,
                    force_large=False,
                )
                try:
                    response = client.messages.create(
                        model=current_model,
                        system=sys_prompt,
                        messages=messages,
                        tools=tools,
                        max_tokens=8000,
                    )
                except Exception as exc:
                    return "failed", f"model call failed: {type(exc).__name__}: {exc}"
                rounds_this_activation += 1
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
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
                    self._print_tool_result(name, block, output)
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

            if rounds_this_activation >= self.max_rounds and not idle_requested:
                return "failed", f"work round budget exhausted ({self.max_rounds})"

            if not self._set_active_status(name, "idle", stop_event):
                return "shutdown", "shutdown requested"
            deadline = time.monotonic() + IDLE_TIMEOUT
            while time.monotonic() < deadline:
                wait_time = min(POLL_INTERVAL, max(0.0, deadline - time.monotonic()))
                if stop_event.wait(wait_time):
                    return "shutdown", "shutdown requested"
                inbox = self.bus.read_inbox(name)
                if inbox:
                    for message in inbox:
                        if message.get("type") == "shutdown_request":
                            return "shutdown", "shutdown requested by message"
                        messages.append({"role": "user", "content": json.dumps(message)})
                    if not self._set_active_status(name, "working", stop_event):
                        return "shutdown", "shutdown requested"
                    break
                task = self.task_mgr.claim_next(name)
                if task:
                    if len(messages) <= 3:
                        messages.insert(0, {"role": "user", "content":
                            f"<identity>You are '{name}', role: {role}, team: {team_name}.</identity>"})
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})
                    messages.append({"role": "user", "content":
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n{task.get('description', '')}</auto-claimed>"})
                    messages.append({"role": "assistant", "content":
                        f"Claimed task #{task['id']}. Working on it."})
                    if not self._set_active_status(name, "working", stop_event):
                        return "shutdown", "shutdown requested"
                    break
            else:
                return "timed_out", f"idle timeout after {IDLE_TIMEOUT} seconds"

    @staticmethod
    def _print_tool_result(name: str, block, output: ToolResult) -> None:
        text = output.content
        path = ""
        if hasattr(block, "input") and "path" in block.input:
            path = block.input["path"].replace(str(WORKDIR), ".")
        if block.name == "bash":
            print(f"  \033[36m[{name}] > bash:\033[0m")
            print(f"  {text[:120]}")
        elif block.name == "read_file":
            lines = text.count("\n") if output.ok else 0
            if output.ok:
                print(f"  \033[34m[{name}] 📄 read_file: {path} ({lines} 行)\033[0m")
            else:
                print(f"  \033[31m[{name}] ⚠ read_file: {text[:120]}\033[0m")
        elif block.name in ("write_file", "edit_file"):
            print(f"  \033[32m[{name}] ✏️ {block.name}: {path}\033[0m")
        else:
            print(f"  \033[33m[{name}] 🔧 {block.name}: {text[:120]}\033[0m")

    def request_shutdown(self, name: str, reason: str = "shutdown requested") -> ToolResult:
        with self._lock:
            member = self._find_unlocked(name)
            if member is None:
                return ToolResult.failure(
                    f"Unknown teammate: {name}",
                    error_kind=ErrorKind.MODEL,
                )
            event = self._stop_events.get(name)
            thread = self.threads.get(name)
            if event is None or thread is None or not thread.is_alive():
                return ToolResult.success(
                    f"Teammate '{name}' is already {member.get('status')}",
                    exit_code=None,
                )
            self._transition_unlocked(member, "stopping", reason)
            atomic_write_json(self.config_path, self.config)
            event.set()
        return ToolResult.success(f"Shutdown requested for '{name}'", exit_code=None)

    def shutdown_all(self, timeout: float = 5.0) -> dict[str, int]:
        with self._lock:
            self._closing = True
            threads = dict(self.threads)
            for name, thread in threads.items():
                if thread.is_alive():
                    event = self._stop_events.get(name)
                    if event:
                        event.set()
                    member = self._find_unlocked(name)
                    if member:
                        self._transition_unlocked(
                            member,
                            "stopping",
                            "runtime shutdown",
                        )
            atomic_write_json(self.config_path, self.config)
        deadline = time.monotonic() + max(timeout, 0.0)
        for thread in threads.values():
            thread.join(max(0.0, deadline - time.monotonic()))
        return {
            "total": len(threads),
            "still_running": sum(thread.is_alive() for thread in threads.values()),
        }

    def list_all(self) -> str:
        with self._lock:
            team_name = self.config["team_name"]
            members = [dict(member) for member in self.config["members"]]
        if not members:
            return "No teammates."
        lines = [f"Team: {team_name}"]
        for member in members:
            reason = f" ({member['statusReason']})" if member.get("statusReason") else ""
            lines.append(
                f"  {member['name']} ({member['role']}): {member['status']}{reason}"
            )
        return "\n".join(lines)

    def member_names(self) -> list:
        with self._lock:
            return [member["name"] for member in self.config["members"]]
