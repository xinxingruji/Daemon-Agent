# 所有负责处理数据结构、通信、后台任务的类都放在这里。


import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Queue
from command_executor import CommandExecutor, CommandPlan
from config import INBOX_DIR, TASKS_DIR, VALID_MSG_TYPES
from core_tools import get_command_executor
from state_persistence import atomic_write_json, atomic_write_text, shared_path_lock
from tool_result import ErrorKind, ToolResult

# === SECTION: todos (s03) ===
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        validated, ip = [], 0
        for i, item in enumerate(items):
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            af = str(item.get("activeForm", "")).strip()
            if not content: raise ValueError(f"Item {i}: content required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {i}: invalid status '{status}'")
            if not af: raise ValueError(f"Item {i}: activeForm required")
            if status == "in_progress": ip += 1
            validated.append({"content": content, "status": status, "activeForm": af})
        if len(validated) > 20: raise ValueError("Max 20 todos")
        if ip > 1: raise ValueError("Only one in_progress allowed")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items: return "No todos."
        lines = []
        for item in self.items:
            m = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}.get(item["status"], "[?]")
            suffix = f" <- {item['activeForm']}" if item["status"] == "in_progress" else ""
            lines.append(f"{m} {item['content']}{suffix}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

    def has_open_items(self) -> bool:
        return any(item.get("status") != "completed" for item in self.items)
    
# === SECTION: skills (s05) ===
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills = {}
        if skills_dir.exists():
            for f in sorted(skills_dir.rglob("SKILL.md")):
                text = f.read_text()
                match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
                meta, body = {}, text
                if match:
                    for line in match.group(1).strip().splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip()
                    body = match.group(2).strip()
                name = meta.get("name", f.parent.name)
                self.skills[name] = {"meta": meta, "body": body}

    def descriptions(self) -> str:
        if not self.skills: return "(no skills)"
        return "\n".join(f"  - {n}: {s['meta'].get('description', '-')}" for n, s in self.skills.items())

    def load(self, name: str) -> ToolResult:
        s = self.skills.get(name)
        if not s:
            return ToolResult.failure(
                f"Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}",
                error_kind=ErrorKind.MODEL,
            )
        return ToolResult.success(
            f"<skill name=\"{name}\">\n{s['body']}\n</skill>",
            exit_code=None,
        )
    
# === SECTION: file_tasks (s07) ===
class TaskManager:
    ACTIVE_STATUSES = {"pending", "in_progress"}
    TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
    VALID_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
    ALLOWED_TRANSITIONS = {
        "pending": {"pending", "in_progress", "completed", "failed", "cancelled"},
        "in_progress": {"pending", "in_progress", "completed", "failed", "cancelled"},
        "failed": {"failed", "pending", "in_progress", "cancelled"},
        "cancelled": {"cancelled", "pending"},
        "completed": {"completed"},
    }

    def __init__(self, task_dir: str | Path | None = None):
        self.task_dir = Path(task_dir) if task_dir is not None else TASKS_DIR
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._lock = shared_path_lock(self.task_dir)

    def _path(self, tid: int) -> Path:
        if not isinstance(tid, int) or isinstance(tid, bool) or tid <= 0:
            raise ValueError("task_id must be a positive integer")
        return self.task_dir / f"task_{tid}.json"

    def _task_files(self) -> list[Path]:
        def task_id(path: Path) -> int:
            try:
                return int(path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                return 2**63 - 1
        return sorted(self.task_dir.glob("task_*.json"), key=task_id)

    def _next_id_unlocked(self) -> int:
        ids = []
        for path in self._task_files():
            try:
                ids.append(int(path.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max(ids, default=0) + 1

    def _load_unlocked(self, tid: int) -> dict:
        path = self._path(tid)
        if not path.exists():
            raise ValueError(f"Task {tid} not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Task {tid} is not a JSON object")
        return data

    def _save_unlocked(self, task: dict) -> None:
        task["updatedAt"] = time.time()
        atomic_write_json(self._path(task["id"]), task)

    def _all_unlocked(self) -> list[dict]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in self._task_files()
        ]

    def create(self, subject: str, description: str = "") -> str:
        subject = str(subject).strip()
        if not subject:
            raise ValueError("task subject is required")
        with self._lock:
            now = time.time()
            task = {
                "id": self._next_id_unlocked(),
                "subject": subject,
                "description": str(description),
                "status": "pending",
                "owner": None,
                "blockedBy": [],
                "createdAt": now,
                "updatedAt": now,
                "failureReason": None,
            }
            self._save_unlocked(task)
            return json.dumps(task, indent=2)

    def get(self, tid: int) -> str:
        with self._lock:
            return json.dumps(self._load_unlocked(tid), indent=2)

    def update(
        self,
        tid: int,
        status: str | None = None,
        add_blocked_by: list | None = None,
        remove_blocked_by: list | None = None,
        failure_reason: str | None = None,
    ) -> str:
        with self._lock:
            task = self._load_unlocked(tid)
            if status == "deleted":
                self._path(tid).unlink(missing_ok=True)
                for dependent in self._all_unlocked():
                    if tid in dependent.get("blockedBy", []):
                        dependent["blockedBy"].remove(tid)
                        self._save_unlocked(dependent)
                return f"Task {tid} deleted"
            if status:
                if status not in self.VALID_STATUSES:
                    raise ValueError(f"invalid task status: {status}")
                current = task.get("status", "pending")
                if status not in self.ALLOWED_TRANSITIONS.get(current, set()):
                    raise ValueError(f"invalid task transition: {current} -> {status}")
                task["status"] = status
                if status == "pending":
                    task["owner"] = None
                if status == "failed":
                    task["failureReason"] = str(failure_reason or "unspecified failure")
                elif status in {"pending", "in_progress", "completed"}:
                    task["failureReason"] = None
                if status == "completed":
                    for dependent in self._all_unlocked():
                        if tid in dependent.get("blockedBy", []):
                            dependent["blockedBy"].remove(tid)
                            self._save_unlocked(dependent)
            blocked = set(task.get("blockedBy", []))
            for blocker in add_blocked_by or []:
                if blocker == tid:
                    raise ValueError("a task cannot block itself")
                self._load_unlocked(blocker)
                blocked.add(blocker)
            blocked.difference_update(remove_blocked_by or [])
            task["blockedBy"] = sorted(blocked)
            self._save_unlocked(task)
            return json.dumps(task, indent=2)

    def list_all(self) -> str:
        with self._lock:
            tasks = self._all_unlocked()
        if not tasks:
            return "No tasks."
        lines = []
        markers = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
            "failed": "[!]",
            "cancelled": "[-]",
        }
        for task in tasks:
            marker = markers.get(task.get("status"), "[?]")
            owner = f" @{task['owner']}" if task.get("owner") else ""
            blocked = f" (blocked by: {task['blockedBy']})" if task.get("blockedBy") else ""
            reason = f" (reason: {task['failureReason']})" if task.get("failureReason") else ""
            lines.append(f"{marker} #{task['id']}: {task['subject']}{owner}{blocked}{reason}")
        return "\n".join(lines)

    def claim(self, tid: int, owner: str) -> str:
        owner = str(owner).strip()
        if not owner:
            raise ValueError("task owner is required")
        with self._lock:
            task = self._load_unlocked(tid)
            if (
                task.get("status") == "in_progress"
                and task.get("owner") == owner
            ):
                return f"Task #{tid} is already claimed by {owner}"
            if task.get("status") != "pending" or task.get("owner"):
                raise ValueError(f"Task {tid} is not available for claim")
            if task.get("blockedBy"):
                raise ValueError(f"Task {tid} is blocked by {task['blockedBy']}")
            task["owner"] = owner
            task["status"] = "in_progress"
            task["failureReason"] = None
            self._save_unlocked(task)
            return f"Claimed task #{tid} for {owner}"

    def claim_next(self, owner: str) -> dict | None:
        owner = str(owner).strip()
        if not owner:
            raise ValueError("task owner is required")
        with self._lock:
            for task in self._all_unlocked():
                if (
                    task.get("status") == "pending"
                    and not task.get("owner")
                    and not task.get("blockedBy")
                ):
                    task["owner"] = owner
                    task["status"] = "in_progress"
                    task["failureReason"] = None
                    self._save_unlocked(task)
                    return dict(task)
        return None

    def release_owner(self, owner: str, reason: str) -> int:
        released = 0
        with self._lock:
            for task in self._all_unlocked():
                if task.get("owner") == owner and task.get("status") == "in_progress":
                    task["owner"] = None
                    task["status"] = "pending"
                    task["lastReleaseReason"] = str(reason)
                    self._save_unlocked(task)
                    released += 1
        return released
    
# === SECTION: background (s08) ===
class BackgroundManager:
    def __init__(self, executor: CommandExecutor | None = None):
        self.tasks = {}
        self.notifications = Queue()
        self.executor = executor or get_command_executor()
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._accepting = True

    def run(self, command: str, timeout: int = 120) -> ToolResult:
        with self._lock:
            if not self._accepting:
                return ToolResult.denied("Background manager is shutting down")
        prepared = self.executor.prepare(
            command,
            timeout=timeout,
            requester="background",
        )
        if isinstance(prepared, ToolResult):
            return prepared
        with self._lock:
            if not self._accepting:
                return ToolResult.denied("Background manager is shutting down")
            tid = str(uuid.uuid4())[:8]
            thread = threading.Thread(target=self._exec, args=(tid, prepared), daemon=True)
            self.tasks[tid] = {"status": "running", "command": command, "result": None}
            self._threads[tid] = thread
            try:
                thread.start()
            except Exception:
                self.tasks.pop(tid, None)
                self._threads.pop(tid, None)
                raise
        return ToolResult.success(
            f"Background task {tid} started: {command[:80]}",
            exit_code=None,
            metadata={"task_id": tid},
        )

    def _exec(self, tid: str, plan: CommandPlan):
        try:
            result = self.executor.run_plan(plan)
        except Exception as exc:
            result = ToolResult.failure(
                f"Background command failed internally: {exc}",
                error_kind=ErrorKind.INTERNAL,
            )
        status = "completed" if result.ok else "error"
        with self._lock:
            self.tasks[tid].update({"status": status, "result": result})
        self.notifications.put({"task_id": tid, "status": status,
                                "result": result.to_model_content()[:500]})

    def check(self, tid: str = None) -> ToolResult:
        if tid:
            with self._lock:
                stored = self.tasks.get(tid)
                t = dict(stored) if stored else None
            if not t:
                return ToolResult.failure(f"Unknown background task: {tid}", error_kind=ErrorKind.MODEL)
            result = t.get("result")
            if isinstance(result, ToolResult):
                return ToolResult(
                    status=result.status,
                    content=f"[{t['status']}] {result.content}",
                    error_kind=result.error_kind,
                    exit_code=result.exit_code,
                    metadata=result.metadata,
                )
            return ToolResult.success(f"[{t['status']}] (running)", exit_code=None)
        with self._lock:
            snapshot = {key: dict(value) for key, value in self.tasks.items()}
        listing = "\n".join(
            f"{key}: [{value['status']}] {value['command'][:60]}"
            for key, value in snapshot.items()
        ) or "No bg tasks."
        return ToolResult.success(listing, exit_code=None)

    def drain(self) -> list:
        notifs = []
        while True:
            try:
                notifs.append(self.notifications.get_nowait())
            except Empty:
                break
        return notifs

    def shutdown(self, timeout: float = 5.0) -> dict[str, int]:
        with self._lock:
            self._accepting = False
            threads = list(self._threads.values())
        deadline = time.monotonic() + max(timeout, 0.0)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        return {
            "total": len(threads),
            "still_running": sum(thread.is_alive() for thread in threads),
        }
    
# === SECTION: messaging (s09) ===
class MessageBus:
    def __init__(self, inbox_dir: str | Path | None = None):
        self.inbox_dir = Path(inbox_dir) if inbox_dir is not None else INBOX_DIR
        self.inbox_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        name = str(name).strip()
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or ":" in name
            or len(name) > 128
        ):
            raise ValueError("invalid inbox name")
        return self.inbox_dir / f"{name}.jsonl"

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            raise ValueError(f"unsupported message type: {msg_type}")
        msg = {"type": msg_type, "from": sender, "content": content,
               "timestamp": time.time()}
        if extra:
            reserved = {"type", "from", "content", "timestamp"}
            overlap = reserved.intersection(extra)
            if overlap:
                raise ValueError(f"message metadata cannot replace: {sorted(overlap)}")
            msg.update(extra)
        path = self._path(to)
        with shared_path_lock(path):
            with path.open("a", encoding="utf-8") as inbox:
                inbox.write(json.dumps(msg, ensure_ascii=False) + "\n")
                inbox.flush()
                os.fsync(inbox.fileno())
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        path = self._path(name)
        with shared_path_lock(path):
            if not path.exists():
                return []
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                return []
            messages = [json.loads(line) for line in text.splitlines() if line.strip()]
            atomic_write_text(path, "")
            return messages

    def broadcast(self, sender: str, content: str, names: list) -> str:
        count = 0
        for n in names:
            if n != sender:
                self.send(sender, n, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"
