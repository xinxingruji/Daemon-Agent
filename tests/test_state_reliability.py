import json
import threading
from types import SimpleNamespace

import pytest

import main
import managers
from managers import BackgroundManager, MessageBus, TaskManager
import state_persistence
from state_persistence import atomic_write_text
import team
from team import TeammateManager
from tool_result import ToolResult, ToolStatus


def create_task(manager: TaskManager, subject: str) -> dict:
    return json.loads(manager.create(subject))


def get_task(manager: TaskManager, task_id: int) -> dict:
    return json.loads(manager.get(task_id))


def test_concurrent_task_creation_allocates_unique_ids(tmp_path):
    task_dir = tmp_path / "tasks"
    managers = [TaskManager(task_dir) for _ in range(3)]
    barrier = threading.Barrier(12)
    created = []
    result_lock = threading.Lock()

    def worker(index):
        barrier.wait(timeout=2)
        task = create_task(managers[index % len(managers)], f"task-{index}")
        with result_lock:
            created.append(task)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(task["id"] for task in created) == list(range(1, 13))
    assert len(list((tmp_path / "tasks").glob("task_*.json"))) == 12


def test_concurrent_claim_has_exactly_one_winner(tmp_path):
    task_dir = tmp_path / "tasks"
    manager = TaskManager(task_dir)
    claimers = [TaskManager(task_dir), TaskManager(task_dir)]
    task_id = create_task(manager, "claim me")["id"]
    barrier = threading.Barrier(8)
    winners = []
    failures = []
    result_lock = threading.Lock()

    def worker(index):
        barrier.wait(timeout=2)
        try:
            claimers[index % len(claimers)].claim(task_id, f"worker-{index}")
            with result_lock:
                winners.append(index)
        except ValueError as exc:
            with result_lock:
                failures.append(str(exc))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    task = get_task(manager, task_id)
    assert len(winners) == 1
    assert len(failures) == 7
    assert task["status"] == "in_progress"
    assert task["owner"] == f"worker-{winners[0]}"


def test_task_transitions_unblock_dependents_and_release_owner(tmp_path):
    manager = TaskManager(tmp_path / "tasks")
    blocker = create_task(manager, "blocker")
    dependent = create_task(manager, "dependent")
    manager.update(dependent["id"], add_blocked_by=[blocker["id"]])
    manager.claim(blocker["id"], "alice")
    manager.update(blocker["id"], status="completed")

    assert get_task(manager, dependent["id"])["blockedBy"] == []
    with pytest.raises(ValueError, match="not available"):
        manager.claim(blocker["id"], "bob")

    releasable = create_task(manager, "releasable")
    manager.claim(releasable["id"], "alice")
    assert manager.release_owner("alice", "agent stopped") == 1
    released = get_task(manager, releasable["id"])
    assert released["status"] == "pending"
    assert released["owner"] is None
    assert released["lastReleaseReason"] == "agent stopped"


def test_failed_task_records_reason_and_rejects_invalid_transition(tmp_path):
    manager = TaskManager(tmp_path / "tasks")
    task = create_task(manager, "failure")
    failed = json.loads(manager.update(
        task["id"],
        status="failed",
        failure_reason="dependency unavailable",
    ))
    assert failed["failureReason"] == "dependency unavailable"
    manager.update(task["id"], status="pending")
    manager.update(task["id"], status="completed")
    with pytest.raises(ValueError, match="completed -> pending"):
        manager.update(task["id"], status="pending")


def test_inbox_concurrent_send_and_drain_loses_no_messages(tmp_path):
    bus = MessageBus(tmp_path / "inbox")
    writer_done = threading.Event()
    received = []

    def writer():
        for index in range(100):
            bus.send("writer", "lead", str(index))
        writer_done.set()

    def reader():
        while not writer_done.is_set():
            received.extend(bus.read_inbox("lead"))
        received.extend(bus.read_inbox("lead"))

    write_thread = threading.Thread(target=writer)
    read_thread = threading.Thread(target=reader)
    write_thread.start()
    read_thread.start()
    write_thread.join(timeout=5)
    read_thread.join(timeout=5)

    assert not write_thread.is_alive()
    assert not read_thread.is_alive()
    assert sorted(int(message["content"]) for message in received) == list(range(100))
    assert bus.read_inbox("lead") == []


def test_invalid_inbox_line_is_not_destructively_drained(tmp_path):
    inbox_dir = tmp_path / "inbox"
    bus = MessageBus(inbox_dir)
    path = inbox_dir / "lead.jsonl"
    original = '{"type":"message","from":"a","content":"ok"}\nnot-json\n'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        bus.read_inbox("lead")

    assert path.read_text(encoding="utf-8") == original


def test_message_metadata_and_inbox_paths_cannot_override_boundaries(tmp_path):
    bus = MessageBus(tmp_path / "inbox")
    with pytest.raises(ValueError, match="replace"):
        bus.send("a", "lead", "hello", extra={"from": "spoofed"})
    with pytest.raises(ValueError, match="invalid inbox"):
        bus.send("a", "../outside", "hello")
    with pytest.raises(ValueError, match="unsupported message"):
        bus.send("a", "lead", "hello", msg_type="unknown")


def test_atomic_state_write_failure_preserves_original(tmp_path, monkeypatch):
    destination = tmp_path / "state.json"
    destination.write_text("original", encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(state_persistence.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_text(destination, "replacement")

    assert destination.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_restart_restores_metadata_but_marks_execution_interrupted(tmp_path):
    team_dir = tmp_path / "team"
    team_dir.mkdir()
    config_path = team_dir / "config.json"
    config_path.write_text(json.dumps({
        "team_name": "demo",
        "members": [
            {"name": "active", "role": "coder", "status": "working"},
            {"name": "done", "role": "reviewer", "status": "shutdown"},
        ],
    }), encoding="utf-8")

    manager = TeammateManager(
        MessageBus(tmp_path / "inbox"),
        TaskManager(tmp_path / "tasks"),
        team_dir=team_dir,
    )

    members = {member["name"]: member for member in manager.config["members"]}
    assert members["active"]["status"] == "interrupted"
    assert "execution context was not persisted" in members["active"]["statusReason"]
    assert members["done"]["status"] == "shutdown"
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["members"][0]["status"] == "interrupted"


def test_subagent_has_explicit_round_budget(monkeypatch):
    tool_call = SimpleNamespace(
        type="tool_use",
        name="read_file",
        input={"path": "README.md"},
        id="call-1",
    )
    response = SimpleNamespace(stop_reason="tool_use", content=[tool_call])
    monkeypatch.setattr(
        team,
        "ROUTER",
        SimpleNamespace(route=lambda **kwargs: "small", record_mistake=lambda query: None),
    )
    monkeypatch.setattr(
        team,
        "client",
        SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response)),
    )
    monkeypatch.setattr(
        team,
        "_invoke_agent_tool",
        lambda *args, **kwargs: ToolResult.success("ok"),
    )

    result = team.run_subagent("loop forever", max_rounds=2)

    assert not result.ok
    assert "round budget exhausted (2)" in result.content
    assert not team.run_subagent("invalid", max_rounds=0).ok


def test_teammate_shutdown_releases_claimed_task(tmp_path, monkeypatch):
    task_manager = TaskManager(tmp_path / "tasks")
    task_id = create_task(task_manager, "owned work")["id"]
    manager = TeammateManager(
        MessageBus(tmp_path / "inbox"),
        task_manager,
        team_dir=tmp_path / "team",
        max_concurrency=1,
    )
    claimed = threading.Event()

    def fake_loop(name, role, prompt, stop_event):
        task_manager.claim(task_id, name)
        claimed.set()
        assert stop_event.wait(timeout=2)
        return "shutdown", "test shutdown"

    monkeypatch.setattr(manager, "_loop", fake_loop)
    assert manager.spawn("worker", "coder", "work").ok
    assert claimed.wait(timeout=2)
    assert manager.request_shutdown("worker").ok
    manager.threads["worker"].join(timeout=2)

    assert not manager.threads["worker"].is_alive()
    task = get_task(task_manager, task_id)
    assert task["status"] == "pending"
    assert task["owner"] is None
    member = manager.config["members"][0]
    assert member["status"] == "shutdown"
    assert member["statusReason"] == "test shutdown"


def test_teammate_concurrency_limit_rejects_extra_worker(tmp_path, monkeypatch):
    manager = TeammateManager(
        MessageBus(tmp_path / "inbox"),
        TaskManager(tmp_path / "tasks"),
        team_dir=tmp_path / "team",
        max_concurrency=1,
    )
    started = threading.Event()

    def waiting_loop(name, role, prompt, stop_event):
        started.set()
        stop_event.wait(timeout=2)
        return "shutdown", "test complete"

    monkeypatch.setattr(manager, "_loop", waiting_loop)
    assert manager.spawn("first", "coder", "work").ok
    assert started.wait(timeout=2)

    rejected = manager.spawn("second", "reviewer", "review")

    assert not rejected.ok
    assert "concurrency limit" in rejected.content
    assert manager.shutdown_all(timeout=2)["still_running"] == 0


class WaitingExecutor:
    def __init__(self):
        self.release = threading.Event()

    def prepare(self, command, *, timeout, requester):
        return object()

    def run_plan(self, plan):
        self.release.wait(timeout=2)
        return ToolResult.success("done")


def test_background_shutdown_waits_and_rejects_new_work():
    executor = WaitingExecutor()
    manager = BackgroundManager(executor=executor)
    assert manager.run("work").ok
    timer = threading.Timer(0.05, executor.release.set)
    timer.start()

    result = manager.shutdown(timeout=1)

    timer.join(timeout=1)
    assert result == {"total": 1, "still_running": 0}
    assert manager.run("late").status is ToolStatus.DENIED


def test_background_start_and_shutdown_are_serialized(monkeypatch):
    real_thread_class = threading.Thread
    start_entered = threading.Event()
    allow_start = threading.Event()

    class DelayedStartThread:
        def __init__(self, *, target, args, daemon, **kwargs):
            self._target = target
            self._args = args
            self._daemon = daemon
            self._thread = None

        def start(self):
            start_entered.set()
            assert allow_start.wait(timeout=2)
            self._thread = real_thread_class(
                target=self._target,
                args=self._args,
                daemon=self._daemon,
            )
            self._thread.start()

        def join(self, timeout=None):
            if self._thread is not None:
                self._thread.join(timeout)

        def is_alive(self):
            return self._thread is not None and self._thread.is_alive()

    monkeypatch.setattr(managers.threading, "Thread", DelayedStartThread)
    manager = BackgroundManager(executor=WaitingExecutor())
    manager.executor.release.set()
    run_result = []
    shutdown_result = []
    shutdown_entered = threading.Event()
    shutdown_done = threading.Event()
    run_thread = real_thread_class(target=lambda: run_result.append(manager.run("work")))

    def shutdown_worker():
        shutdown_entered.set()
        shutdown_result.append(manager.shutdown(timeout=1))
        shutdown_done.set()

    stop_thread = real_thread_class(target=shutdown_worker)
    run_thread.start()
    assert start_entered.wait(timeout=2)
    stop_thread.start()
    assert shutdown_entered.wait(timeout=2)
    assert not shutdown_done.wait(timeout=0.05)
    allow_start.set()
    run_thread.join(timeout=2)
    stop_thread.join(timeout=2)

    assert run_result[0].ok
    assert shutdown_result == [{"total": 1, "still_running": 0}]


def test_teammate_start_and_shutdown_are_serialized(tmp_path, monkeypatch):
    real_thread_class = threading.Thread
    start_entered = threading.Event()
    allow_start = threading.Event()

    class DelayedStartThread:
        def __init__(self, *, target, args, daemon, name=None, **kwargs):
            self._target = target
            self._args = args
            self._daemon = daemon
            self._name = name
            self._thread = None

        def start(self):
            start_entered.set()
            assert allow_start.wait(timeout=2)
            self._thread = real_thread_class(
                target=self._target,
                args=self._args,
                daemon=self._daemon,
                name=self._name,
            )
            self._thread.start()

        def join(self, timeout=None):
            if self._thread is not None:
                self._thread.join(timeout)

        def is_alive(self):
            return self._thread is not None and self._thread.is_alive()

    monkeypatch.setattr(team.threading, "Thread", DelayedStartThread)
    manager = TeammateManager(
        MessageBus(tmp_path / "inbox"),
        TaskManager(tmp_path / "tasks"),
        team_dir=tmp_path / "team",
    )
    monkeypatch.setattr(
        manager,
        "_loop",
        lambda name, role, prompt, stop_event: (
            ("shutdown", "shutdown requested")
            if stop_event.wait(timeout=2)
            else ("failed", "test timeout")
        ),
    )
    spawn_result = []
    shutdown_result = []
    shutdown_entered = threading.Event()
    shutdown_done = threading.Event()
    spawn_thread = real_thread_class(
        target=lambda: spawn_result.append(manager.spawn("worker", "coder", "work"))
    )

    def shutdown_worker():
        shutdown_entered.set()
        shutdown_result.append(manager.shutdown_all(timeout=1))
        shutdown_done.set()

    stop_thread = real_thread_class(target=shutdown_worker)
    spawn_thread.start()
    assert start_entered.wait(timeout=2)
    stop_thread.start()
    assert shutdown_entered.wait(timeout=2)
    assert not shutdown_done.wait(timeout=0.05)
    allow_start.set()
    spawn_thread.join(timeout=2)
    stop_thread.join(timeout=2)

    assert spawn_result[0].ok
    assert shutdown_result == [{"total": 1, "still_running": 0}]
    assert manager.config["members"][0]["status"] == "shutdown"


def test_runtime_shutdown_is_idempotent(monkeypatch):
    calls = []

    class FakeManager:
        def shutdown_all(self, timeout):
            calls.append(("team", timeout))
            return {"total": 0, "still_running": 0}

    class FakeBackground:
        def shutdown(self, timeout):
            calls.append(("background", timeout))
            return {"total": 0, "still_running": 0}

    monkeypatch.setattr(main, "TEAM", FakeManager())
    monkeypatch.setattr(main, "BG", FakeBackground())
    monkeypatch.setattr(main, "_runtime_shutdown", False)

    first = main.shutdown_runtime(timeout=0.25)
    second = main.shutdown_runtime(timeout=0.25)

    assert first["teammates"]["still_running"] == 0
    assert second == {}
    assert calls == [("team", 0.25), ("background", 0.25)]
