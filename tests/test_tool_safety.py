import ast
from pathlib import Path
import subprocess

import pytest

import command_executor
from command_executor import CommandExecutor, CommandRisk
import core_tools
from managers import BackgroundManager
from tool_result import ErrorKind, ToolResult, ToolStatus


ROOT = Path(__file__).resolve().parents[1]


def test_tool_result_uses_typed_status_instead_of_error_prefixes():
    result = ToolResult.failure("command failed", error_kind=ErrorKind.TOOL, exit_code=2)

    assert not result.ok
    assert result.status is ToolStatus.ERROR
    assert result.error_kind is ErrorKind.TOOL
    assert not result.should_record_mistake
    assert "status=error" in result.to_model_content()
    assert not result.content.startswith("Error:")


def test_only_model_failures_are_router_mistakes():
    malformed = ToolResult.failure("bad arguments", error_kind=ErrorKind.MODEL)
    environment = ToolResult.failure("missing executable", error_kind=ErrorKind.ENVIRONMENT)
    permission = ToolResult.denied("not allowed")

    assert malformed.should_record_mistake
    assert not environment.should_record_mistake
    assert not permission.should_record_mistake


def test_direct_command_runs_without_shell_and_without_secrets(tmp_path, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout=b"ok", stderr=b"")

    monkeypatch.setenv("DAEMON_TEST_SECRET", "must-not-leak")
    monkeypatch.setattr(command_executor.subprocess, "run", fake_run)
    executor = CommandExecutor(tmp_path)

    result = executor.execute("python -m pytest --version")

    assert result.ok
    assert captured["shell"] is False
    assert isinstance(captured["argv"], list)
    assert "DAEMON_TEST_SECRET" not in captured["env"]
    assert captured["env"]["PYTHONUTF8"] == "1"


def test_shell_syntax_waits_for_one_time_human_approval(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=b"approved", stderr=b"")

    monkeypatch.setattr(command_executor.subprocess, "run", fake_run)
    executor = CommandExecutor(tmp_path)

    pending = executor.execute("echo one | echo two", requester="lead")

    assert pending.status is ToolStatus.APPROVAL_REQUIRED
    assert calls == []
    request_id = pending.metadata["request_id"]
    approved = executor.approve(request_id)
    repeated = executor.approve(request_id)

    assert approved.ok
    assert len(calls) == 1
    assert calls[0][1]["shell"] is False
    assert repeated.status is ToolStatus.DENIED


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "cat .env",
        "type litellm_config.yaml",
        "python .git/hooks/example.py",
    ],
)
def test_catastrophic_and_protected_commands_are_never_approvable(tmp_path, command):
    executor = CommandExecutor(tmp_path)

    result = executor.execute(command)

    assert result.status is ToolStatus.DENIED
    assert executor.pending() == []


def test_git_push_and_deletion_require_approval(tmp_path):
    executor = CommandExecutor(tmp_path)

    push = executor.prepare("git push origin feature")
    configured_push = executor.prepare("git -c color.ui=false push origin feature")
    deletion = executor.prepare("rm generated.txt")
    package_install = executor.prepare("python -m pip install example")

    assert isinstance(push, ToolResult)
    assert push.status is ToolStatus.APPROVAL_REQUIRED
    assert isinstance(configured_push, ToolResult)
    assert configured_push.status is ToolStatus.APPROVAL_REQUIRED
    assert isinstance(deletion, ToolResult)
    assert deletion.status is ToolStatus.APPROVAL_REQUIRED
    assert isinstance(package_install, ToolResult)
    assert package_install.status is ToolStatus.APPROVAL_REQUIRED


def test_nonzero_command_is_tool_failure_not_model_failure(tmp_path, monkeypatch):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 3, stdout=b"", stderr=b"failed")

    monkeypatch.setattr(command_executor.subprocess, "run", fake_run)
    result = CommandExecutor(tmp_path).execute("python script.py")

    assert result.status is ToolStatus.ERROR
    assert result.error_kind is ErrorKind.TOOL
    assert result.exit_code == 3
    assert not result.should_record_mistake


def test_empty_command_is_a_model_failure(tmp_path):
    result = CommandExecutor(tmp_path).execute("   ")

    assert result.error_kind is ErrorKind.MODEL
    assert result.should_record_mistake


@pytest.mark.parametrize("path", [".env", ".env.local", "litellm_config.yaml", ".git/config"])
def test_file_tools_block_protected_paths(tmp_path, monkeypatch, path):
    monkeypatch.setattr(core_tools, "WORKDIR", tmp_path)

    result = core_tools.run_read(path)

    assert result.status is ToolStatus.ERROR
    assert result.error_kind is ErrorKind.PERMISSION


def test_env_example_remains_editable(tmp_path, monkeypatch):
    monkeypatch.setattr(core_tools, "WORKDIR", tmp_path)

    result = core_tools.run_write(".env.example", "SAFE=value\n")

    assert result.ok
    assert (tmp_path / ".env.example").read_text(encoding="utf-8") == "SAFE=value\n"


def test_atomic_write_failure_preserves_original_file(tmp_path, monkeypatch):
    monkeypatch.setattr(core_tools, "WORKDIR", tmp_path)
    target = tmp_path / "example.txt"
    target.write_text("original", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(core_tools.os, "replace", fail_replace)
    result = core_tools.run_write("example.txt", "replacement")

    assert not result.ok
    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".example.txt.*.tmp")) == []


def test_background_manager_uses_same_approval_policy(tmp_path):
    manager = BackgroundManager(executor=CommandExecutor(tmp_path))

    result = manager.run("echo one | echo two")

    assert result.status is ToolStatus.APPROVAL_REQUIRED
    assert manager.tasks == {}


def test_production_subprocess_calls_never_enable_shell_true():
    offenders = []
    for path in ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == []


def test_readonly_git_command_is_directly_executable():
    plan = CommandExecutor(ROOT).assess("git status --short")

    assert plan.risk is CommandRisk.SAFE
    assert not plan.use_shell_syntax
