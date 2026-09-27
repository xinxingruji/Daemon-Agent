"""Cross-platform command execution with one-time human approvals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
import uuid

from tool_result import ErrorKind, ToolResult


MAX_OUTPUT_CHARS = 50_000


class CommandRisk(str, Enum):
    SAFE = "safe"
    WORKSPACE = "workspace"
    APPROVAL_REQUIRED = "approval_required"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True)
class CommandPlan:
    command: str
    argv: tuple[str, ...]
    risk: CommandRisk
    reason: str
    timeout: int
    requester: str
    use_shell_syntax: bool = False


class CommandExecutor:
    """The only production path from an agent tool to a subprocess."""

    _SHELL_META = re.compile(r"(?:&&|\|\||[|;<>&\n`]|\$\()")
    _FORBIDDEN_PATTERNS = (
        re.compile(r"\brm\s+-[^\n]*r[^\n]*f[^\n]*\s+/(?:\s|$)", re.IGNORECASE),
        re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b", re.IGNORECASE),
        re.compile(r"\bformat(?:\.com)?\s+[a-z]:", re.IGNORECASE),
        re.compile(r"\b(?:del|erase)\s+/[sq].*\s+[a-z]:[\\/]", re.IGNORECASE),
        re.compile(r"\bremove-item\b.*\b-recurse\b.*(?:[a-z]:[\\/]|/)(?:\s|$)", re.IGNORECASE),
    )
    _DELETE_COMMANDS = {"rm", "rmdir", "del", "erase", "remove-item", "rd"}
    _SHELL_PROGRAMS = {
        "bash", "sh", "zsh", "fish", "cmd", "cmd.exe", "powershell",
        "powershell.exe", "pwsh", "pwsh.exe",
    }
    _NETWORK_PROGRAMS = {
        "curl", "curl.exe", "wget", "invoke-webrequest", "scp", "ssh",
    }
    _PACKAGE_PROGRAMS = {"pip", "pip3", "npm", "pnpm", "yarn", "apt", "apt-get", "brew", "winget", "choco"}
    _GIT_APPROVAL_SUBCOMMANDS = {
        "push", "pull", "fetch", "clone", "reset", "clean", "rebase",
        "restore", "rm", "gc", "prune",
    }
    _GIT_READONLY_SUBCOMMANDS = {
        "status", "log", "diff", "show", "rev-parse", "merge-base",
        "ls-files", "ls-tree", "blame", "grep", "describe", "shortlog",
    }
    _WINDOWS_SAFE_BUILTINS = {
        "ls": "dir",
        "ls -l": "dir",
        "ls -la": "dir /a",
        "ls -lh": "dir",
        "ls -lha": "dir /a",
        "ls -al": "dir /a",
        "dir": "dir",
        "dir /a": "dir /a",
        "pwd": "cd",
        "clear": "cls",
        "cls": "cls",
    }

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self._pending: dict[str, CommandPlan] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _strip_windows_quotes(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            return value[1:-1]
        return value

    def _split(self, command: str) -> tuple[str, ...]:
        try:
            parts = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            raise ValueError(f"Cannot parse command: {exc}") from exc
        if os.name == "nt":
            parts = [self._strip_windows_quotes(part) for part in parts]
        if not parts:
            raise ValueError("Command is empty")
        return tuple(parts)

    @staticmethod
    def _references_protected_path(command: str) -> bool:
        normalized = command.lower().replace("\\", "/")
        tokens = re.split(r"\s+", normalized)
        for raw_token in tokens:
            token = raw_token.strip("\"'`()[]{};,:")
            parts = [part for part in token.split("/") if part not in ("", ".")]
            if ".git" in parts:
                return True
            if not parts:
                continue
            name = parts[-1]
            if name == "litellm_config.yaml":
                return True
            if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
                return True
        return False

    @staticmethod
    def _base_program(argv: tuple[str, ...]) -> str:
        return Path(argv[0]).name.lower()

    @staticmethod
    def _git_subcommand(argv: tuple[str, ...]) -> str:
        options_with_values = {
            "-c", "-C", "--exec-path", "--git-dir", "--work-tree",
            "--namespace", "--config-env",
        }
        skip_next = False
        for item in argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if item in options_with_values:
                skip_next = True
                continue
            if not item.startswith("-"):
                return item.lower()
        return ""

    def assess(self, command: str, *, timeout: int = 120, requester: str = "agent") -> CommandPlan:
        if not isinstance(command, str):
            raise TypeError("Command must be a string")
        command = command.strip()
        if not command:
            raise ValueError("Command is empty")
        if timeout < 1 or timeout > 3600:
            raise ValueError("Timeout must be between 1 and 3600 seconds")

        shell_syntax = bool(self._SHELL_META.search(command))
        argv = self._split(command)
        program = self._base_program(argv)
        lowered = command.lower()

        if any(pattern.search(command) for pattern in self._FORBIDDEN_PATTERNS):
            return CommandPlan(
                command, argv, CommandRisk.FORBIDDEN,
                "catastrophic system command", timeout, requester, shell_syntax,
            )
        if self._references_protected_path(command):
            return CommandPlan(
                command, argv, CommandRisk.FORBIDDEN,
                "command references a protected credential or Git-internal path",
                timeout, requester, shell_syntax,
            )
        if len(command) > 10_000:
            return CommandPlan(
                command, argv, CommandRisk.APPROVAL_REQUIRED,
                "unusually long command", timeout, requester, shell_syntax,
            )
        if shell_syntax:
            return CommandPlan(
                command, argv, CommandRisk.APPROVAL_REQUIRED,
                "shell operators require human review", timeout, requester, True,
            )
        if program in self._DELETE_COMMANDS:
            return CommandPlan(
                command, argv, CommandRisk.APPROVAL_REQUIRED,
                "file deletion requires human approval", timeout, requester,
            )
        if program in self._SHELL_PROGRAMS:
            return CommandPlan(
                command, argv, CommandRisk.APPROVAL_REQUIRED,
                "explicit shell execution requires human approval", timeout, requester,
            )
        if program in self._NETWORK_PROGRAMS:
            return CommandPlan(
                command, argv, CommandRisk.APPROVAL_REQUIRED,
                "network command requires human approval", timeout, requester,
            )
        if program in self._PACKAGE_PROGRAMS and any(
            item.lower() in {"install", "add", "remove", "uninstall", "update", "upgrade", "i", "ci"}
            for item in argv[1:]
        ):
            return CommandPlan(
                command, argv, CommandRisk.APPROVAL_REQUIRED,
                "package changes require human approval", timeout, requester,
            )
        if program in {"python", "python.exe", "python3", "py", "node", "node.exe"} and any(
            item in {"-c", "-e", "--eval"} for item in argv[1:]
        ):
            return CommandPlan(
                command, argv, CommandRisk.APPROVAL_REQUIRED,
                "inline code execution requires human approval", timeout, requester,
            )
        if program in {"python", "python.exe", "python3", "py"}:
            lowered_argv = tuple(item.lower() for item in argv[1:])
            if any(
                lowered_argv[index:index + 2] in {("-m", "pip"), ("-m", "ensurepip")}
                for index in range(max(0, len(lowered_argv) - 1))
            ) and any(
                item in {"install", "uninstall", "download", "wheel"}
                for item in lowered_argv
            ):
                return CommandPlan(
                    command, argv, CommandRisk.APPROVAL_REQUIRED,
                    "Python package changes require human approval", timeout, requester,
                )
        if program == "git" and self._git_subcommand(argv) in self._GIT_APPROVAL_SUBCOMMANDS:
            return CommandPlan(
                command, argv, CommandRisk.APPROVAL_REQUIRED,
                f"git {self._git_subcommand(argv)} changes durable or remote state",
                timeout, requester,
            )
        if program == "git" and self._git_subcommand(argv) == "branch" and any(
            item in {"-d", "-D", "--delete"} for item in argv[2:]
        ):
            return CommandPlan(
                command, argv, CommandRisk.APPROVAL_REQUIRED,
                "branch deletion requires human approval", timeout, requester,
            )
        if re.search(r"\b(?:sudo|runas)\b", lowered):
            return CommandPlan(
                command, argv, CommandRisk.APPROVAL_REQUIRED,
                "privilege escalation requires human approval", timeout, requester,
            )

        readonly_programs = {"rg", "grep", "findstr", "where", "which", "ls", "dir", "pwd"}
        if program == "git":
            risk = (
                CommandRisk.SAFE
                if self._git_subcommand(argv) in self._GIT_READONLY_SUBCOMMANDS
                else CommandRisk.WORKSPACE
            )
        else:
            risk = CommandRisk.SAFE if program in readonly_programs else CommandRisk.WORKSPACE
        return CommandPlan(command, argv, risk, "direct execution without shell syntax", timeout, requester)

    def prepare(
        self,
        command: str,
        *,
        timeout: int = 120,
        requester: str = "agent",
    ) -> CommandPlan | ToolResult:
        try:
            plan = self.assess(command, timeout=timeout, requester=requester)
        except (TypeError, ValueError) as exc:
            return ToolResult.failure(str(exc), error_kind=ErrorKind.MODEL)
        if plan.risk is CommandRisk.FORBIDDEN:
            return ToolResult.denied(
                f"Command blocked: {plan.reason}",
                metadata={"risk": plan.risk.value, "requester": requester},
            )
        if plan.risk is CommandRisk.APPROVAL_REQUIRED:
            request_id = uuid.uuid4().hex[:8]
            with self._lock:
                self._pending[request_id] = plan
            return ToolResult.approval_required(
                f"Approval required for command '{command}'. Use /approve {request_id} or /deny {request_id}.",
                metadata={
                    "request_id": request_id,
                    "risk": plan.risk.value,
                    "reason": plan.reason,
                    "requester": requester,
                },
            )
        return plan

    @staticmethod
    def safe_environment(source: dict[str, str] | None = None) -> dict[str, str]:
        source = os.environ if source is None else source
        allowed = {
            "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC",
            "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TERM",
            "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
        }
        env = {key: value for key, value in source.items() if key.upper() in allowed}
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        return env

    @staticmethod
    def _decode_output(data: bytes) -> str:
        for encoding in ("utf-8", "gbk"):
            try:
                return data.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace").strip()

    @staticmethod
    def _probe_exit_is_success(argv: tuple[str, ...], returncode: int) -> bool:
        if returncode != 1 or not argv:
            return False
        return Path(argv[0]).name.lower() in {"grep", "rg", "diff", "cmp"}

    def _runtime_argv(self, plan: CommandPlan) -> list[str]:
        if os.name == "nt":
            safe_builtin = self._WINDOWS_SAFE_BUILTINS.get(plan.command.strip().lower())
            if safe_builtin is not None:
                command_shell = os.environ.get("COMSPEC") or str(
                    Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "cmd.exe"
                )
                return [command_shell, "/d", "/s", "/c", safe_builtin]
        if not plan.use_shell_syntax:
            return list(plan.argv)
        if os.name == "nt":
            command_shell = os.environ.get("COMSPEC") or str(
                Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "cmd.exe"
            )
            return [command_shell, "/d", "/s", "/c", plan.command]
        return ["/bin/sh", "-c", plan.command]

    def run_plan(self, plan: CommandPlan) -> ToolResult:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                self._runtime_argv(plan),
                shell=False,
                cwd=self.workspace,
                env=self.safe_environment(),
                capture_output=True,
                timeout=plan.timeout,
            )
        except FileNotFoundError as exc:
            return ToolResult.failure(
                f"Executable not found: {plan.argv[0]}",
                error_kind=ErrorKind.ENVIRONMENT,
                metadata={"exception": type(exc).__name__, "requester": plan.requester},
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(
                f"Command timed out after {plan.timeout}s",
                error_kind=ErrorKind.ENVIRONMENT,
                metadata={"timeout": plan.timeout, "requester": plan.requester},
            )
        except OSError as exc:
            return ToolResult.failure(
                f"Command could not start: {exc}",
                error_kind=ErrorKind.ENVIRONMENT,
                metadata={"exception": type(exc).__name__, "requester": plan.requester},
            )

        output = self._decode_output(completed.stdout + completed.stderr)
        truncated = len(output) > MAX_OUTPUT_CHARS
        output = output[:MAX_OUTPUT_CHARS]
        elapsed_ms = round((time.monotonic() - started) * 1000)
        metadata = {
            "duration_ms": elapsed_ms,
            "risk": plan.risk.value,
            "requester": plan.requester,
            "truncated": truncated,
        }
        if completed.returncode == 0 or self._probe_exit_is_success(plan.argv, completed.returncode):
            if not output:
                output = "(no output)"
            return ToolResult.success(output, exit_code=completed.returncode, metadata=metadata)
        if not output:
            output = f"Command exited with code {completed.returncode}"
        return ToolResult.failure(
            output,
            error_kind=ErrorKind.TOOL,
            exit_code=completed.returncode,
            metadata=metadata,
        )

    def execute(self, command: str, *, timeout: int = 120, requester: str = "agent") -> ToolResult:
        prepared = self.prepare(command, timeout=timeout, requester=requester)
        if isinstance(prepared, ToolResult):
            return prepared
        return self.run_plan(prepared)

    def pending(self) -> list[dict[str, str | int]]:
        with self._lock:
            items = list(self._pending.items())
        return [
            {
                "request_id": request_id,
                "command": plan.command,
                "reason": plan.reason,
                "requester": plan.requester,
                "timeout": plan.timeout,
            }
            for request_id, plan in items
        ]

    def approve(self, request_id: str) -> ToolResult:
        with self._lock:
            plan = self._pending.pop(request_id, None)
        if plan is None:
            return ToolResult.denied(f"Unknown or already resolved approval request: {request_id}")
        return self.run_plan(plan)

    def deny(self, request_id: str) -> ToolResult:
        with self._lock:
            plan = self._pending.pop(request_id, None)
        if plan is None:
            return ToolResult.denied(f"Unknown or already resolved approval request: {request_id}")
        return ToolResult.denied(
            f"Command denied: {plan.command}",
            metadata={"request_id": request_id, "requester": plan.requester},
        )
