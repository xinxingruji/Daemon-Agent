"""Structured results shared by every agent tool execution path."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    APPROVAL_REQUIRED = "approval_required"
    DENIED = "denied"


class ErrorKind(str, Enum):
    MODEL = "model"
    TOOL = "tool"
    ENVIRONMENT = "environment"
    PERMISSION = "permission"
    INTERNAL = "internal"


@dataclass(frozen=True)
class ToolResult:
    """A typed tool outcome with a backwards-compatible text boundary."""

    status: ToolStatus
    content: str
    error_kind: ErrorKind | None = None
    exit_code: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        content: str,
        *,
        exit_code: int | None = 0,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            status=ToolStatus.SUCCESS,
            content=str(content),
            exit_code=exit_code,
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        content: str,
        *,
        error_kind: ErrorKind,
        exit_code: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            status=ToolStatus.ERROR,
            content=str(content),
            error_kind=error_kind,
            exit_code=exit_code,
            metadata=metadata or {},
        )

    @classmethod
    def approval_required(
        cls,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            status=ToolStatus.APPROVAL_REQUIRED,
            content=str(content),
            error_kind=ErrorKind.PERMISSION,
            metadata=metadata or {},
        )

    @classmethod
    def denied(
        cls,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            status=ToolStatus.DENIED,
            content=str(content),
            error_kind=ErrorKind.PERMISSION,
            metadata=metadata or {},
        )

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.SUCCESS

    @property
    def should_record_mistake(self) -> bool:
        """Only malformed model tool use is a router-learning signal."""
        return self.error_kind is ErrorKind.MODEL

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status.value,
            "content": self.content,
        }
        if self.error_kind is not None:
            data["error_kind"] = self.error_kind.value
        if self.exit_code is not None:
            data["exit_code"] = self.exit_code
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    def to_model_content(self) -> str:
        """Keep successful output familiar while labeling non-success states."""
        if self.ok:
            return self.content
        kind = f" error_kind={self.error_kind.value}" if self.error_kind else ""
        return f"[tool status={self.status.value}{kind}] {self.content}"

    def __str__(self) -> str:
        return self.content


def ensure_tool_result(value: object) -> ToolResult:
    """Wrap legacy successful string handlers during the migration."""
    if isinstance(value, ToolResult):
        return value
    return ToolResult.success(str(value), exit_code=None)
