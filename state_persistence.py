"""Small atomic persistence primitives for thread-based runtime state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import threading
from typing import Mapping


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


def shared_path_lock(path: str | Path) -> threading.RLock:
    """Return one process-wide lock for all users of the same resolved path."""
    key = Path(path).resolve()
    with _LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous_mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else None
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        if previous_mode is not None:
            os.chmod(temporary_name, previous_mode)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, document: Mapping[str, object]) -> None:
    atomic_write_text(path, json.dumps(document, ensure_ascii=False, indent=2))
