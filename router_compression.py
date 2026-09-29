"""Validation and atomic persistence for Router compression outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Callable, Mapping, Sequence

from embedding_service import normalize_embedding_text


COMPRESSION_LIMITS = {
    "mistake": (10, 20),
    "seed": (5, 8),
}


def parse_compression_response(raw_text: str, target: str) -> list[str]:
    """Parse and strictly validate the model's proposed replacement queries."""
    if target not in COMPRESSION_LIMITS:
        raise ValueError(f"unsupported compression target: {target}")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("compression response must be non-empty text")

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"compression response is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, list):
        raise ValueError("compression response must be a JSON array")

    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(parsed):
        if not isinstance(value, str):
            raise ValueError(f"compression item {index} must be a string")
        text = normalize_embedding_text(value)
        if not text:
            raise ValueError(f"compression item {index} must be non-empty")
        if text in seen:
            raise ValueError(f"compression response contains duplicate item: {text!r}")
        seen.add(text)
        normalized.append(text)

    minimum, maximum = COMPRESSION_LIMITS[target]
    if not minimum <= len(normalized) <= maximum:
        raise ValueError(
            f"{target} compression must contain {minimum}-{maximum} items; "
            f"received {len(normalized)}",
        )
    return normalized


def build_compressed_records(
    queries: Sequence[str],
    *,
    embedding_getter: Callable[[str], list[float]],
    expected_dimension: int,
) -> list[dict[str, object]]:
    """Require every proposed query to produce one valid, consistent vector."""
    if expected_dimension <= 0:
        raise ValueError("expected embedding dimension must be positive")
    records: list[dict[str, object]] = []
    for query in queries:
        vector = embedding_getter(query)
        if (
            not isinstance(vector, list)
            or len(vector) != expected_dimension
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       for value in vector)
        ):
            raise ValueError(
                f"embedding for {query!r} is missing or does not match "
                f"dimension {expected_dimension}",
            )
        records.append({"query": query, "vector": [float(value) for value in vector]})
    return records


def _atomic_replace_bytes(path: str | Path, payload: bytes) -> None:
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


def atomic_write_json(path: str | Path, document: Mapping[str, object]) -> None:
    payload = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_replace_bytes(path, payload)


def atomic_write_jsonl(path: str | Path, records: Sequence[Mapping[str, object]]) -> None:
    text = "".join(
        json.dumps(dict(record), ensure_ascii=False) + "\n"
        for record in records
    )
    _atomic_replace_bytes(path, text.encode("utf-8"))
