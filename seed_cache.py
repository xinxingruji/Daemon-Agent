"""Seed-vector cache schema and validation helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CACHE_SCHEMA_VERSION = 1
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v2-moe"
DEFAULT_EMBEDDING_URL = "http://localhost:11434/api/embeddings"
ROUTE_NAMES = ("small", "large")


@dataclass(frozen=True)
class SeedCacheReport:
    valid: bool
    legacy: bool
    dimension: int | None
    counts: dict[str, int]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def compute_source_hash(routes: Mapping[str, Sequence[str]]) -> str:
    """Return a stable hash of the built-in seed utterances."""
    normalized = {name: list(routes.get(name, ())) for name in ROUTE_NAMES}
    payload = json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_seed_cache_document(
    entries: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    embedding_model: str,
    source_routes: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Build a versioned cache document and reject inconsistent vectors."""
    route_data = {name: [dict(item) for item in entries.get(name, ())]
                  for name in ROUTE_NAMES}
    dimensions: set[int] = set()
    for route_name in ROUTE_NAMES:
        if not route_data[route_name]:
            raise ValueError(f"{route_name} must contain at least one seed")
        for index, item in enumerate(route_data[route_name]):
            text = item.get("text")
            vector = item.get("vector")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{route_name}[{index}].text must be non-empty")
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"{route_name}[{index}].vector must be non-empty")
            if not all(isinstance(value, (int, float)) for value in vector):
                raise ValueError(f"{route_name}[{index}].vector must be numeric")
            dimensions.add(len(vector))
        actual_texts = {item["text"] for item in route_data[route_name]}
        missing = [
            text for text in source_routes.get(route_name, ())
            if text not in actual_texts
        ]
        if missing:
            raise ValueError(
                f"{route_name} is missing {len(missing)} built-in seed(s)",
            )
    if len(dimensions) != 1:
        raise ValueError("seed vectors must be non-empty and share one dimension")

    dimension = dimensions.pop()
    return {
        "_meta": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "embedding_model": embedding_model,
            "embedding_dimension": dimension,
            "source_hash": compute_source_hash(source_routes),
        },
        **route_data,
    }


def validate_seed_cache(
    data: Any,
    *,
    expected_model: str,
    source_routes: Mapping[str, Sequence[str]],
) -> SeedCacheReport:
    """Validate both versioned and legacy cache documents without mutating them."""
    errors: list[str] = []
    warnings: list[str] = []
    counts = {name: 0 for name in ROUTE_NAMES}
    detected_dimensions: set[int] = set()
    route_texts = {name: set() for name in ROUTE_NAMES}

    if not isinstance(data, dict):
        return SeedCacheReport(
            False, False, None, counts, ("cache root must be an object",), (),
        )

    meta = data.get("_meta")
    legacy = meta is None
    if legacy:
        warnings.append(
            "legacy cache has no metadata; run precompute_seeds.py to upgrade it",
        )
    elif not isinstance(meta, dict):
        errors.append("_meta must be an object")
        meta = {}

    for route_name in ROUTE_NAMES:
        entries = data.get(route_name)
        if not isinstance(entries, list):
            errors.append(f"{route_name} must be a list")
            continue
        counts[route_name] = len(entries)
        if not entries:
            errors.append(f"{route_name} must contain at least one seed")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"{route_name}[{index}] must be an object")
                continue
            if not isinstance(entry.get("text"), str) or not entry["text"].strip():
                errors.append(f"{route_name}[{index}].text must be non-empty")
            else:
                route_texts[route_name].add(entry["text"])
            vector = entry.get("vector")
            if not isinstance(vector, list) or not vector:
                errors.append(f"{route_name}[{index}].vector must be non-empty")
                continue
            if not all(isinstance(value, (int, float)) for value in vector):
                errors.append(f"{route_name}[{index}].vector must be numeric")
                continue
            detected_dimensions.add(len(vector))

    if len(detected_dimensions) > 1:
        errors.append("seed vectors use inconsistent dimensions")
    dimension = next(iter(detected_dimensions), None)

    for route_name in ROUTE_NAMES:
        missing = [
            text for text in source_routes.get(route_name, ())
            if text not in route_texts[route_name]
        ]
        if missing:
            errors.append(
                f"{route_name} is missing {len(missing)} built-in seed(s)",
            )

    if not legacy:
        if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
            errors.append(
                f"unsupported schema_version {meta.get('schema_version')!r}; "
                f"expected {CACHE_SCHEMA_VERSION}",
            )
        if meta.get("embedding_model") != expected_model:
            errors.append(
                "embedding model mismatch: "
                f"cache={meta.get('embedding_model')!r}, expected={expected_model!r}",
            )
        expected_hash = compute_source_hash(source_routes)
        if meta.get("source_hash") != expected_hash:
            errors.append("built-in seed utterances changed since cache generation")
        if meta.get("embedding_dimension") != dimension:
            errors.append(
                "embedding_dimension metadata does not match stored vectors",
            )

    return SeedCacheReport(
        not errors, legacy, dimension, counts, tuple(errors), tuple(warnings),
    )
