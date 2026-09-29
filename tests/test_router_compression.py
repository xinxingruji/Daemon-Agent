import json
from pathlib import Path
from unittest.mock import patch

import pytest

import router_compression
from router_compression import (
    atomic_write_jsonl,
    build_compressed_records,
    parse_compression_response,
)
from tests.conftest import make_router


def valid_queries(count):
    return [f"generalized task {index}" for index in range(count)]


def test_parse_fenced_compression_response():
    raw = "```json\n" + json.dumps(valid_queries(5)) + "\n```"

    assert parse_compression_response(raw, "seed") == valid_queries(5)


@pytest.mark.parametrize(
    "payload, message",
    [
        (json.dumps(valid_queries(4)), "5-8"),
        (json.dumps(valid_queries(9)), "5-8"),
        (json.dumps(["ok"] * 5), "duplicate"),
        (json.dumps([1, 2, 3, 4, 5]), "string"),
        (json.dumps({"items": valid_queries(5)}), "array"),
        ("not-json", "valid JSON"),
    ],
)
def test_invalid_compression_response_is_rejected(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_compression_response(payload, "seed")


def test_all_compressed_embeddings_must_match_dimension():
    queries = valid_queries(5)

    def getter(query):
        return [1.0, 2.0] if query != queries[-1] else [1.0]

    with pytest.raises(ValueError, match="dimension 2"):
        build_compressed_records(
            queries,
            embedding_getter=getter,
            expected_dimension=2,
        )


def test_atomic_jsonl_failure_preserves_original(tmp_path, monkeypatch):
    destination = tmp_path / "mistakes.json"
    destination.write_text("original\n", encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(router_compression.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_jsonl(destination, [{"query": "new", "vector": [1.0]}])

    assert destination.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob(".mistakes.json.*.tmp")) == []


def test_mistake_compression_keeps_arrivals_and_persists_before_swap(tmp_path):
    path = tmp_path / "mistakes.json"
    router = make_router(mistake_file=str(path))
    snapshot = [
        {"query": "old-1", "vector": [1.0] * 768},
        {"query": "old-2", "vector": [1.0] * 768},
    ]
    arrival = {"query": "new-arrival", "vector": [0.5] * 768}
    replacement = [{"query": "compressed", "vector": [0.25] * 768}]
    router.mistake_book = snapshot + [arrival]

    router._apply_compressed_mistakes(snapshot, replacement)

    assert router.mistake_book == replacement + [arrival]
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines == replacement + [arrival]


def test_seed_persistence_failure_does_not_mutate_memory():
    router = make_router()
    base_texts = list(router.route_embeddings_text["small"])
    base_vectors = [list(vector) for vector in router.route_embeddings["small"]]
    router.base_small_count = len(base_texts)
    snapshot = ["dynamic-1"]
    router.route_embeddings_text["small"].append(snapshot[0])
    router.route_embeddings["small"].append([0.4] * 768)
    before_texts = list(router.route_embeddings_text["small"])
    before_vectors = [list(vector) for vector in router.route_embeddings["small"]]
    records = [{"query": "compressed", "vector": [0.7] * 768}]

    with patch.object(router, "_save_seed_vectors_data", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            router._apply_compressed_seeds(snapshot, records)

    assert router.route_embeddings_text["small"] == before_texts
    assert router.route_embeddings["small"] == before_vectors
    assert router.route_embeddings_text["small"][:len(base_texts)] == base_texts
    assert router.route_embeddings["small"][:len(base_vectors)] == base_vectors


def test_stale_seed_snapshot_is_rejected_without_mutation():
    router = make_router()
    router.base_small_count = len(router.route_embeddings_text["small"])
    router.route_embeddings_text["small"].append("current-dynamic")
    router.route_embeddings["small"].append([0.4] * 768)
    before_texts = list(router.route_embeddings_text["small"])
    before_vectors = [list(vector) for vector in router.route_embeddings["small"]]

    with pytest.raises(RuntimeError, match="snapshot changed"):
        router._apply_compressed_seeds(
            ["stale-dynamic"],
            [{"query": "compressed", "vector": [0.7] * 768}],
        )

    assert router.route_embeddings_text["small"] == before_texts
    assert router.route_embeddings["small"] == before_vectors
