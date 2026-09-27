import json

import precompute_seeds
from seed_cache import CACHE_SCHEMA_VERSION


def _configure_small_fixture(monkeypatch, tmp_path):
    output = tmp_path / "seed_vectors.json"
    monkeypatch.setattr(precompute_seeds, "OUTPUT", output)
    monkeypatch.setattr(precompute_seeds, "SMALL", ["small seed"])
    monkeypatch.setattr(precompute_seeds, "LARGE", ["large seed"])
    monkeypatch.setattr(precompute_seeds, "MAX_WORKERS", 1)
    return output


def test_partial_precompute_does_not_overwrite_existing_cache(monkeypatch, tmp_path):
    output = _configure_small_fixture(monkeypatch, tmp_path)
    output.write_text("existing cache", encoding="utf-8")
    monkeypatch.setattr(
        precompute_seeds,
        "_get_embedding",
        lambda text: [] if text == "large seed" else [1.0, 0.0],
    )

    assert precompute_seeds.main() == 1
    assert output.read_text(encoding="utf-8") == "existing cache"
    assert not output.with_suffix(".json.tmp").exists()


def test_complete_precompute_writes_versioned_cache(monkeypatch, tmp_path):
    output = _configure_small_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        precompute_seeds, "_get_embedding", lambda text: [1.0, 0.0],
    )

    assert precompute_seeds.main() == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["_meta"]["schema_version"] == CACHE_SCHEMA_VERSION
    assert data["_meta"]["embedding_model"] == precompute_seeds.MODEL_NAME
    assert data["_meta"]["embedding_dimension"] == 2
    assert len(data["small"]) == 1
    assert len(data["large"]) == 1
