import copy

import pytest

from seed_cache import (
    CACHE_SCHEMA_VERSION,
    build_seed_cache_document,
    compute_source_hash,
    validate_seed_cache,
)


ROUTES = {"small": ["list files"], "large": ["design architecture"]}
ENTRIES = {
    "small": [{"text": "list files", "vector": [1.0, 0.0]}],
    "large": [{"text": "design architecture", "vector": [0.0, 1.0]}],
}


def test_build_versioned_seed_cache():
    document = build_seed_cache_document(
        ENTRIES, embedding_model="embed-model", source_routes=ROUTES,
    )

    assert document["_meta"] == {
        "schema_version": CACHE_SCHEMA_VERSION,
        "embedding_model": "embed-model",
        "embedding_dimension": 2,
        "source_hash": compute_source_hash(ROUTES),
    }
    report = validate_seed_cache(
        document, expected_model="embed-model", source_routes=ROUTES,
    )
    assert report.valid
    assert not report.legacy
    assert report.dimension == 2


def test_legacy_seed_cache_is_accepted_with_warning():
    report = validate_seed_cache(
        copy.deepcopy(ENTRIES),
        expected_model="embed-model",
        source_routes=ROUTES,
    )

    assert report.valid
    assert report.legacy
    assert report.warnings

    incomplete = validate_seed_cache(
        copy.deepcopy(ENTRIES),
        expected_model="embed-model",
        source_routes={
            "small": ["list files", "read file"],
            "large": ["design architecture"],
        },
    )
    assert not incomplete.valid
    assert any("missing 1 built-in" in error for error in incomplete.errors)


@pytest.mark.parametrize("field,value,error_fragment", [
    ("embedding_model", "other-model", "model mismatch"),
    ("embedding_dimension", 3, "does not match"),
    ("source_hash", "stale", "utterances changed"),
])
def test_versioned_cache_metadata_mismatch(field, value, error_fragment):
    document = build_seed_cache_document(
        ENTRIES, embedding_model="embed-model", source_routes=ROUTES,
    )
    document["_meta"][field] = value

    report = validate_seed_cache(
        document, expected_model="embed-model", source_routes=ROUTES,
    )

    assert not report.valid
    assert any(error_fragment in error for error in report.errors)


def test_inconsistent_vector_dimensions_are_rejected():
    entries = copy.deepcopy(ENTRIES)
    entries["large"][0]["vector"] = [0.0, 1.0, 2.0]

    with pytest.raises(ValueError, match="share one dimension"):
        build_seed_cache_document(
            entries, embedding_model="embed-model", source_routes=ROUTES,
        )
