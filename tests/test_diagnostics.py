import json

from diagnostics import (
    check_model_mapping,
    check_seed_cache,
    render_diagnostics,
    run_diagnostics,
    safe_endpoint,
)
from seed_cache import DEFAULT_EMBEDDING_MODEL, build_seed_cache_document
from utterances import LARGE, SMALL


def _write_valid_seed_cache(path):
    entries = {
        "small": [
            {"text": text, "vector": [1.0, 0.0]}
            for text in SMALL
        ],
        "large": [
            {"text": text, "vector": [0.0, 1.0]}
            for text in LARGE
        ],
    }
    document = build_seed_cache_document(
        entries,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        source_routes={"small": SMALL, "large": LARGE},
    )
    path.write_text(json.dumps(document), encoding="utf-8")


def test_safe_endpoint_removes_credentials_and_query():
    endpoint = safe_endpoint(
        "https://user:password@example.com:444/v1?api_key=secret",
    )

    assert endpoint == "https://example.com:444/v1"
    assert "password" not in endpoint
    assert "secret" not in endpoint


def test_model_mapping_requires_small_and_large(tmp_path):
    path = tmp_path / "litellm_config.yaml"
    path.write_text(json.dumps({
        "model_list": [
            {"model_name": "small"},
            {"model_name": "large"},
        ],
    }), encoding="utf-8")

    assert check_model_mapping(path).status == "ok"

    path.write_text(json.dumps({
        "model_list": [{"model_name": "small"}],
    }), encoding="utf-8")
    result = check_model_mapping(path)
    assert result.status == "error"
    assert "large" in result.message


def test_seed_cache_diagnostic_reports_versioned_cache(tmp_path):
    path = tmp_path / "seed_vectors.json"
    _write_valid_seed_cache(path)

    result = check_seed_cache(path, DEFAULT_EMBEDDING_MODEL)

    assert result.status == "ok"
    assert "dimension=2" in result.message


def test_diagnostics_never_render_env_secret(tmp_path):
    secret = "sk-must-not-be-rendered"
    (tmp_path / ".env").write_text(
        "ANTHROPIC_BASE_URL=http://localhost:4000\n"
        f"ANTHROPIC_API_KEY={secret}\n",
        encoding="utf-8",
    )
    (tmp_path / "litellm_config.yaml").write_text(json.dumps({
        "model_list": [
            {"model_name": "small"},
            {"model_name": "large"},
        ],
    }), encoding="utf-8")
    _write_valid_seed_cache(tmp_path / "seed_vectors.json")

    output = render_diagnostics(run_diagnostics(tmp_path, offline=True))

    assert secret not in output
    assert "value hidden" in output
