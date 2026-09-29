import json

import pytest

from router_eval import (
    EvaluationCase,
    evaluate_cases,
    evaluate_thresholds,
    load_cases,
    recommended_result,
)


def test_versioned_evaluation_dataset_is_balanced_and_valid():
    cases = load_cases()

    assert len(cases) >= 20
    assert sum(case.expected == "small" for case in cases) == sum(
        case.expected == "large" for case in cases
    )
    assert len({case.id for case in cases}) == len(cases)


def test_default_threshold_has_no_false_downgrades_in_proxy_dataset():
    metrics = evaluate_cases(load_cases(), threshold=0.45)

    assert metrics.false_downgrades == 0
    assert metrics.large_accuracy == 1.0
    assert metrics.accuracy >= 0.80
    assert metrics.mean_policy_latency_ms >= 0


def test_threshold_recommendation_penalizes_false_downgrades():
    cases = load_cases()
    results = evaluate_thresholds(cases, [0.35, 0.40, 0.45])

    recommended = recommended_result(results)

    assert recommended.threshold == 0.40
    assert recommended.false_downgrades == 0


def test_invalid_evaluation_dataset_is_rejected(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schema_version": 99, "cases": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_cases(path)


def test_metrics_distinguish_false_upgrade_and_false_downgrade():
    cases = [
        EvaluationCase("small", "列出文件", "small", "x", "x"),
        EvaluationCase("large", "列出文件", "large", "x", "x"),
    ]

    metrics = evaluate_cases(cases, threshold=0.20)

    assert metrics.false_downgrades == 1
    assert metrics.false_upgrades == 0
