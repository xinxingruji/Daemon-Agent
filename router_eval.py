"""Offline, reproducible Router evaluation and threshold comparison.

The default hashing embedder is a lexical proxy for CI and regression trends. It
does not claim to measure the quality of the production semantic embedding model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Iterable, Sequence

from routing_policy import RoutingPolicy
from utterances import LARGE, SMALL


EVAL_SCHEMA_VERSION = 1
DEFAULT_CASES_FILE = Path(__file__).with_name("routing_eval_cases.json")


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    query: str
    expected: str
    category: str
    reason: str
    total_tokens: int = 0


@dataclass(frozen=True)
class EvaluationMetrics:
    threshold: float
    total: int
    correct: int
    accuracy: float
    small_accuracy: float
    large_accuracy: float
    false_downgrades: int
    false_upgrades: int
    estimated_cost_units: int
    mean_policy_latency_ms: float
    predictions: tuple[dict[str, object], ...]


def load_cases(path: str | Path = DEFAULT_CASES_FILE) -> list[EvaluationCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != EVAL_SCHEMA_VERSION:
        raise ValueError(f"evaluation schema_version must be {EVAL_SCHEMA_VERSION}")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("evaluation cases must be a non-empty list")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"case {index} must be an object")
        required = ("id", "query", "expected", "category", "reason")
        if any(not isinstance(item.get(field), str) or not item[field].strip()
               for field in required):
            raise ValueError(f"case {index} has missing or empty string fields")
        if item["expected"] not in {"small", "large"}:
            raise ValueError(f"case {index} expected must be small or large")
        if item["id"] in seen_ids or item["query"] in seen_queries:
            raise ValueError(f"case {index} duplicates an id or query")
        seen_ids.add(item["id"])
        seen_queries.add(item["query"])
        total_tokens = item.get("total_tokens", 0)
        if not isinstance(total_tokens, int) or total_tokens < 0:
            raise ValueError(f"case {index} total_tokens must be non-negative")
        cases.append(EvaluationCase(
            id=item["id"],
            query=item["query"],
            expected=item["expected"],
            category=item["category"],
            reason=item["reason"],
            total_tokens=total_tokens,
        ))
    return cases


def _tokens(text: str) -> list[str]:
    lowered = text.casefold()
    latin = re.findall(r"[a-z0-9_]+", lowered)
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    cjk: list[str] = []
    for run in cjk_runs:
        cjk.extend(run)
        cjk.extend(run[index:index + 2] for index in range(len(run) - 1))
    return latin + cjk


def hashing_embedding(text: str, dimension: int = 256) -> list[float]:
    """Create a stable signed hashing vector for offline proxy evaluation."""
    vector = [0.0] * dimension
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def evaluate_cases(
    cases: Sequence[EvaluationCase],
    *,
    threshold: float,
    mistake_threshold: float = 0.75,
) -> EvaluationMetrics:
    policy = RoutingPolicy(
        threshold=threshold,
        mistake_threshold=mistake_threshold,
        safe_tokens=3000,
        penalty_step=4000,
    )
    route_embeddings = {
        "small": [hashing_embedding(text) for text in SMALL],
        "large": [hashing_embedding(text) for text in LARGE],
    }
    predictions: list[dict[str, object]] = []
    latencies: list[float] = []
    for case in cases:
        query_vector = hashing_embedding(case.query)
        started = time.perf_counter()
        decision = policy.decide(
            query_vector,
            route_embeddings=route_embeddings,
            total_tokens=case.total_tokens,
        )
        latencies.append((time.perf_counter() - started) * 1000)
        predictions.append({
            "id": case.id,
            "expected": case.expected,
            "predicted": decision.route,
            "correct": decision.route == case.expected,
            "score": decision.highest_score,
            "dynamic_threshold": decision.dynamic_threshold,
            "category": case.category,
        })

    total = len(cases)
    correct = sum(bool(item["correct"]) for item in predictions)
    small_items = [item for item in predictions if item["expected"] == "small"]
    large_items = [item for item in predictions if item["expected"] == "large"]
    false_downgrades = sum(
        item["expected"] == "large" and item["predicted"] == "small"
        for item in predictions
    )
    false_upgrades = sum(
        item["expected"] == "small" and item["predicted"] == "large"
        for item in predictions
    )
    estimated_cost = sum(1 if item["predicted"] == "small" else 10 for item in predictions)
    return EvaluationMetrics(
        threshold=threshold,
        total=total,
        correct=correct,
        accuracy=correct / total,
        small_accuracy=sum(bool(item["correct"]) for item in small_items) / len(small_items),
        large_accuracy=sum(bool(item["correct"]) for item in large_items) / len(large_items),
        false_downgrades=false_downgrades,
        false_upgrades=false_upgrades,
        estimated_cost_units=estimated_cost,
        mean_policy_latency_ms=sum(latencies) / len(latencies),
        predictions=tuple(predictions),
    )


def evaluate_thresholds(
    cases: Sequence[EvaluationCase],
    thresholds: Iterable[float],
) -> list[EvaluationMetrics]:
    return [evaluate_cases(cases, threshold=value) for value in thresholds]


def recommended_result(results: Sequence[EvaluationMetrics]) -> EvaluationMetrics:
    if not results:
        raise ValueError("at least one evaluation result is required")
    return min(
        results,
        key=lambda item: (
            item.false_downgrades * 3 + item.false_upgrades,
            -item.accuracy,
            item.estimated_cost_units,
        ),
    )


def _serializable(metrics: EvaluationMetrics) -> dict[str, object]:
    data = asdict(metrics)
    data["predictions"] = list(metrics.predictions)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES_FILE))
    parser.add_argument("--threshold", type=float, action="append", dest="thresholds")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--output", help="Optionally write the report to a JSON file")
    args = parser.parse_args()

    thresholds = args.thresholds or [0.35, 0.40, 0.45, 0.50, 0.55]
    cases = load_cases(args.cases)
    results = evaluate_thresholds(cases, thresholds)
    recommendation = recommended_result(results)
    report = {
        "mode": "offline-lexical-proxy",
        "warning": "Proxy metrics are for deterministic regression trends, not production semantic quality.",
        "case_count": len(cases),
        "results": [_serializable(item) for item in results],
        "proxy_recommended_threshold": recommendation.threshold,
    }

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Offline lexical proxy evaluation (not production semantic quality)")
        print("threshold  accuracy  small_acc  large_acc  false_down  false_up  cost")
        for item in results:
            print(
                f"{item.threshold:>9.2f}  {item.accuracy:>8.1%}  "
                f"{item.small_accuracy:>9.1%}  {item.large_accuracy:>9.1%}  "
                f"{item.false_downgrades:>10}  {item.false_upgrades:>8}  "
                f"{item.estimated_cost_units:>4}",
            )
        print(f"Proxy recommendation: {recommendation.threshold:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
