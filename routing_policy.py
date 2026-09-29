"""Pure routing policy separated from embedding and persistence concerns."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


Vector = Sequence[float]


def cosine_similarity(vec1: Vector, vec2: Vector) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    best_route: str
    highest_score: float
    dynamic_threshold: float
    route_scores: dict[str, float]
    mistake_score: float = 0.0
    intercepted_by_mistake: bool = False
    intercepted_by_context: bool = False


class RoutingPolicy:
    def __init__(
        self,
        *,
        threshold: float,
        mistake_threshold: float,
        safe_tokens: int,
        penalty_step: int,
        penalty_rate: float = 0.05,
    ):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not 0 <= mistake_threshold <= 1:
            raise ValueError("mistake_threshold must be between 0 and 1")
        if safe_tokens < 0:
            raise ValueError("safe_tokens must be non-negative")
        if penalty_step <= 0:
            raise ValueError("penalty_step must be positive")
        if penalty_rate < 0:
            raise ValueError("penalty_rate must be non-negative")
        self.threshold = threshold
        self.mistake_threshold = mistake_threshold
        self.safe_tokens = safe_tokens
        self.penalty_step = penalty_step
        self.penalty_rate = penalty_rate

    def threshold_for(self, total_tokens: int) -> float:
        dynamic_threshold = self.threshold
        if total_tokens > self.safe_tokens:
            extra_steps = (total_tokens - self.safe_tokens) // self.penalty_step
            dynamic_threshold = min(
                0.99,
                self.threshold + extra_steps * self.penalty_rate,
            )
        return dynamic_threshold

    def decide(
        self,
        query_vector: Vector,
        *,
        route_embeddings: Mapping[str, Sequence[Vector]],
        mistake_vectors: Sequence[Vector] = (),
        total_tokens: int = 0,
    ) -> RoutingDecision:
        route_scores = {name: 0.0 for name in route_embeddings}
        route_scores.setdefault("small", 0.0)
        route_scores.setdefault("large", 0.0)

        mistake_score = 0.0
        for vector in mistake_vectors:
            score = cosine_similarity(query_vector, vector)
            mistake_score = max(mistake_score, score)
            if score >= self.mistake_threshold:
                return RoutingDecision(
                    route="large",
                    best_route="large",
                    highest_score=0.0,
                    dynamic_threshold=self.threshold_for(total_tokens),
                    route_scores=route_scores,
                    mistake_score=mistake_score,
                    intercepted_by_mistake=True,
                )

        best_route = "large"
        highest_score = 0.0
        for route_name, embeddings in route_embeddings.items():
            for embedding in embeddings:
                score = cosine_similarity(query_vector, embedding)
                route_scores[route_name] = max(route_scores.get(route_name, 0.0), score)
                if score > highest_score:
                    highest_score = score
                    best_route = route_name

        dynamic_threshold = self.threshold_for(total_tokens)
        if best_route == "large" or highest_score < self.threshold:
            route = "large"
        elif highest_score >= dynamic_threshold:
            route = "small"
        else:
            route = "large"

        return RoutingDecision(
            route=route,
            best_route=best_route,
            highest_score=highest_score,
            dynamic_threshold=dynamic_threshold,
            route_scores=route_scores,
            mistake_score=mistake_score,
            intercepted_by_context=(
                best_route == "small"
                and highest_score >= self.threshold
                and highest_score < dynamic_threshold
            ),
        )
