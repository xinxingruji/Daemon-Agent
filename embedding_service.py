"""Embedding retrieval with bounded, observable, thread-safe reuse."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import threading
import unicodedata
import urllib.request
from typing import Callable


EmbeddingFetcher = Callable[[str], list[float]]
DEFAULT_EMBEDDING_CACHE_SIZE = 256


def normalize_embedding_text(text: str) -> str:
    """Normalize equivalent whitespace/Unicode forms without changing case."""
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


@dataclass(frozen=True)
class EmbeddingCacheStats:
    hits: int
    misses: int
    requests: int
    failures: int
    size: int
    capacity: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "requests": self.requests,
            "failures": self.failures,
            "size": self.size,
            "capacity": self.capacity,
            "hit_rate": self.hit_rate,
        }


class EmbeddingProvider:
    """Fetch embeddings once per normalized query and retain a bounded LRU."""

    def __init__(
        self,
        *,
        model_name: str,
        api_url: str,
        cache_size: int = DEFAULT_EMBEDDING_CACHE_SIZE,
        timeout: float = 60,
        fetcher: EmbeddingFetcher | None = None,
    ):
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model_name = model_name
        self.api_url = api_url
        self.cache_size = cache_size
        self.timeout = timeout
        self._fetcher = fetcher or self._fetch_http
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._requests = 0
        self._failures = 0

    @staticmethod
    def _valid_vector(vector: object) -> bool:
        return (
            isinstance(vector, list)
            and bool(vector)
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in vector)
        )

    def _fetch_http(self, text: str) -> list[float]:
        payload = {"model": self.model_name, "prompt": text}
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        vector = data.get("embedding") if isinstance(data, dict) else None
        return vector if self._valid_vector(vector) else []

    def get(self, text: str) -> list[float]:
        key = normalize_embedding_text(text)
        if not key:
            return []

        owner = False
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._hits += 1
                return list(cached)
            event = self._inflight.get(key)
            if event is None:
                event = threading.Event()
                self._inflight[key] = event
                self._misses += 1
                owner = True

        if not owner:
            event.wait()
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return list(cached)
                return []

        vector: list[float] = []
        try:
            with self._lock:
                self._requests += 1
            candidate = self._fetcher(key)
            if self._valid_vector(candidate):
                vector = [float(value) for value in candidate]
            else:
                with self._lock:
                    self._failures += 1
        except Exception:
            with self._lock:
                self._failures += 1
        finally:
            with self._lock:
                if vector and self.cache_size:
                    self._cache[key] = tuple(vector)
                    self._cache.move_to_end(key)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
                self._inflight.pop(key, None)
                event.set()
        return vector

    def clear(self, *, reset_stats: bool = False) -> None:
        with self._lock:
            self._cache.clear()
            if reset_stats:
                self._hits = self._misses = self._requests = self._failures = 0

    def stats(self) -> EmbeddingCacheStats:
        with self._lock:
            return EmbeddingCacheStats(
                hits=self._hits,
                misses=self._misses,
                requests=self._requests,
                failures=self._failures,
                size=len(self._cache),
                capacity=self.cache_size,
            )
