import threading
import time

from embedding_service import EmbeddingProvider, normalize_embedding_text
from tests.conftest import make_router


def make_provider(fetcher, cache_size=2):
    return EmbeddingProvider(
        model_name="test-model",
        api_url="http://unused.invalid",
        cache_size=cache_size,
        fetcher=fetcher,
    )


def test_normalized_queries_share_one_cached_embedding():
    calls = []

    def fetcher(text):
        calls.append(text)
        return [1.0, 2.0]

    provider = make_provider(fetcher)

    first = provider.get("  hello\tworld ")
    second = provider.get("hello world")

    assert first == second == [1.0, 2.0]
    assert calls == ["hello world"]
    assert provider.stats().to_dict() == {
        "hits": 1,
        "misses": 1,
        "requests": 1,
        "failures": 0,
        "size": 1,
        "capacity": 2,
        "hit_rate": 0.5,
    }


def test_lru_cache_evicts_oldest_entry():
    calls = []

    def fetcher(text):
        calls.append(text)
        return [float(len(calls))]

    provider = make_provider(fetcher, cache_size=2)
    provider.get("one")
    provider.get("two")
    provider.get("one")
    provider.get("three")
    provider.get("two")

    assert calls == ["one", "two", "three", "two"]
    assert provider.stats().size == 2


def test_failed_embedding_is_not_cached_and_can_retry():
    attempts = 0

    def fetcher(text):
        nonlocal attempts
        attempts += 1
        return [] if attempts == 1 else [0.5]

    provider = make_provider(fetcher)

    assert provider.get("retry") == []
    assert provider.get("retry") == [0.5]
    assert attempts == 2
    assert provider.stats().failures == 1


def test_concurrent_identical_queries_coalesce_to_one_request():
    fetch_started = threading.Event()
    allow_finish = threading.Event()
    calls = 0
    results = []

    def fetcher(text):
        nonlocal calls
        calls += 1
        fetch_started.set()
        assert allow_finish.wait(timeout=2)
        return [0.25, 0.75]

    provider = make_provider(fetcher)
    first = threading.Thread(target=lambda: results.append(provider.get("same")))
    second = threading.Thread(target=lambda: results.append(provider.get("same")))
    first.start()
    assert fetch_started.wait(timeout=2)
    second.start()
    time.sleep(0.02)
    allow_finish.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert calls == 1
    assert results == [[0.25, 0.75], [0.25, 0.75]]
    assert provider.stats().hits == 1


def test_router_reuses_embedding_for_repeated_query():
    calls = 0

    def fetcher(text):
        nonlocal calls
        calls += 1
        return [0.5] * 768

    provider = make_provider(fetcher, cache_size=8)
    router = make_router(embedding_provider=provider)

    assert router.route("repeat query") == "small"
    assert router.route("  repeat   query ") == "small"
    assert calls == 1
    assert router.embedding_cache_stats()["hits"] == 1


def test_normalization_preserves_case_but_normalizes_unicode_and_space():
    assert normalize_embedding_text("Ａ  B\nC") == "A B C"
    assert normalize_embedding_text("A") != normalize_embedding_text("a")
