import json
import os
import tempfile
from unittest.mock import patch

import pytest

from router import Claude_Router
from tests.conftest import make_router


class TestCosineSimilarity:
    def test_identical(self, router):
        vec = [1.0, 2.0, 3.0]
        assert router._cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal(self, router):
        assert router._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self, router):
        assert router._cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty(self, router):
        assert router._cosine_similarity([], [1.0, 2.0]) == 0.0
        assert router._cosine_similarity([1.0, 2.0], []) == 0.0


class TestRouteDecision:
    def test_force_large(self, router):
        assert router.route("hello", force_large=True) == "large"

    def test_empty_query(self, router):
        assert router.route("") == "large"
        assert router.route("   ") == "large"

    def test_mistake_intercepted(self):
        """触发错题本时应返回 large"""
        vec = [1.0, 1.0, 1.0]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"query": "失败", "vector": vec}, f)
            mistake_file = f.name

        try:
            r = make_router(mistake_file=mistake_file)
            r.mistake_book = [{"query": "失败", "vector": vec}]
            with patch.object(r, "_get_embedding", return_value=vec):
                assert r.route("失败") == "large"
        finally:
            os.unlink(mistake_file)

    def test_below_threshold_goes_large(self):
        """得分低于 threshold 应返回 large"""
        r = make_router(threshold=0.5)
        # 注入和 query 向量相似度 ≈ 0 的 small 种子
        r.route_embeddings = {"small": [[1.0, 0.0, 0.0]], "large": []}
        with patch.object(r, "_get_embedding", return_value=[0.0, 1.0, 0.0]):
            assert r.route("无关内容") == "large"

    def test_token_penalty_raises_bar(self):
        """长上下文提升动态及格线"""
        r = make_router(threshold=0.4, safe_tokens=100, penalty_step=100)
        r.penalty_rate = 0.1
        # seed=[1,0,0], query≈[0.9,0.4,0] → cos≈0.914
        r.route_embeddings = {"small": [[1.0, 0.0, 0.0]], "large": []}

        with patch.object(r, "_get_embedding", return_value=[0.9, 0.4, 0.0]):
            # 短上下文，得分 0.914 > 0.4 → small
            assert r.route("hello", total_tokens=50) == "small"
            # 长上下文，动态及格线升至 0.99，得分 0.914 < 0.99 → large
            assert r.route("hello", total_tokens=10000) == "large"


class TestMistakeRecording:
    def test_record_mistake_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mistake_file = os.path.join(tmpdir, "mistakes.json")
            r = make_router(mistake_file=mistake_file)
            r.mistake_book = []

            with patch.object(r, "_get_embedding", return_value=[1.0, 0.0, 0.0]):
                r.record_mistake("翻车了")

            assert os.path.exists(mistake_file)
            with open(mistake_file, encoding="utf-8") as f:
                record = json.loads(f.readline().strip())
            assert record["query"] == "翻车了"
            assert "vector" in record

    def test_mistake_book_max_limit(self):
        """错题本超过 max_mistakes 应淘汰最老条目"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mistake_file = os.path.join(tmpdir, "mistakes.json")
            r = make_router(mistake_file=mistake_file, max_mistakes=3)
            r.mistake_book = []

            with patch.object(r, "_get_embedding", return_value=[1.0, 0.0, 0.0]):
                r.record_mistake("q1")
                r.record_mistake("q2")
                r.record_mistake("q3")
                r.record_mistake("q4")

            assert len(r.mistake_book) == 3
            assert r.mistake_book[0]["query"] == "q2"
            assert r.mistake_book[-1]["query"] == "q4"
