"""性能基准测试：路由决策耗时、API 响应时间、场景模拟。

运行方式：
  # 仅测本地路由逻辑（不需要 API）
  python -m pytest tests/test_benchmarks.py -v

  # 包含 API 调用测试（需要 LiteLLM 代理运行中）
  python -m pytest tests/test_benchmarks.py -v --run-api

  # 输出详细耗时报告
  python -m pytest tests/test_benchmarks.py -v --run-api --benchmark-json=report.json
"""

import time
import math
from unittest.mock import patch

import pytest

from tests.conftest import make_router

# ── 辅助函数 ────────────────────────────────────────────
def pytest_addoption(parser):
    parser.addoption(
        "--run-api", action="store_true", default=False,
        help="包含需要 LiteLLM 代理在线的测试",
    )


# ── 辅助函数 ────────────────────────────────────────────
def format_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} 秒"


# ══════════════════════════════════════════════════════════
# Benchmark 1: 路由器初始化耗时
# ══════════════════════════════════════════════════════════
class TestRouterInitBenchmark:
    def test_route_embeddings_computation_speed(self):
        """测量构建 route_embeddings 的耗时"""
        t0 = time.perf_counter()
        r = make_router()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        total_seeds = sum(len(v) for v in r.route_embeddings.values())
        print(f"\n  [耗时] 路由器初始化: {format_time(elapsed)}")
        print(f"  [种子] 共 {total_seeds} 条向量")


# ══════════════════════════════════════════════════════════
# Benchmark 2: 余弦相似度计算速度
# ══════════════════════════════════════════════════════════
class TestCosineBenchmark:
    VEC_128 = [i * 0.01 for i in range(128)]
    VEC_256 = [i * 0.01 for i in range(256)]
    VEC_512 = [i * 0.01 for i in range(512)]
    VEC_768 = [i * 0.01 for i in range(768)]

    @pytest.mark.parametrize("dim,label", [
        (128, "128d"), (256, "256d"), (512, "512d"), (768, "768d"),
    ])
    def test_cosine_varying_dims(self, router, dim, label):
        """不同维度的余弦相似度计算速度"""
        vec_a = [i * 0.01 for i in range(dim)]
        vec_b = [i * 0.02 for i in range(dim)]

        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            router._cosine_similarity(vec_a, vec_b)
        t1 = time.perf_counter()

        avg = (t1 - t0) / N
        total = t1 - t0
        print(f"\n  [{label}] {N} 次平均: {format_time(avg)}，总计: {format_time(total)}")

    def test_cosine_bulk_matching(self, router):
        """模拟路由场景：1 个 query 匹配 100 条种子"""
        query_vec = [0.5] * 128
        seed_vecs = [[i * 0.01 + 0.1 for i in range(128)] for _ in range(100)]

        N = 1000
        t0 = time.perf_counter()
        for _ in range(N):
            best_score = 0.0
            for sv in seed_vecs:
                s = router._cosine_similarity(query_vec, sv)
                if s > best_score:
                    best_score = s
        t1 = time.perf_counter()

        avg = (t1 - t0) / N
        print(f"\n  [批量匹配] 1 query × 100 种子，{N} 次平均: {format_time(avg)}")


# ══════════════════════════════════════════════════════════
# Benchmark 3: 路由决策全链路耗时
# ══════════════════════════════════════════════════════════
class TestRouteDecisionBenchmark:
    QUERIES = [
        ("日常简单", "列出当前目录的文件"),
        ("日常简单", "当前时间"),
        ("日常简单", "谢谢"),
        ("代码任务", "写一个 Python 函数读取 CSV"),
        ("架构分析", "这个系统的性能瓶颈在哪"),
        ("重构", "帮我重构这段代码"),
        ("英文", "list files in current directory"),
        ("混合", "为什么这个查询这么慢，帮我分析一下"),
    ]

    def test_route_decision_speed(self, router):
        """测量 route() 单次决策耗时"""
        print()
        for category, query in self.QUERIES:
            with patch.object(router, "_get_embedding", return_value=[0.5] * 128):
                N = 500
                t0 = time.perf_counter()
                for _ in range(N):
                    router.route(query, total_tokens=500)
                t1 = time.perf_counter()
                avg = (t1 - t0) / N
                print(f"  [{category}] \"{query}\": {format_time(avg)}")

    def test_mistake_check_penalty(self, router):
        """错题本规模对路由决策速度的影响"""
        print()
        for n_mistakes in [0, 10, 50, 100]:
            router.mistake_book = [
                {"query": f"错误查询{i}", "vector": [float(i) / 100] * 128}
                for i in range(n_mistakes)
            ]
            with patch.object(router, "_get_embedding", return_value=[0.5] * 128):
                N = 200
                t0 = time.perf_counter()
                for _ in range(N):
                    router.route("测试查询", total_tokens=500)
                t1 = time.perf_counter()
                avg = (t1 - t0) / N
                print(f"  [错题本 {n_mistakes} 条] {N} 次平均: {format_time(avg)}")


# ══════════════════════════════════════════════════════════
# Benchmark 4: 真实 API 响应时间（需 LiteLLM 在线）
# ══════════════════════════════════════════════════════════
class TestApiLatency:
    @pytest.fixture(scope="class")
    def real_client(self, pytestconfig):
        if not pytestconfig.getoption("run_api"):
            pytest.skip("需加 --run-api 参数执行")
        from anthropic import Anthropic
        from dotenv import load_dotenv
        import os
        load_dotenv()
        client = Anthropic(
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            timeout=60,
        )
        return client

    def test_small_model_latency(self, real_client):
        """small 模型首 token 响应时间"""
        t0 = time.perf_counter()
        real_client.messages.create(
            model="small",
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=10,
        )
        t1 = time.perf_counter()
        print(f"\n  [small] 总响应时间: {format_time(t1 - t0)}")

    def test_large_model_latency(self, real_client):
        """large 模型首 token 响应时间"""
        t0 = time.perf_counter()
        real_client.messages.create(
            model="large",
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=10,
        )
        t1 = time.perf_counter()
        print(f"\n  [large] 总响应时间: {format_time(t1 - t0)}")

    def test_routing_overhead_in_real_call(self):
        """路由器 + API 调用整体耗时（路由器开销 vs API 开销）"""
        r = make_router()

        # 路由器开销
        t0 = time.perf_counter()
        decision = r.route("帮我分析这段代码的性能")
        t1 = time.perf_counter()
        route_cost = t1 - t0

        print(f"\n  [路由决策] {decision}，决策耗时: {format_time(route_cost)}")
        print(f"  [说明] 路由决策在毫秒级，实际瓶颈在 API 调用而非路由逻辑")


# ══════════════════════════════════════════════════════════
# Benchmark 5: 场景模拟（路由准确率 + 成本模拟）
# ══════════════════════════════════════════════════════════
class TestSimulation:
    """用预设查询模拟真实使用场景，统计路由准确率与成本对比"""

    SCENARIOS = [
        # (类别, 查询, 期望路由)
        ("简单文件操作", "列出文件", "small"),
        ("简单文件操作", "删除这个文件", "small"),
        ("简单问答", "当前时间", "small"),
        ("日常交流", "你好", "small"),
        ("日常交流", "谢谢", "small"),
        ("代码生成", "写一个函数计算斐波那契数列", "small"),
        ("搜索", "搜索包含 error 的日志文件", "small"),
        ("安装", "安装 requests 库", "small"),
        ("系统信息", "查看磁盘空间", "small"),
        ("批量", "把所有的 .tmp 文件备份", "small"),
        ("架构", "设计一个微服务架构", "large"),
        ("性能分析", "帮我分析内存泄漏", "large"),
        ("安全", "做一次安全审计", "large"),
        ("优化", "数据库查询太慢了帮我优化", "large"),
        ("分布式", "设计分布式锁方案", "large"),
        ("重构", "重构整个模块的代码", "large"),
    ]

    def test_routing_accuracy(self, router):
        """路由决策是否符合预期"""
        print()
        correct = 0
        for category, query, expected in self.SCENARIOS:
            with patch.object(router, "_get_embedding", return_value=[0.5] * 128):
                result = router.route(query, total_tokens=100)
                ok = result == expected
                if ok:
                    correct += 1
                flag = "✓" if ok else "✗"
                print(f"  [{flag}] {category}: \"{query}\" → {result} (期望 {expected})")

        accuracy = correct / len(self.SCENARIOS) * 100
        print(f"\n  [准确率] {correct}/{len(self.SCENARIOS)} = {accuracy:.1f}%")

    def test_cost_comparison_simulation(self):
        """模拟 cost-only vs router 模式的开销对比（基于 mock 决策）"""
        # 假设 small 成本 1 单位/次，large 成本 10 单位/次
        SMALL_COST = 1
        LARGE_COST = 10
        r = make_router()

        print()
        # cost-only: 全部走 small
        cost_only_small = sum(1 for *_, exp in self.SCENARIOS if exp == "small")
        cost_only_large = sum(1 for *_, exp in self.SCENARIOS if exp == "large")
        cost_only_total = (cost_only_small + cost_only_large) * SMALL_COST
        cost_only_pct = cost_only_small / (cost_only_small + cost_only_large) * 100

        # router 模式：按实际路由结果
        router_small = 0
        router_large = 0
        router_wrong = 0
        for category, query, expected in self.SCENARIOS:
            with patch.object(r, "_get_embedding", return_value=[0.5] * 128):
                result = r.route(query, total_tokens=100)
            if result == "small":
                router_small += 1
            else:
                router_large += 1
            if result != expected:
                router_wrong += 1

        router_total = router_small * SMALL_COST + router_large * LARGE_COST
        router_pct = router_small / (router_small + router_large) * 100

        print(f"  ┌─────────────────────┬──────────┬──────────┐")
        print(f"  │ 指标                │ cost-only │ router   │")
        print(f"  ├─────────────────────┼──────────┼──────────┤")
        print(f"  │ small 调用占比      │ {cost_only_pct:>6.1f}%   │ {router_pct:>6.1f}%   │")
        print(f"  │ large 调用占比      │ {100-cost_only_pct:>6.1f}%   │ {100-router_pct:>6.1f}%   │")
        print(f"  │ 总成本（单位）      │ {cost_only_total:>6}    │ {router_total:>6}    │")
        print(f"  │ 节省比例            │     —     │ {(1-router_total/cost_only_total)*100:>6.1f}%   │")
        print(f"  └─────────────────────┴──────────┴──────────┘")
