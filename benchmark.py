"""性能基准测试脚本 —— 输出表格并导出 JSON。

运行方式：
  python benchmark.py           # 仅本地
  python benchmark.py --api     # 含 API 调用
"""

import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from unittest.mock import patch

from tests.conftest import make_router


@contextmanager
def quiet():
    """临时静音标准输出"""
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


def format_time(s: float) -> str:
    if s < 1e-6:
        return f"{s * 1e9:.1f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    return f"{s:.2f} 秒"


results = []


def record(name: str, value, unit: str = "", detail: str = ""):
    results.append({"name": name, "value": value, "unit": unit, "detail": detail})
    if detail:
        print(f"  {name:40s} {format_time(value):>10s}   ({detail})")
    else:
        print(f"  {name:40s} {format_time(value):>10s}")


def main():
    print("=" * 65)
    print("  性能基准测试")
    print("=" * 65)

    # ── 1. 路由器初始化 ──
    print("\n── 1. 路由器初始化 ──")
    with quiet():
        t0 = time.perf_counter()
        r = make_router()
        t1 = time.perf_counter()
    total_seeds = sum(len(v) for v in r.route_embeddings.values())
    record("路由初始化总耗时", t1 - t0, "秒", f"{total_seeds} 条种子向量")

    # ── 2. 余弦相似度计算 ──
    print("\n── 2. 余弦相似度 ──")
    for dim in [128, 256, 512, 768]:
        a = [i * 0.01 for i in range(dim)]
        b = [i * 0.02 for i in range(dim)]
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            r._cosine_similarity(a, b)
        t1 = time.perf_counter()
        record(f"余弦 {dim}d (单次)", (t1 - t0) / N, "秒", f"{N} 次平均")

    # 3. 批量匹配
    query_vec = [0.5] * 768
    seed_vecs = [[i * 0.01 + 0.1 for i in range(768)] for _ in range(100)]
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        best = 0.0
        for sv in seed_vecs:
            s = r._cosine_similarity(query_vec, sv)
            if s > best:
                best = s
    t1 = time.perf_counter()
    record("批量匹配 1×100 (单次)", (t1 - t0) / N, "秒", f"{N} 次平均")

    # ── 3. 路由决策 ──
    print("\n── 3. 路由决策耗时 ──")
    queries = [
        ("简单", "列出文件"),
        ("简单", "当前时间"),
        ("代码", "写一个 Python 函数"),
        ("架构", "设计微服务架构"),
    ]
    for cat, q in queries:
        with patch.object(r, "_get_embedding", return_value=[0.5] * 768):
            N = 500
            with quiet():
                t0 = time.perf_counter()
                for _ in range(N):
                    r.route(q, total_tokens=500)
                t1 = time.perf_counter()
            record(f"路由决策 [{cat}]", (t1 - t0) / N, "秒", f'"{q}"')

    # ── 4. 错题本规模影响 ──
    print("\n── 4. 错题本规模对决策速度的影响 ──")
    for n in [0, 10, 50, 100]:
        r.mistake_book = [
            {"query": f"err{i}", "vector": [float(i) / 100] * 768}
            for i in range(n)
        ]
        with patch.object(r, "_get_embedding", return_value=[0.5] * 768):
            N = 200
            with quiet():
                t0 = time.perf_counter()
                for _ in range(N):
                    r.route("测试", total_tokens=500)
                t1 = time.perf_counter()
            record(f"错题本 {n:>3} 条 (单次)", (t1 - t0) / N, "秒", f"{N} 次平均")

    # ── 5. API 延迟（需要 --api） ──
    if "--api" in sys.argv:
        print("\n── 5. API 响应时间 ──")
        from anthropic import Anthropic
        from dotenv import load_dotenv
        load_dotenv(override=True)
        if os.getenv("ANTHROPIC_BASE_URL"):
            os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        client = Anthropic(
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            timeout=120,
        )

        msg = [{"role": "user", "content": "你好"}]
        API_RUNS = 3

        # small
        times = []
        for _ in range(API_RUNS):
            t0 = time.perf_counter()
            client.messages.create(model="small", messages=msg, max_tokens=10)
            times.append(time.perf_counter() - t0)
        record("small (deepseek-chat)", statistics.median(times), "秒",
               f"{API_RUNS} 次中位数: {', '.join(format_time(t) for t in times)}")

        # large
        times = []
        for _ in range(API_RUNS):
            t0 = time.perf_counter()
            client.messages.create(model="large", messages=msg, max_tokens=10)
            times.append(time.perf_counter() - t0)
        record("large (deepseek-v4-pro)", statistics.median(times), "秒",
               f"{API_RUNS} 次中位数: {', '.join(format_time(t) for t in times)}")

        # 路由开销 vs API 开销
        t0 = time.perf_counter()
        decision = r.route("帮我分析这段代码")
        t1 = time.perf_counter()
        route_cost = t1 - t0
        record("路由决策开销", route_cost, "秒", f"→ {decision}")
    else:
        print("\n── 5. API 延迟 ──")
        print("  跳过（加 --api 参数启用）")

    # ── 导出 JSON ──
    output_file = "benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 结果已导出到 {output_file}")


if __name__ == "__main__":
    main()
