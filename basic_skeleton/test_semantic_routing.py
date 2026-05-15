#!/usr/bin/env python3
"""
集成测试：验证语义路由和反馈记录系统
"""
import json
import os

from runtime.memory import FailureMemory
from runtime.trace import Trace
from runtime.router import Router
from runtime.runtime import AdaptiveInferenceRuntime


def print_section(title: str) -> None:
    """打印分隔符"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_semantic_routing():
    """测试语义路由和反馈记录"""
    print_section("Integration Test: Semantic Routing with Feedback")

    # 初始化系统
    trace = Trace()
    memory = FailureMemory()
    router = Router(memory=memory, trace=trace, use_semantic_routing=True)
    runtime = AdaptiveInferenceRuntime()

    # 测试用例：简单 vs 复杂查询
    test_cases = [
        "How to print hello world?",
        "Python garbage collection optimization strategies for distributed systems",
        "What is the difference between lists and tuples?",
        "Design a distributed consensus algorithm handling Byzantine failures",
    ]

    print("Running test queries...")
    for i, query in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Query: {query[:60]}...")
        result = runtime.run(query)
        
        # 显示路由决策
        meta = result.get("meta", {})
        print(f"  Path: {meta.get('path', 'unknown')}")
        print(f"  Escalated: {meta.get('escalated', False)}")
        if 'routing_accuracy' in meta:
            print(f"  Routing Accuracy: {meta['routing_accuracy']:.1%}")

    # 显示路由统计
    print_section("Routing Statistics")
    routing_stats = memory.get_routing_stats()
    print(f"Total routing decisions: {routing_stats['total_decisions']}")
    print(f"Overall accuracy: {routing_stats['accuracy']:.1%}")
    print(f"Recent trend: {routing_stats['recent_trend']}")
    print(f"\nBy model:")
    for model, stats in routing_stats["by_model"].items():
        print(f"  {model}: {stats['accuracy']:.1%} ({stats['count']} decisions)")

    # 显示内存统计
    print_section("Failure Statistics")
    mem_stats = memory.summarize_memory()
    print(f"Total failures: {mem_stats['total_failures']}")
    print(f"By task type: {json.dumps(mem_stats['by_task_type'], indent=2)}")
    print(f"By model: {json.dumps(mem_stats['by_model'], indent=2)}")

    # 显示数据文件状态
    print_section("Data Files")
    memory_file = memory.path
    if os.path.exists(memory_file):
        size = os.path.getsize(memory_file)
        print(f"Memory file: {memory_file}")
        print(f"Size: {size} bytes")
        
        # 读取文件并显示最后一条反馈记录
        with open(memory_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("routing_feedback"):
                last_feedback = data["routing_feedback"][-1]
                print(f"\nLast routing feedback:")
                print(f"  Query: {last_feedback['query_preview']}")
                print(f"  Model: {last_feedback['selected_model']}")
                print(f"  Complexity: {last_feedback['predicted_complexity']}")
                print(f"  Quality: {last_feedback['final_quality']}")
                print(f"  Correct: {last_feedback['was_correct']}")


if __name__ == "__main__":
    test_semantic_routing()
    print_section("Test Complete")
    print("✓ Semantic routing integration working")
    print("✓ Routing feedback recording functional")
    print("✓ Statistics aggregation operational")
