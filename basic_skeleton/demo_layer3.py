#!/usr/bin/env python3
"""
演示脚本：展示Layer 3智能路由系统的完整工作流程
- 语义复杂度分析
- 多因子加权路由决策
- 路由反馈记录
- 动态学习和准确度统计
"""

import sys
import json

# 确保可以导入本地模块
sys.path.insert(0, '.')

from runtime.memory import FailureMemory
from runtime.trace import Trace
from runtime.router import Router
from runtime.semantic_router import SemanticRouter


def demo_semantic_complexity():
    """演示1：语义复杂度分析"""
    print("\n" + "="*60)
    print("DEMO 1: Semantic Complexity Analysis")
    print("="*60 + "\n")
    
    sr = SemanticRouter()
    
    test_queries = [
        ("What is Python?", "Easy query"),
        ("How to print hello world?", "Easy query"),
        ("Optimize garbage collection in Python", "Hard query"),
        ("Design distributed consensus algorithm", "Hard query"),
    ]
    
    for query, label in test_queries:
        score = sr.predict_complexity(query)
        category = "HARD" if score > 0.6 else "EASY"
        print(f"[{category}] {query[:50]}")
        print(f"    Score: {score:.3f}")
        
        # Show similar seeds for debugging
        similarities = sr.debug_similarity(query)
        if similarities["easy_matches"]:
            print(f"    Similar easy: {similarities['easy_matches'][0][0][:40]}")
        if similarities["hard_matches"]:
            print(f"    Similar hard: {similarities['hard_matches'][0][0][:40]}")
        print()


def demo_multifactor_routing():
    """演示2：多因子加权路由"""
    print("="*60)
    print("DEMO 2: Multi-Factor Weighted Routing")
    print("="*60 + "\n")
    
    memory = FailureMemory()
    trace = Trace()
    router = Router(memory=memory, trace=trace, use_semantic_routing=True)
    
    queries = [
        "Python hello world",
        "Distributed system consensus algorithm",
        "List vs tuple explanation",
        "Garbage collection optimization strategies",
    ]
    
    print("Query Routing Decisions:")
    for query in queries:
        route = router.route(query)
        target = route["target_model"]
        reason = route["reason"]
        
        print(f"\nQuery: {query}")
        print(f"  Target: {target.upper()}")
        print(f"  Decision Score: {route.get('decision_score', 'N/A'):.3f}")
        print(f"  Semantic Score: {route.get('semantic_score', 'N/A'):.3f}")
        print(f"  Reason: {reason}")


def demo_routing_feedback():
    """演示3：路由反馈记录和统计"""
    print("\n" + "="*60)
    print("DEMO 3: Routing Feedback Recording")
    print("="*60 + "\n")
    
    memory = FailureMemory()
    
    # 模拟一些路由决策和结果
    feedback_scenarios = [
        {
            "query": "How to print hello world?",
            "complexity": 0.2,
            "model": "small",
            "quality": "GOOD",
            "explanation": "Easy query, small model succeeded (cost-effective)"
        },
        {
            "query": "Garbage collection optimization",
            "complexity": 0.8,
            "model": "large",
            "quality": "GOOD",
            "explanation": "Hard query, large model delivered quality"
        },
        {
            "query": "Python list methods",
            "complexity": 0.3,
            "model": "small",
            "quality": "BAD",
            "explanation": "Small model failed, needed escalation"
        },
    ]
    
    print("Recording routing feedback:")
    for scenario in feedback_scenarios:
        memory.save_routing_feedback(
            query=scenario["query"],
            predicted_complexity=scenario["complexity"],
            selected_model=scenario["model"],
            final_quality=scenario["quality"],
        )
        
        status = "CORRECT" if (
            (scenario["model"] == "small" and scenario["quality"] == "GOOD") or
            (scenario["model"] == "large" and scenario["quality"] == "GOOD")
        ) else "INCORRECT"
        
        print(f"\n[{status}] {scenario['query'][:40]}")
        print(f"      Model: {scenario['model']}, Quality: {scenario['quality']}")
        print(f"      {scenario['explanation']}")
    
    # 显示统计
    print("\n" + "-"*60)
    print("Routing Statistics Summary:")
    stats = memory.get_routing_stats()
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Overall accuracy: {stats['accuracy']:.1%}")
    print(f"  Recent trend: {stats['recent_trend']}")
    print(f"\n  By model:")
    for model, model_stats in stats["by_model"].items():
        print(f"    {model}: {model_stats['accuracy']:.1%} ({model_stats['count']} decisions)")


def demo_layer3_learning():
    """演示4：Layer 3 学习系统概述"""
    print("\n" + "="*60)
    print("DEMO 4: Layer 3 Self-Learning System Overview")
    print("="*60 + "\n")
    
    print("""
Layer 3 Architecture (Intelligence Feedback Loop):
    
1. ROUTING DECISIONS
   ├─ Query semantic complexity scoring
   ├─ Task-specific failure rate checking
   ├─ Multi-factor weighted decision scoring
   └─ Probabilistic routing (small vs large)

2. EXECUTION & EVALUATION
   ├─ Execute selected model
   ├─ Judge answer quality
   └─ Record outcome with routing score

3. FEEDBACK RECORDING
   ├─ save_routing_feedback() captures:
   │  ├─ predicted_complexity (semantic score)
   │  ├─ selected_model (small/large)
   │  ├─ final_quality (GOOD/BAD)
   │  └─ was_correct (accurate routing decision?)
   └─ Persistent storage in failure_memory.json

4. ANALYTICS & LEARNING
   ├─ get_routing_stats() computes:
   │  ├─ Overall accuracy
   │  ├─ Per-model accuracy
   │  ├─ Recent trend (improving/stable)
   │  └─ Decision volume by model
   └─ Ready for dynamic threshold adjustment

5. CONTINUOUS IMPROVEMENT
   ├─ Track routing decision quality over time
   ├─ Identify model-specific accuracy patterns
   ├─ Detect improving vs degrading trends
   └─ Foundation for adaptive thresholds (future)

Key Innovation: Unlike traditional routing (rules-based),
this learns what decisions were actually CORRECT,
enabling true "semantic understanding" that improves with use.
    """)
    
    print("\nFuture Enhancement: Dynamic Threshold Adjustment")
    print("  If small model accuracy < 60%, increase threshold")
    print("  If large model accuracy > 90%, decrease threshold")
    print("  This creates a closed-loop self-improving system")


if __name__ == "__main__":
    try:
        demo_semantic_complexity()
        demo_multifactor_routing()
        demo_routing_feedback()
        demo_layer3_learning()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE: Layer 3 System Fully Operational")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
