#!/usr/bin/env python3
"""
quick_demo.py - 快速演示脚本

展示集成框架的主要功能，无需交互式输入。
"""

import json
import time
from pathlib import Path

def print_section(title):
    """打印分段标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_agent_capabilities():
    """演示 Agent 能力"""
    print_section("📍 集成 Agent 框架演示")
    
    from agent import TodoManager, TOOLS, estimate_tokens
    
    # 1. Todo 管理
    print("1️⃣ Todo 管理系统")
    print("-" * 50)
    
    todo = TodoManager()
    items = [
        {"title": "分析项目结构", "status": "completed"},
        {"title": "集成 Agent 框架", "status": "completed"},
        {"title": "编写文档", "status": "in-progress"},
        {"title": "测试集成", "status": "not-started"},
    ]
    result = todo.update(items)
    print(result)
    print(f"有未完成任务: {todo.has_open_items()}")
    
    # 2. 工具系统
    print("\n2️⃣ 工具系统")
    print("-" * 50)
    print(f"已定义 {len(TOOLS)} 个工具:")
    for tool in TOOLS:
        print(f"  • {tool['name']:15} - {tool['description'][:50]}")
    
    # 3. 上下文管理
    print("\n3️⃣ 上下文管理（Token 估算）")
    print("-" * 50)
    
    test_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]
    
    tokens = estimate_tokens(test_messages)
    print(f"示例消息约 {tokens} tokens")
    print(f"压缩阈值: 100000 tokens")


def demo_inference_integration():
    """演示推理集成"""
    print_section("🧠 自适应推理集成演示")
    
    from agent import run_inference_query
    
    print("运行一次推理查询: 'What is 2+2?'")
    print("-" * 50)
    
    result = run_inference_query("What is 2+2?")
    
    print(f"📍 返回结构:")
    for key in result.keys():
        print(f"  • {key}")
    
    print(f"\n📍 答案预览 (前 150 字):")
    print(f"  {result['answer'][:150]}...")
    
    print(f"\n📍 路由元信息:")
    meta = result.get("meta", {})
    for key, value in list(meta.items())[:5]:
        print(f"  • {key}: {str(value)[:60]}")


def demo_failure_memory():
    """演示失败记忆"""
    print_section("💾 失败记忆与学习")
    
    failure_file = Path("data/failure_memory.json")
    
    if failure_file.exists():
        data = json.loads(failure_file.read_text())
        
        print("📍 失败记忆统计:")
        failures = data.get("failures", [])
        routing = data.get("routing_feedback", [])
        
        print(f"  • 总失败案例: {len(failures)}")
        print(f"  • 总路由反馈: {len(routing)}")
        
        if failures:
            print(f"\n📍 最近失败案例:")
            for f in failures[-3:]:
                print(f"  • [{f.get('task_type', '?')}] {f.get('query_preview', '')[:50]}")
                print(f"    失败原因: {f.get('reason', [])}")
        
        if routing:
            correct = sum(1 for r in routing if r.get("was_correct"))
            accuracy = (correct / len(routing) * 100) if routing else 0
            print(f"\n📍 路由准确率: {accuracy:.1f}% ({correct}/{len(routing)})")
    else:
        print("💡 尚无失败记忆文件，运行推理后会自动生成。")


def demo_architecture():
    """演示整体架构"""
    print_section("🏗️ 架构总览")
    
    architecture = """
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Modal Agent System                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Qwen 3.5 LLM    │
                    │  (Multi-turn)    │
                    └──────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
          ┌─────────┐   ┌──────────┐   ┌────────────┐
          │  Tools  │   │   Todo   │   │ Inference  │
          │ Dispatch│   │ Management  │ Integration│
          └─────────┘   └──────────┘   └────────────┘
                │             │              │
                └─────────────┼──────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌──────────────┐          ┌──────────────────┐
        │  Local File  │          │ Adaptive Routing │
        │     I/O      │          │   Subsystem      │
        │              │          │                  │
        │ • read_file  │          │ • Router         │
        │ • write_file │          │ • Evaluator      │
        │ • edit_file  │          │ • Escalation     │
        │ • bash       │          │ • Memory         │
        └──────────────┘          └──────────────────┘
                │                           │
                └───────────────┬───────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │ Local Persistence    │
                    │ (JSON-based Memory)  │
                    └──────────────────────┘
"""
    
    print(architecture)
    
    print("📍 工作流程:")
    print("""
1. 用户输入 → 本地 Qwen LLM (多轮对话)
2. Agent 可以使用工具:
   - 文件操作 (read/write/edit)
   - 命令执行 (bash)
   - 任务管理 (TodoWrite)
   - 智能推理 (inference 工具 → 自适应路由)
3. 推理结果返回给 Agent
4. Agent 继续对话或继续调用工具
5. 失败案例保存到本地记忆
    """)


def demo_usage():
    """演示使用方式"""
    print_section("🚀 使用方式")
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│ 模式 1: Agent 模式（推荐 - 完整功能）                         │
│                                                             │
│   $ python main.py --mode agent                             │
│                                                             │
│   特性: 多轮对话 + 工具 + 自适应推理                          │
│   需要: 本地 Ollama + 千问模型                               │
│   优势: 最灵活、功能最全面                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 模式 2: 推理模式（轻量 - 仅自适应推理）                       │
│                                                             │
│   单次:  $ python main.py --mode inference \\              │
│          --query "Your question"                          │
│                                                             │
│   交互:  $ python main.py --mode inference                 │
│                                                             │
│   特性: 语义路由 + 失败评估 + 升级处理                        │
│   需要: 无（可选 Ollama）                                    │
│   优势: 成本最低、速度快、本地运行                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 工具列表（在 Agent 模式中可用）                               │
│                                                             │
│   • bash            - 执行 Shell 命令                       │
│   • read_file       - 读取文件内容                           │
│   • write_file      - 创建/覆盖文件                          │
│   • edit_file       - 精确编辑文件（替换文本）               │
│   • TodoWrite       - 管理任务列表                           │
│   • inference       - 调用自适应推理                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
    """)


def main():
    """运行演示"""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║           Adaptive Inference Runtime + Agent Framework            ║")
    print("║                         快速演示                                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    try:
        demo_agent_capabilities()
        demo_inference_integration()
        demo_failure_memory()
        demo_architecture()
        demo_usage()
        
        print_section("✅ 演示完成")
        print("""
下一步:

1. 查看详细文档:
   $ cat AGENT_GUIDE.md

2. 运行集成测试:
   $ python test_integrated_agent.py

3. 启动 Agent:
   $ python main.py --mode agent

4. 或使用推理模式:
   $ python main.py --mode inference
        """)
        
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
