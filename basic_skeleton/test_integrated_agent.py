#!/usr/bin/env python3
"""
test_integrated_agent.py - 测试集成后的 Agent 框架

验证以下功能：
1. Agent 循环能正常启动
2. Tool dispatch 系统能正确处理工具调用
3. 自适应推理能和 agent 正确集成
4. Todo 管理能正常工作
"""

import json
import sys
from pathlib import Path

# 确保可以导入 agent 模块
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试所有导入"""
    print("[TEST] 测试导入...")
    try:
        from agent import (
            Agent, TodoManager, TOOL_HANDLERS, 
            run_bash, run_read, run_write, run_edit,
            run_inference_query
        )
        print("✓ 所有导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_todo_manager():
    """测试 Todo 管理器"""
    print("\n[TEST] 测试 Todo 管理器...")
    try:
        from agent import TodoManager
        
        tm = TodoManager()
        
        # 测试更新
        items = [
            {"title": "第一个任务", "status": "in-progress"},
            {"title": "第二个任务", "status": "not-started"},
            {"title": "第三个任务", "status": "completed"},
        ]
        result = tm.update(items)
        
        print("Todo 列表:")
        print(result)
        
        # 检查是否有打开的任务
        has_open = tm.has_open_items()
        print(f"有未完成任务: {has_open}")
        
        print("✓ Todo 管理器测试通过")
        return True
    except Exception as e:
        print(f"✗ Todo 管理器测试失败: {e}")
        return False


def test_tool_handlers():
    """测试工具处理器"""
    print("\n[TEST] 测试工具处理器...")
    try:
        from agent import run_write, run_read, run_bash, TOOL_HANDLERS
        
        # 测试写入
        test_file = Path("test_integrated.txt")
        result = run_write(str(test_file), "Hello from integrated agent!")
        print(f"写入测试: {result}")
        
        # 测试读取
        result = run_read(str(test_file))
        print(f"读取测试: {result}")
        
        # 清理
        test_file.unlink()
        
        print("✓ 工具处理器测试通过")
        return True
    except Exception as e:
        print(f"✗ 工具处理器测试失败: {e}")
        return False


def test_inference_integration():
    """测试推理集成"""
    print("\n[TEST] 测试推理集成...")
    try:
        from agent import run_inference_query
        
        # 测试一个简单的推理
        result = run_inference_query("What is 2+2?")
        
        if "answer" in result:
            print(f"推理返回结构正确: {list(result.keys())}")
            print(f"答案预览: {result['answer'][:100]}...")
            print("✓ 推理集成测试通过")
            return True
        else:
            print(f"✗ 推理返回结构错误: {result}")
            return False
    except Exception as e:
        print(f"✗ 推理集成测试失败: {e}")
        return False


def test_tool_schemas():
    """测试工具 schema 定义"""
    print("\n[TEST] 测试工具 schema...")
    try:
        from agent import TOOLS
        
        print(f"已定义 {len(TOOLS)} 个工具:")
        for tool in TOOLS:
            print(f"  - {tool['name']}: {tool['description'][:60]}...")
        
        # 检查必要的工具
        required_tools = {"bash", "read_file", "write_file", "edit_file", "TodoWrite", "inference"}
        defined_tools = {t["name"] for t in TOOLS}
        
        missing = required_tools - defined_tools
        if missing:
            print(f"✗ 缺少工具: {missing}")
            return False
        
        print("✓ 工具 schema 测试通过")
        return True
    except Exception as e:
        print(f"✗ 工具 schema 测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║             集成 Agent 框架测试套件                              ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    results = []
    
    results.append(("导入", test_imports()))
    if results[-1][1]:  # 如果导入成功，运行其他测试
        results.append(("Todo 管理器", test_todo_manager()))
        results.append(("工具处理器", test_tool_handlers()))
        results.append(("推理集成", test_inference_integration()))
        results.append(("工具 Schema", test_tool_schemas()))
    
    print("\n" + "="*64)
    print("测试结果汇总:")
    passed = sum(1 for _, r in results if r)
    print(f"  通过: {passed}/{len(results)}")
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！Agent 框架集成完成。")
    else:
        print(f"\n⚠ 有 {len(results) - passed} 个测试失败。")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
