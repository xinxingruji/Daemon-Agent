import argparse
from runtime.runtime import AdaptiveInferenceRuntime


def run_once(query: str) -> None:
    """推理模式：单次查询"""
    runtime = AdaptiveInferenceRuntime()
    result = runtime.run(query)
    print("\n=== FINAL ANSWER ===")
    print(result["answer"])
    print("\n=== META ===")
    for key, value in result["meta"].items():
        print(f"{key}: {value}")
    runtime.trace.print_runtime_summary()


def interactive_mode() -> None:
    """推理模式：交互式循环"""
    runtime = AdaptiveInferenceRuntime()
    print("Adaptive Inference Runtime (type 'exit' to quit)")
    while True:
        query = input("\nUser Query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        result = runtime.run(query)
        print("\nAssistant>")
        print(result["answer"])


def agent_mode() -> None:
    """Agent 模式：完整的 s_full 框架 + 工具系统"""
    # 动态导入 agent.py 避免依赖问题
    try:
        from agent import main as agent_main
        agent_main()
    except ImportError as e:
        print(f"Error: Could not import agent module: {e}")
        print("Make sure agent.py is in the same directory.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive Inference Runtime with s_full Framework Integration"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inference", "agent"],
        default="agent",
        help="Running mode: 'inference' (adaptive model routing), 'agent' (full agent loop with tools)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Single query mode (only for 'inference' mode)"
    )
    args = parser.parse_args()

    if args.mode == "agent":
        # 完整 Agent 模式
        agent_mode()
    else:
        # 推理模式
        if args.query:
            run_once(args.query)
        else:
            interactive_mode()


if __name__ == "__main__":
    main()
