import argparse
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from runtime.runtime import AdaptiveInferenceRuntime


def run_once(query: str) -> None:
    runtime = AdaptiveInferenceRuntime()
    result = runtime.run(query)
    print("\n=== FINAL ANSWER ===")
    print(result["answer"])
    print("\n=== META ===")
    for key, value in result["meta"].items():
        print(f"{key}: {value}")
    runtime.trace.print_runtime_summary()


def interactive_mode() -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight Adaptive Inference Runtime MVP")
    parser.add_argument("--query", type=str, default="", help="Single query mode")
    args = parser.parse_args()

    if args.query:
        run_once(args.query)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
