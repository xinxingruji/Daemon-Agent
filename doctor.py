"""Command-line health check for Daemon-Agent."""

import argparse
import json
from pathlib import Path

from diagnostics import diagnostics_exit_code, render_diagnostics, run_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check Daemon-Agent configuration without exposing secrets.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip Ollama and LiteLLM reachability checks.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0,
        help="Network timeout in seconds (default: 2.0).",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero")
    results = run_diagnostics(
        Path.cwd(), offline=args.offline, timeout=args.timeout,
    )
    if args.json:
        print(json.dumps([item.as_dict() for item in results], ensure_ascii=False, indent=2))
    else:
        print(render_diagnostics(results))
    return diagnostics_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
