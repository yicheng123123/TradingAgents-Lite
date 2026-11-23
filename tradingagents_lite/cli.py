"""Download a bounded snapshot or evaluate a frozen task set."""
import argparse
from contextlib import ExitStack
from datetime import date
import json
from pathlib import Path

from .data import read_tasks, snapshot
from .runner import run
from .schema import Budget
from .workflow import MomentumBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("snapshot")
    download.add_argument("--tasks", type=Path, required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--start", type=date.fromisoformat, required=True)
    download.add_argument("--end", type=date.fromisoformat, required=True)
    evaluate = sub.add_parser("run")
    evaluate.add_argument("--tasks", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--backend", choices=("offline", "llm"), default="offline")
    evaluate.add_argument("--seed", type=int, default=42, help="Deterministic statistics seed")
    evaluate.add_argument("--provider-seed", type=int, help="Optional seed if supported by the provider")
    evaluate.add_argument("--max-tool-rounds", type=int, default=2)
    evaluate.add_argument("--max-tokens", type=int, default=1200)
    evaluate.add_argument("--no-react", action="store_true", help="Prefetch the indicator evidence")
    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            result = snapshot(args.output, read_tasks(args.tasks), args.start, args.end)
        else:
            budget = Budget(max_tool_rounds=args.max_tool_rounds, max_tokens=args.max_tokens)
            with ExitStack() as stack:
                if args.backend == "offline":
                    factory = lambda: MomentumBackend(budget)
                    config = {"backend": "offline", "policy": "deterministic_momentum_5",
                              "model": None, "api_calls": 0}
                else:
                    from .provider import ModelBackend, client_from_env
                    client, model, api_host = client_from_env()
                    stack.enter_context(client)
                    factory = lambda: ModelBackend(client, model, budget, args.provider_seed)
                    config = {"backend": "llm", "model": model, "api_host": api_host, "temperature": 0,
                              "provider_seed": args.provider_seed, "sdk_retries": 1}
                result = run(args.tasks, args.data, args.output, factory, config,
                             budget, args.seed, not args.no_react)
        print(json.dumps(result, indent=2))
        if args.command == "run" and result["failed"]:
            raise SystemExit(1)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
