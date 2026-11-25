"""Download a bounded snapshot or evaluate frozen A0/A0.5/B1 protocols."""
import argparse
from contextlib import ExitStack
from datetime import date
import json
from pathlib import Path

from .data import read_tasks, snapshot
from .runner import run
from .schema import Budget
from .workflow import MomentumBackend


def _add_frozen_paths(parser, *, tasks_default="tasks/development.jsonl"):
    parser.add_argument("--tasks", type=Path, default=Path(tasks_default))
    parser.add_argument("--data", type=Path, default=Path("data/yahoo-2024-v1"))


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
    evaluate.add_argument("--evidence", type=Path, help="Optional frozen financial evidence snapshot")
    evaluate.add_argument("--backend", choices=("offline", "llm"), default="offline")
    evaluate.add_argument("--seed", type=int, default=42, help="Statistics seed")
    evaluate.add_argument("--provider-seed", type=int, help="Optional generation seed if supported by provider")
    evaluate.add_argument("--max-tool-rounds", type=int, default=2)
    evaluate.add_argument("--max-tokens", type=int, default=1200)
    evaluate.add_argument("--phase", choices=("B1", "A1", "A2", "A3", "B2", "B3", "Reserve"), default="B1",
                          help="Project token budget phase for real LLM requests")
    evaluate.add_argument("--rag-index", type=Path, help="Open an existing A1 chunk index")
    evaluate.add_argument("--rag-mode", choices=("R1", "R2", "R3", "R4"), default="R3")
    evaluate.add_argument("--rag-dispatch", choices=("fixed", "agentic"), default="agentic")
    evaluate.add_argument("--model-cache", type=Path, default=Path("data/local-models"))
    evaluate.add_argument("--no-react", action="store_true", help="Prefetch the indicator evidence")

    baseline = sub.add_parser("b0", help="Run the frozen 20-task deterministic development baseline")
    _add_frozen_paths(baseline)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--seed", type=int, default=42)

    b1 = sub.add_parser("b1", help="Run a paired GLM smoke or Development experiment")
    _add_frozen_paths(b1)
    b1.add_argument("--evidence", type=Path, default=Path("data/financial-evidence-dev-003"))
    b1.add_argument("--b0-results", type=Path, default=Path("results/b0-development-001"))
    b1.add_argument("--output", type=Path, required=True)
    b1.add_argument("--scope", choices=("smoke", "development"), default="smoke")
    b1.add_argument("--seed", type=int, default=42)
    b1.add_argument("--budget-cap-cny", type=float,
                    help="Required for the 20-task Development run; must cover the conservative cost ceiling")

    fetch = sub.add_parser("evidence-fetch")
    for flag in ("tasks", "data", "cache", "output"):
        fetch.add_argument(f"--{flag}", type=Path, required=True)
    fetch.add_argument("--offline", action="store_true")
    fetch.add_argument("--apple-pages", nargs="+", type=int, default=[40, 41, 42, 43, 44])

    report = sub.add_parser("evidence-coverage")
    for flag in ("tasks", "evidence", "output"):
        report.add_argument(f"--{flag}", type=Path, required=True)

    index = sub.add_parser("rag-index", help="Index frozen A0.5 evidence or a local document manifest")
    index.add_argument("--evidence", type=Path, default=Path("data/financial-evidence-dev-003"))
    index.add_argument("--cache", type=Path, default=Path("data/evidence-cache-001"))
    index.add_argument("--manifest", type=Path, help="JSONL document metadata plus relative path")
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--model-cache", type=Path, default=Path("data/local-models"))
    index.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    index.add_argument("--strategy", choices=("structure", "fixed"), default="structure")
    index.add_argument("--chunk-size", type=int, default=220)
    index.add_argument("--overlap", type=int, default=32)

    bench = sub.add_parser("rag-benchmark", help="Run offline FinanceBench retrieval metrics")
    bench.add_argument("--source", type=Path, required=True)
    bench.add_argument("--pdf-root", type=Path, help="Local directory of PDFs from the pinned source revision")
    bench.add_argument("--workspace", type=Path, default=Path("data/local-models"))
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--pdf-engine", choices=("pypdf", "pdfium"), default="pypdf")
    bench.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    bench.add_argument("--modes", nargs="+", choices=("R1", "R2", "R3", "R4"), default=("R1", "R2", "R3", "R4"))

    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            result = snapshot(args.output, read_tasks(args.tasks), args.start, args.end)
        elif args.command == "evidence-fetch":
            from .evidence_pipeline import collect
            if len(args.apple_pages) > 12 or any(page < 1 for page in args.apple_pages):
                raise ValueError("Provide 1–12 positive archive pages")
            result = collect(args.tasks, args.data, args.output, args.cache,
                             offline=args.offline, apple_pages=args.apple_pages)
        elif args.command == "evidence-coverage":
            from .evidence_pipeline import coverage
            result = coverage(args.tasks, args.evidence, args.output)
        elif args.command == "rag-index":
            from .rag import build_index
            result = build_index(args)
        elif args.command == "rag-benchmark":
            from .rag_benchmark import benchmark
            result = benchmark(args.source, args.workspace, args.output, pdf_root=args.pdf_root,
                               device=args.device, modes=args.modes, pdf_engine=args.pdf_engine)
        elif args.command == "b0":
            budget = Budget()
            factory = lambda: MomentumBackend(budget)
            config = {"backend": "offline", "policy": "deterministic_momentum_5",
                      "model": None, "api_calls": 0}
            result = run(args.tasks, args.data, args.output, factory, config, budget, args.seed, True,
                         run_label="B0")
            if result["failed_tasks"]:
                raise SystemExit(1)
        elif args.command == "b1":
            from .b1 import run_b1
            result = run_b1(args.scope, args.tasks, args.data, args.evidence, args.b0_results,
                            args.output, args.seed, args.budget_cap_cny)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if result["status"] != "completed":
                raise SystemExit(2)
            return
        else:
            budget = Budget(max_tool_rounds=args.max_tool_rounds, max_tokens=args.max_tokens)
            with ExitStack() as stack:
                rag_factory = None
                if args.rag_index is not None:
                    from .rag import RagConfig, RagView, Retriever
                    from .vectors import LocalEmbedding, LocalReranker, VectorStore
                    if args.phase != "A1" or not (args.rag_index / "chunks-encoder.json").is_file():
                        raise ValueError("RAG requires --phase A1 and an existing index")
                    if args.no_react and args.rag_dispatch == "agentic":
                        raise ValueError("Agentic retrieval requires ReAct")
                    encoder = LocalEmbedding(args.model_cache)
                    store = stack.enter_context(VectorStore(args.rag_index, encoder, "chunks"))
                    reranker = LocalReranker(args.model_cache) if args.rag_mode == "R4" else None
                    cfg = RagConfig(mode=args.rag_mode, dispatch=args.rag_dispatch)
                    retriever = Retriever(store, reranker)
                    rag_factory = lambda task: RagView(retriever, task, "development", cfg)
                if args.backend == "offline":
                    factory = lambda: MomentumBackend(budget)
                    config = {"backend": "offline", "policy": "deterministic_momentum_5",
                              "model": None, "api_calls": 0}
                else:
                    from .provider import ModelBackend, client_from_env
                    from .token_budget import TokenBudget
                    client, model, api_host = client_from_env()
                    stack.enter_context(client)
                    token_budget = TokenBudget()
                    factory = lambda: ModelBackend(
                        client, model, budget, args.provider_seed,
                        token_budget=token_budget, phase=args.phase,
                        **({"max_prompt_bytes": 16000} if args.rag_index else {}))
                    config = {"backend": "llm", "model": model, "api_host": api_host,
                              "temperature": 0, "provider_seed": args.provider_seed,
                              "phase": args.phase, "sdk_retries": 0, "application_retries": 1}
                if rag_factory:
                    from dataclasses import asdict
                    config["rag"] = asdict(cfg)
                result = run(args.tasks, args.data, args.output, factory, config,
                             budget, args.seed, not args.no_react, evidence_path=args.evidence,
                             token_budget=token_budget if args.backend == "llm" else None,
                             experiment_id=str(args.output.resolve()),
                             phase=args.phase, rag_factory=rag_factory)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
