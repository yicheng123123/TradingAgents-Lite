"""Download a bounded snapshot or evaluate frozen A0/A0.5/B1 protocols."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
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
    evaluate.add_argument("--memory-store", type=Path, help="Enable A2 with a persistent episode store")
    evaluate.add_argument("--memory-mode", choices=("M0", "M1", "M2", "M3"), default="M3")
    evaluate.add_argument("--memory-arm", help="Stable scope name. Defaults to the experiment output path.")
    evaluate.add_argument("--memory-partition", default="development")
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
    b1.add_argument("--resume-from", type=Path,
                    help="Reuse a validated successful dev-01/dev-02 Smoke and run the remaining 18 tasks")
    b1.add_argument("--budget-cap-cny", type=float,
                    help="Required for the 18-task continuation; must cover its conservative cost ceiling")

    b1_merge = sub.add_parser("b1-merge", help="Validate and merge the B1 Smoke and 18-task continuation archives")
    _add_frozen_paths(b1_merge)
    b1_merge.add_argument("--evidence", type=Path, default=Path("data/financial-evidence-dev-003"))
    b1_merge.add_argument("--b0-results", type=Path, default=Path("results/b0-development-001"))
    b1_merge.add_argument("--smoke", type=Path, required=True)
    b1_merge.add_argument("--continuation", type=Path, required=True)
    b1_merge.add_argument("--output", type=Path, required=True)
    b1_merge.add_argument("--seed", type=int, default=42)

    b1_v2 = sub.add_parser("b1-v2", help="Run the isolated paired GLM-5.3-Flash baseline protocol")
    _add_frozen_paths(b1_v2)
    b1_v2.add_argument("--evidence", type=Path, default=Path("data/financial-evidence-dev-003"))
    b1_v2.add_argument("--b0-results", type=Path, default=Path("results/b0-development-001"))
    b1_v2.add_argument("--output", type=Path, required=True)
    b1_v2.add_argument("--scope", choices=("smoke", "development"), default="smoke")
    b1_v2.add_argument("--smoke", type=Path,
                        help="Verified successful Flash Smoke archive required for Development")
    b1_v2.add_argument("--seed", type=int, default=42)

    a3 = sub.add_parser("a3", help="Run the optional multi-agent research graph")
    _add_frozen_paths(a3)
    a3.add_argument("--output", type=Path, required=True)
    a3.add_argument("--evidence", type=Path)
    a3.add_argument("--fundamentals", type=Path, help="Opt in to a frozen SEC fundamentals directory")
    a3.add_argument("--partition", choices=("development", "expansion-v2"), default="development")
    a3.add_argument("--rag-index", type=Path)
    a3.add_argument("--rag-mode", choices=("R1", "R2", "R3", "R4"), default="R3")
    a3.add_argument("--rag-dispatch", choices=("fixed", "agentic"), default="fixed")
    a3.add_argument("--memory-mode", choices=("M0", "M1", "M2", "M3"), default="M0")
    a3.add_argument("--backend", choices=("offline", "llm"), default="offline")
    a3.add_argument("--transport", choices=("local", "mcp"), default="local")
    a3.add_argument("--debate", choices=("D0", "D1", "D2"), default="D0")
    a3.add_argument("--verifier", action="store_true")
    a3.add_argument("--repair", action="store_true")
    a3.add_argument("--supplement", action="store_true")

    a3_bench = sub.add_parser("a3-benchmark", help="Run scripted, deterministic and real MCP offline cases")
    _add_frozen_paths(a3_bench)
    a3_bench.add_argument("--output", type=Path, required=True)
    a3_bench.add_argument("--evidence", type=Path, default=Path("data/financial-evidence-dev-003"))
    a3_bench.add_argument("--rag-index", type=Path, default=Path("data/a1-development-001"))

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

    episodic = sub.add_parser("memory-benchmark", help="Replay Development and test memory locally")
    _add_frozen_paths(episodic)
    episodic.add_argument("--model-cache", type=Path, default=Path("data/local-models"))
    episodic.add_argument("--output", type=Path, required=True)

    repair = sub.add_parser("memory-index", help="Check or rebuild the Qdrant projection from SQLite")
    repair.add_argument("--store", type=Path, required=True)
    repair.add_argument("--model-cache", type=Path, default=Path("data/local-models"))
    repair.add_argument("--rebuild", action="store_true")

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
        elif args.command == "memory-benchmark":
            from .memory_benchmark import benchmark
            result = benchmark(args.tasks, args.data, args.model_cache, args.output)
        elif args.command == "memory-index":
            from .memory import MemoryStore
            from .vectors import LocalEmbedding
            if not (args.store / "episodes.sqlite").is_file():
                raise ValueError("Expected an existing memory SQLite store")
            with MemoryStore(args.store, LocalEmbedding(args.model_cache)) as store:
                result = store.rebuild() if args.rebuild else store.consistency()
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
                            args.output, args.seed, args.budget_cap_cny, args.resume_from)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if result["status"] != "completed":
                raise SystemExit(2)
        elif args.command == "b1-merge":
            from .b1 import merge_b1
            result = merge_b1(args.smoke, args.continuation, args.tasks, args.data,
                              args.evidence, args.b0_results, args.output, args.seed)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if result["status"] != "completed":
                raise SystemExit(2)
            return
        elif args.command == "b1-v2":
            from .b1_v2 import run_b1_v2
            result = run_b1_v2(args.scope, args.tasks, args.data, args.evidence, args.b0_results,
                               args.output, seed=args.seed, smoke_path=args.smoke)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if result["status"] != "completed":
                raise SystemExit(2)
            return
        elif args.command == "a3":
            from .agents import A3Config
            from .experiment import execute_experiment
            config = A3Config(args.debate, args.transport, args.verifier, args.repair, args.supplement,
                              partition=args.partition)
            result = execute_experiment(args.tasks, args.data, args.output, config,
                backend=args.backend, evidence=args.evidence, rag_index=args.rag_index,
                rag_mode=args.rag_mode, rag_dispatch=args.rag_dispatch, memory_mode=args.memory_mode,
                fundamentals=args.fundamentals)
        elif args.command == "a3-benchmark":
            from .a3_benchmark import benchmark
            result = benchmark(args.tasks, args.data, args.evidence, args.rag_index, args.output)
        else:
            budget = Budget(max_tool_rounds=args.max_tool_rounds, max_tokens=args.max_tokens)
            with ExitStack() as stack:
                rag_factory = None
                memory = None
                encoder = None
                if args.memory_store is not None:
                    from .memory import MemoryConfig, MemoryReplay, MemoryStore
                    from .vectors import LocalEmbedding
                    if args.phase != "A2":
                        raise ValueError("Memory requires --phase A2")
                    encoder = LocalEmbedding(args.model_cache)
                    scope = f"{args.memory_mode}:{args.memory_arm or args.output.resolve()}"
                    store = stack.enter_context(MemoryStore(args.memory_store, encoder))
                    memory = MemoryReplay(store, MemoryConfig(args.memory_mode, args.memory_partition, scope))
                if args.rag_index is not None:
                    from .rag import RagConfig, RagView, Retriever
                    from .vectors import LocalEmbedding, LocalReranker, VectorStore
                    if args.phase not in {"A1", "A2"} or not (args.rag_index / "chunks-encoder.json").is_file():
                        raise ValueError("RAG requires phase A1 or A2 and an existing index")
                    if args.no_react and args.rag_dispatch == "agentic":
                        raise ValueError("Agentic retrieval requires ReAct")
                    encoder = encoder or LocalEmbedding(args.model_cache)
                    store = stack.enter_context(VectorStore(args.rag_index, encoder, "chunks"))
                    reranker = LocalReranker(args.model_cache) if args.rag_mode == "R4" else None
                    cfg = RagConfig(mode=args.rag_mode, dispatch=args.rag_dispatch)
                    retriever = Retriever(store, reranker)
                    partition = memory.config.partition if memory else "development"
                    rag_factory = lambda task: RagView(retriever, task, partition, cfg)
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
                        **({"max_prompt_bytes": 20000} if memory else
                           {"max_prompt_bytes": 16000} if args.rag_index else {}))
                    config = {"backend": "llm", "model": model, "api_host": api_host,
                              "temperature": 0, "provider_seed": args.provider_seed,
                              "phase": args.phase, "sdk_retries": 0, "application_retries": 1}
                if rag_factory:
                    config["rag"] = asdict(cfg)
                if memory is not None:
                    config["memory"] = asdict(memory.config)
                result = run(args.tasks, args.data, args.output, factory, config,
                             budget, args.seed, not args.no_react, evidence_path=args.evidence,
                             token_budget=token_budget if args.backend == "llm" else None,
                             experiment_id=str(args.output.resolve()),
                             phase=args.phase, rag_factory=rag_factory, memory=memory)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
