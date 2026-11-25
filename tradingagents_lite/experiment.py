"""Explicit execution boundary shared by the CLI and the local research UI."""
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import sqlite3

from .agents import A3Config, A3Runtime
from .data import close_available, sha256
from .runner import run
from .schema import Budget
from .token_budget import TokenBudget
from .workflow import MomentumBackend


A3_PROMPT_BYTES = 12000
A3_OUTPUT_TOKENS = 512


def upper_tokens(config: A3Config) -> int:
    # Analyst up to 3 calls, Trader 1, one repair for each, and at most two role pairs.
    roles = 0 if config.debate == "D0" else 4 if config.supplement else 2
    return (4 + (2 if config.repair else 0) + roles) * TokenBudget.reservation_size(
        A3_PROMPT_BYTES, A3_OUTPUT_TOKENS)


def _seed_interactive_memory(store, source: Path, task, mode, scope):
    from .memory import Episode

    if not source.is_file():
        return
    cutoff = close_available(task.decision_date).timestamp()
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("""
            SELECT record FROM versions WHERE partition_name='development' AND arm=?
            AND ticker=? AND decision_ts < ? AND outcome_ts <= ? AND eligible_ts <= ?
            ORDER BY eligible_ts, rowid
        """, (mode, task.symbol, cutoff, cutoff, cutoff)).fetchall()
    for (record,) in rows:
        episode = Episode.model_validate_json(record)
        store.put(episode.model_copy(update={"dataset_partition": "interactive", "experiment_arm": scope}))


def execute_experiment(tasks: Path, market: Path, output: Path, config: A3Config, *,
                       backend: str = "offline", evidence: Path | None = None,
                       rag_index: Path | None = None, rag_mode: str = "R3",
                       rag_dispatch: str = "fixed", memory_mode: str = "M0",
                       models: Path = Path("data/local-models"),
                       memory_seed: Path | None = None, run_id: str | None = None,
                       fundamentals: Path | None = None) -> dict:
    from .data import read_tasks

    if backend not in {"offline", "llm"} or memory_mode not in {"M0", "M1", "M2", "M3"}:
        raise ValueError("Unsupported backend or memory mode")
    if output.exists():
        raise FileExistsError("Run archive already exists")
    budget = Budget(max_tokens=A3_OUTPUT_TOKENS)
    run_id = run_id or str(output.resolve())
    selected = read_tasks(tasks)
    if any(task.task_id.startswith("v2-holdout-") for task in selected):
        raise ValueError("V2 holdout tasks are sealed")
    with ExitStack() as stack:
        rag_factory = memory = rag_options = encoder = ledger = None
        fundamentals_factory = fundamentals_options = None
        if fundamentals is not None:
            from .fundamentals import FundamentalsView

            fundamentals_options = {"root": str(fundamentals.resolve()), "partition": config.partition}
            if config.transport == "local":
                fundamentals_factory = lambda task: FundamentalsView.load(fundamentals, task, config.partition)
        if rag_index is not None:
            from .rag import RagConfig, RagView, Retriever
            from .vectors import LocalEmbedding, LocalReranker, VectorStore

            if not (rag_index / "chunks-encoder.json").is_file():
                raise FileNotFoundError("Build a local RAG index before enabling retrieval")
            cfg = RagConfig(mode=rag_mode, dispatch=rag_dispatch)
            rag_options = {"index": str(rag_index.resolve()), "models": str(models.resolve()),
                           "config": asdict(cfg), "partition": config.partition}
            if config.transport == "local":
                encoder = LocalEmbedding(models, allow_download=False)
                binding = stack.enter_context(VectorStore(rag_index, encoder, "chunks"))
                reranker = LocalReranker(models, allow_download=False) if rag_mode == "R4" else None
                retriever = Retriever(binding, reranker)
                rag_factory = lambda task: RagView(retriever, task, config.partition, cfg, provenance=True)
        if memory_mode != "M0":
            from .memory import MemoryConfig, MemoryReplay, MemoryStore
            from .vectors import LocalEmbedding

            encoder = encoder or LocalEmbedding(models, allow_download=False)
            store = stack.enter_context(MemoryStore(output.with_name(output.name + "-memory"), encoder))
            partition = "interactive" if memory_seed is not None else config.partition
            if memory_seed is not None:
                if len(selected) != 1:
                    raise ValueError("Interactive history imports require exactly one frozen task")
                _seed_interactive_memory(store, memory_seed, selected[0], memory_mode, run_id)
            memory = MemoryReplay(store, MemoryConfig(memory_mode, partition, run_id))

        def runtime(task, identity):
            transport = None
            if config.transport == "mcp":
                from .mcp_finance import MCPTransport

                transport = MCPTransport({"task": task.model_dump(mode="json"), "market": str(market.resolve()),
                    "evidence": str(evidence.resolve()) if evidence else None, "rag": rag_options,
                    "fundamentals": fundamentals_options})
            return A3Runtime(config, identity, task, transport=transport,
                             fixed_rag=bool(rag_index and rag_dispatch == "fixed"))

        description = {"backend": backend, "a3": asdict(config), "rag": rag_mode if rag_index else None,
                       "memory": memory_mode, "phase": "A3"}
        if fundamentals is not None:
            description["fundamentals_manifest_sha256"] = sha256(fundamentals / "manifest.json")
        if backend == "offline":
            factory = lambda: ResearchOffline(budget)
        else:
            from .provider import ModelBackend, client_from_env

            if len(selected) != 1:
                raise ValueError("This A3 LLM entry accepts one explicitly selected task")
            client, model, api_host = client_from_env()
            stack.enter_context(client)
            ledger = TokenBudget()
            if not ledger.can_reserve(upper_tokens(config), experiment_id=run_id, phase="A3")["fits"]:
                raise ValueError("Insufficient A3 or project token budget for this run")
            description.update(model=model, api_host=api_host)
            factory = lambda: ModelBackend(client, model, budget, None, token_budget=ledger,
                                           phase="A3", max_prompt_bytes=A3_PROMPT_BYTES, max_retries=0)
        return run(tasks, market, output, factory, description, budget, 42, True,
                   evidence_path=evidence, run_label="A3", phase="A3", rag_factory=rag_factory, memory=memory,
                   a3_factory=runtime, token_budget=ledger, experiment_id=run_id,
                   fundamentals_factory=fundamentals_factory)


class ResearchOffline(MomentumBackend):
    """Keep the momentum rule and cite retrieved context when it was actually observed."""

    def analyst(self, messages, tools):
        import json

        result = super().analyst(messages, tools)
        if not result.get("content") or result.get("tool_calls"):
            return result
        report = json.loads(result["content"])
        for message in messages[2:]:
            if message["role"] not in {"user", "tool"}:
                continue
            try:
                value = json.loads(message["content"])
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                report["evidence_ids"].extend(row["evidence_id"] for row in value.get("items", []))
        report["evidence_ids"] = list(dict.fromkeys(report["evidence_ids"]))[:4]
        return {"content": json.dumps(report)}

    def trader_a3(self, report, evidence, historical, debate, repair_code=None):
        from .schema import Action, Proposal

        stances = {role["action"] for role in debate.values()}
        if len(stances) > 1 and len(evidence) < 2:
            return Proposal(action=Action.HOLD, rationale="Conflicting arguments have thin current evidence.",
                            evidence_ids=report.evidence_ids)
        return self.trader(report, evidence)
