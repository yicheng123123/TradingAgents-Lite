"""Isolated decision and settlement, with append-only task trajectories and exports."""
import csv
from contextlib import nullcontext
from datetime import UTC, datetime
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
import shutil
from time import perf_counter

from .data import DataError, HistoricalData, close_available, read_tasks, sha256
from .evidence import EvidenceStore
from .schema import Action, Budget, Decision
from .workflow import build_workflow


def wilson(hits: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.959963984540054
    p = hits / n
    denominator = 1 + z*z/n
    middle = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / denominator
    return [max(0.0, middle-half), min(1.0, middle+half)]


def clustered_bootstrap(rows: list[dict], seed: int) -> list[float] | None:
    """Resample decision dates to keep same-date instruments together."""
    groups: dict[str, list[float]] = {}
    for row in rows:
        if row.get("alpha") is not None:
            groups.setdefault(row["decision_date"], []).append(row["alpha"])
    clusters = list(groups.values())
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(5000):
        draw = rng.choices(clusters, k=len(clusters))
        samples.append(statistics.fmean(value for cluster in draw for value in cluster))
    samples.sort()
    return [samples[125], samples[4874]]


def _percentile(values: list[float], percentile: float) -> float | None:
    """Linear-interpolated percentile, deterministic for small task/call samples."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _cost(usage: dict, model_config: dict) -> float | None:
    if model_config.get("backend") == "offline":
        return 0.0
    price = model_config.get("pricing_cny_per_million")
    if not price or usage.get("missing_usage_calls", 0) or usage.get("failed_calls", 0) or usage.get("retries", 0):
        return None
    prompt_tokens = usage["prompt_tokens"]
    cached_tokens = min(usage.get("cached_tokens") or 0, prompt_tokens)
    cached_rate = price.get("cached_input", price["input"])
    billed_output = usage["completion_tokens"] + usage.get("separate_reasoning_tokens", 0)
    return ((prompt_tokens - cached_tokens) * price["input"] + cached_tokens * cached_rate +
            billed_output * price["output"]) / 1_000_000


def _basic_summary(rows: list[dict], seed: int, model_config: dict) -> dict:
    scored = [r for r in rows if r["status"] == "ok" and r["alpha"] is not None]
    active = [r for r in scored if r["direction"] != 0]
    hits = sum(bool(r["directional_hit"]) for r in active)
    tool_events = [t for r in rows for t in r.get("tool_trace", [])]
    failed_events = [(r, i, t) for r in rows for i, t in enumerate(r.get("tool_trace", []))
                     if t.get("status") == "error"]
    recovered_events = sum(
        any(later.get("status") == "ok" for later in r.get("tool_trace", [])[i+1:])
        for r, i, _ in failed_events
    )
    recovered_tasks = len({r["task_id"] for r, i, _ in failed_events
                           if any(later.get("status") == "ok" for later in r.get("tool_trace", [])[i+1:])})
    structured = [event for r in rows for event in r.get("structured_output_trace", [])]
    citations = [event for r in rows for event in r.get("citation_trace", [])]
    cited = [item for event in citations for item in event.get("cited_ids", [])]
    valid_cited = [item for event in citations for item in event.get("valid_ids", [])]
    invalid_cited = [item for event in citations for item in event.get("invalid_ids", [])]
    financial_cited = [item for event in citations for item in event.get("financial_ids", [])]
    usage_rows = [r["usage"] for r in rows if isinstance(r.get("usage"), dict)]
    total_usage = {
        key: sum(item.get(key, 0) or 0 for item in usage_rows)
        for key in ("calls", "logical_calls", "retries", "failed_calls", "prompt_tokens",
                    "completion_tokens", "reasoning_tokens", "separate_reasoning_tokens", "cached_tokens",
                    "total_tokens_charged", "missing_usage_calls", "reasoning_detail_missing_calls",
                    "cached_detail_missing_calls", "budget_denials")
    }
    all_alpha = [r["alpha"] for r in scored]
    by_model = sorted({call.get("actual_model") for r in rows for call in r.get("llm_calls", [])
                       if call.get("actual_model")})
    task_latency = [r["latency_seconds"] for r in rows]
    call_latency = [call["latency_seconds"] for r in rows for call in r.get("llm_calls", [])
                    if call.get("latency_seconds") is not None]
    return {
        "tasks": len(rows),
        "completed_tasks": len(scored),
        "task_completion_rate": len(scored) / len(rows) if rows else None,
        "failed_tasks": len(rows) - len(scored),
        "failure_status_counts": {
            status: sum(r["status"] == status for r in rows)
            for status in sorted({r["status"] for r in rows if r["status"] != "ok"})
        },
        "structured_output_count": len(structured),
        "structured_output_valid_count": sum(event.get("valid") is True for event in structured),
        "structured_output_valid_rate": (
            sum(event.get("valid") is True for event in structured) / len(structured) if structured else None
        ),
        "structured_output_valid_tasks": sum(
            len(r.get("structured_output_trace", [])) == 2 and
            all(event.get("valid") is True for event in r["structured_output_trace"])
            for r in rows
        ),
        "directional_n": len(active),
        "non_hold_coverage": len(active) / len(rows) if rows else None,
        "directional_coverage": len(active) / len(rows) if rows else None,
        "directional_hits": hits,
        "directional_hit_rate": hits / len(active) if active else None,
        "hit_rate_wilson_95": wilson(hits, len(active)),
        "mean_alpha": statistics.fmean(all_alpha) if all_alpha else None,
        "active_mean_alpha": statistics.fmean(r["alpha"] for r in active) if active else None,
        "hold_reference_mean_alpha": 0.0 if scored else None,
        "delta_vs_hold": statistics.fmean(all_alpha) if all_alpha else None,
        "alpha_date_bootstrap_95": clustered_bootstrap(scored, seed),
        "date_clusters": len({r["decision_date"] for r in scored}),
        "tool_calls": len(tool_events),
        "tool_successes": sum(t.get("status") == "ok" for t in tool_events),
        "tool_success_rate": (
            sum(t.get("status") == "ok" for t in tool_events) / len(tool_events) if tool_events else None
        ),
        "tool_failure_events": len(failed_events),
        "tool_recovery_events": recovered_events,
        "tool_recovered_tasks": recovered_tasks,
        "tool_recovery_rate": recovered_events / len(failed_events) if failed_events else None,
        "citation_references": len(cited),
        "valid_citation_references": len(valid_cited),
        "invalid_citation_references": len(invalid_cited),
        "evidence_citation_valid_rate": len(valid_cited) / len(cited) if cited else None,
        "invalid_citation_rate": len(invalid_cited) / len(cited) if cited else None,
        "financial_citation_references": len(financial_cited),
        "eligible_financial_citation_task_rate": (
            len({r["task_id"] for r in rows if any(
                event.get("financial_ids") for event in r.get("citation_trace", []))}) / len(rows)
            if rows else None
        ),
        "llm_calls": total_usage["calls"],
        "llm_logical_calls": total_usage["logical_calls"],
        "llm_retries": total_usage["retries"],
        "llm_failed_calls": total_usage["failed_calls"],
        "input_tokens": total_usage["prompt_tokens"],
        "output_tokens": total_usage["completion_tokens"],
        "reasoning_tokens": total_usage["reasoning_tokens"],
        "cached_tokens": total_usage["cached_tokens"],
        "total_tokens_charged": total_usage["total_tokens_charged"],
        "missing_usage_calls": total_usage["missing_usage_calls"],
        "reasoning_detail_missing_calls": total_usage["reasoning_detail_missing_calls"],
        "cached_detail_missing_calls": total_usage["cached_detail_missing_calls"],
        "token_budget_denials": total_usage["budget_denials"],
        "api_cost_cny_estimate": _cost(total_usage, model_config),
        "actual_models_observed": by_model,
        "latency_mean_seconds": statistics.fmean(task_latency) if task_latency else None,
        "latency_p50_seconds": _percentile(task_latency, 0.50),
        "latency_p95_seconds": _percentile(task_latency, 0.95),
        "llm_call_latency_mean_seconds": statistics.fmean(call_latency) if call_latency else None,
        "llm_call_latency_p50_seconds": _percentile(call_latency, 0.50),
        "llm_call_latency_p95_seconds": _percentile(call_latency, 0.95),
        "input_tokens_per_task": (
            total_usage["prompt_tokens"] / len(rows) if rows and usage_rows else None
        ),
        "output_tokens_per_task": (
            total_usage["completion_tokens"] / len(rows) if rows and usage_rows else None
        ),
        "reasoning_tokens_per_task": (
            total_usage["reasoning_tokens"] / len(rows) if rows and usage_rows else None
        ),
        "total_tokens_charged_per_task": (
            total_usage["total_tokens_charged"] / len(rows) if rows and usage_rows else None
        ),
        "completed_tasks_per_100k_tokens": (
            len(scored) * 100_000 / total_usage["total_tokens_charged"]
            if total_usage["total_tokens_charged"] else None
        ),
        "directional_hits_per_100k_tokens": (
            hits * 100_000 / total_usage["total_tokens_charged"]
            if total_usage["total_tokens_charged"] else None
        ),
        "token_usage": total_usage if usage_rows else None,
        "rag_recall_at_5": None,
        "rag_mrr": None,
        "memory_retrieval_quality": None,
        "context_evidence_preservation": None,
    }


def summary(rows: list[dict], seed: int, model_config: dict | None = None) -> dict:
    config = model_config or {}
    result = _basic_summary(rows, seed, config)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("symbol", "UNKNOWN"), []).append(row)
    result["by_ticker"] = {
        symbol: _basic_summary(group, seed, config) for symbol, group in sorted(groups.items())
    }
    return result


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def run(tasks_path: Path, data_path: Path, out: Path, backend_factory, model_config: dict,
        budget: Budget, seed: int, react: bool, evidence_path: Path | None = None,
        run_label: str | None = None, protocol_metadata: dict | None = None,
        token_budget=None, experiment_id: str | None = None, phase: str | None = None,
        rag_factory=None, memory=None, a3_factory=None, fundamentals_factory=None) -> dict:
    """Run fixed tasks without sharing outcomes or context across cells."""
    tasks = read_tasks(tasks_path)
    if memory is not None:
        tasks.sort(key=lambda task: (close_available(task.decision_date), task.task_id))
    market = HistoricalData(data_path)
    if any(market.is_sealed(task) for task in tasks):
        raise DataError("V2 holdout tasks are sealed for interactive and benchmark execution")
    financial = EvidenceStore.load(evidence_path) if evidence_path is not None else None
    experiment_id = experiment_id or str(out.resolve())
    phase = phase or model_config.get("phase", "B1")
    provenance = {
        "started_at": datetime.now(UTC).isoformat(), "code_git_sha": _git("rev-parse", "HEAD"),
        "code_worktree_dirty": bool(_git("status", "--porcelain")),
        "dependencies": {name: importlib.metadata.version(name) for name in ("langgraph", "pydantic", "openai")},
        "lock_sha256": sha256(Path("uv.lock")) if Path("uv.lock").exists() else None,
        "tasks_sha256": sha256(tasks_path), "snapshot_sha256": sha256(data_path / "manifest.json"),
        "snapshot": market.manifest, "config": model_config, "statistics_seed": seed,
        "budget": budget.model_dump(), "react": react,
        "ablation": run_label or ("A0.5" if financial is not None else "A0"),
        "execution": "next shared session open to horizon-th shared session close, OHLC price return",
        "alpha": "position direction × (asset price return − SPY price return); no costs",
        "protocol_metadata": protocol_metadata or {},
        "token_budget_phase": phase,
    }
    if financial is not None:
        evidence_manifest = evidence_path / "source_manifest.json"
        if json.loads(evidence_manifest.read_text())["market_calendar_manifest_sha256"] != provenance["snapshot_sha256"]:
            raise ValueError("Financial evidence was gated with a different market calendar snapshot")
        provenance["evidence_manifest_sha256"] = sha256(evidence_manifest)
        provenance["evidence_sha256"] = sha256(evidence_path / "evidence.jsonl")
    out.mkdir(parents=True, exist_ok=False)
    run_path = out / "run.json"
    run_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = []
    budget_exhausted = False
    ledger = sqlite3.connect(out / "state.sqlite")
    try:
        ledger.execute("CREATE TABLE outcomes (task_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        with (out / "tasks.jsonl").open("x", encoding="utf-8") as stream:
            for task in tasks:
                start = perf_counter()
                backend = backend_factory()
                backend_budget = getattr(backend, "token_budget", None)
                if token_budget is None and backend_budget is not None:
                    token_budget = backend_budget
                if hasattr(backend, "set_budget_context"):
                    backend.set_budget_context(
                        experiment_id=experiment_id, task_id=task.task_id, phase=phase)
                row = task.model_dump(mode="json")
                state, outcome = {}, None
                runtime = None
                try:
                    view = market.view(task)
                    runtime = a3_factory(task, experiment_id) if a3_factory is not None else None
                    with runtime if runtime is not None else nullcontext():
                        state = build_workflow(view, backend, budget, react,
                             financial=financial.view(task) if financial is not None else None,
                             rag=rag_factory(task) if rag_factory is not None else None,
                             fundamentals=fundamentals_factory(task) if fundamentals_factory is not None else None,
                             memory=memory, a3=runtime,
                             max_tool_calls_per_round=model_config.get("max_tool_calls_per_round", 2),
                             compact_tool_results=model_config.get("compact_tool_results", False),
                             flash_v2_prompt=model_config.get("protocol_version") == "b1-flash-v2"
                             ).invoke({}, {"recursion_limit": 40 if runtime else 30})
                    decision = Decision.model_validate(state["decision"])
                    model_error = any(e.get("stage") == "model" for e in state.get("errors", []))
                    status = "model_error" if model_error else "workflow_error" if state["errors"] else "ok"
                except Exception as exc:
                    decision = Decision(action=Action.HOLD, rationale=type(exc).__name__, evidence_ids=[],
                                        risk_veto="workflow_failure")
                    state = {"errors": [{"stage": "runtime", "code": type(exc).__name__}]}
                    status = "runtime_error"
                try:
                    outcome = market.outcome(task)
                except DataError as exc:
                    state.setdefault("errors", []).append({"stage": "settlement", "code": type(exc).__name__})
                    status = "settlement_error"
                direction = decision.action.direction
                call_log = getattr(backend, "call_log", [])
                usage = backend.usage
                row.update(
                    status=status, decision=decision.model_dump(mode="json"), direction=direction,
                    errors=state.get("errors", []), evidence=state.get("evidence", {}),
                    tool_trace=state.get("tool_trace", []), structured_output_trace=state.get("structured_output_trace", []),
                    citation_trace=state.get("citation_trace", []), report=state.get("report"),
                    proposal=state.get("proposal"), llm_calls=call_log,
                    latency_seconds=perf_counter()-start,
                    usage=usage, outcome=outcome.model_dump(mode="json") if outcome else None,
                    alpha=direction*(outcome.asset_return-outcome.benchmark_return)
                    if status == "ok" and outcome else None,
                    directional_hit=direction*outcome.asset_return > 0
                    if status == "ok" and outcome and direction else None,
                )
                if memory is not None:
                    row["memory_trace"] = state.get("memory_trace")
                    if status == "ok" and outcome is not None:
                        memory.enqueue(task, decision, outcome, state.get("evidence", {}))
                if runtime is not None:
                    row["agent_trace"] = runtime.events
                    row["debate"] = state.get("debate", {})
                    row["debate_trace"] = state.get("debate_trace", [])
                    row["role_calls"] = runtime.role_calls
                    row["verification"] = state.get("verification")
                line = json.dumps(row, ensure_ascii=False)
                with ledger:
                    ledger.execute("INSERT INTO outcomes VALUES (?,?)", (task.task_id, line))
                stream.write(line + "\n")
                stream.flush()
                rows.append(row)
                if any(call.get("error_code") == "token_budget_exhausted"
                       for call in call_log):
                    budget_exhausted = True
                    break
    finally:
        ledger.close()

    if memory is not None and not budget_exhausted:
        memory.finish()
    shutil.copyfile(out / "tasks.jsonl", out / "trajectory.jsonl")
    result = summary(rows, seed, model_config)
    result.update(requested_tasks=len(tasks), run_status="budget_exhausted" if budget_exhausted else "completed",
                  budget_exhausted=budget_exhausted)
    if token_budget is not None:
        result["token_budget"] = token_budget.snapshot(experiment_id=experiment_id, phase=phase)
    with (out / "summary.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result))
        writer.writeheader()
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in result.items()})
    with (out / "REPORT.md").open("x", encoding="utf-8") as f:
        f.write("# TradingAgents-Lite experiment run\n\n"
                f"Backend: {model_config['backend']}. Variant: {provenance['ablation']}.\n\n"
                "These are OHLC snapshot research results. Historical provider revisions, corporate-action "
                "adjustments, hindsight universe and model pretraining are outside the cutoff guarantee. "
                "No transaction costs, dividend cash flows, borrow or executable portfolio are modelled. "
                "Intervals are descriptive; the date bootstrap keeps same-day instruments together and "
                "does not model serial dependence. Failed tasks are excluded from scored returns and remain "
                "visible in task trajectories.\n\n| Metric | Value |\n|---|---|\n")
        for key, value in result.items():
            f.write(f"| {key} | {json.dumps(value, ensure_ascii=False)} |\n")
    observed_models = sorted({call.get("actual_model") for row in rows for call in row["llm_calls"]
                              if call.get("actual_model")})
    provenance.update(finished_at=datetime.now(UTC).isoformat(),
                      status="budget_exhausted" if budget_exhausted else "completed",
                      requested_tasks=len(tasks), processed_tasks=len(rows),
                      actual_models_observed=observed_models)
    if token_budget is not None:
        provenance["token_budget"] = result["token_budget"]
    run_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
