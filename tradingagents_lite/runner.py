"""Isolated decision and settlement, with an append-only run ledger and exports."""
import csv
from datetime import UTC, datetime
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
from time import perf_counter

from .data import HistoricalData, read_tasks, sha256
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
        groups.setdefault(row["decision_date"], []).append(row["alpha"])
    clusters = list(groups.values())
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    statistics_ = []
    for _ in range(5000):
        sample = rng.choices(clusters, k=len(clusters))
        statistics_.append(statistics.fmean(v for cluster in sample for v in cluster))
    statistics_.sort()
    return [statistics_[125], statistics_[4874]]


def summary(rows: list[dict], seed: int) -> dict:
    scored = [r for r in rows if r["status"] == "ok" and r["alpha"] is not None]
    active = [r for r in scored if r["direction"] != 0]
    hits = sum(r["directional_hit"] for r in active)
    trials = [t for r in rows for t in r["tool_trace"]]
    failed = [r for r in rows if any(t["status"] == "error" for t in r["tool_trace"])]
    recovered = sum(
        r["status"] == "ok" and any(
            t["status"] == "ok" and any(p["status"] == "error" for p in r["tool_trace"][:i])
            for i, t in enumerate(r["tool_trace"])
        ) for r in failed
    )
    avg = statistics.fmean(r["alpha"] for r in scored) if scored else None
    usage = [r["usage"] for r in rows if r["usage"] is not None]
    return dict(
        tasks=len(rows), scored=len(scored), failed=len(rows)-len(scored), directional_n=len(active),
        directional_coverage=len(active)/len(rows),
        directional_hits=hits, directional_hit_rate=hits/len(active) if active else None,
        hit_rate_wilson_95=wilson(hits, len(active)), mean_alpha=avg,
        active_mean_alpha=statistics.fmean(r["alpha"] for r in active) if active else None,
        hold_reference_mean_alpha=0 if scored else None, delta_vs_hold=avg,
        alpha_date_bootstrap_95=clustered_bootstrap(scored, seed),
        date_clusters=len({r["decision_date"] for r in scored}),
        tool_calls=len(trials), tool_successes=sum(t["status"] == "ok" for t in trials),
        tool_success_rate=sum(t["status"] == "ok" for t in trials)/len(trials) if trials else None,
        tool_failure_tasks=len(failed), recovered_tasks=recovered,
        tool_recovery_rate=recovered/len(failed) if failed else None,
        latency_mean_seconds=statistics.fmean(r["latency_seconds"] for r in rows),
        token_usage={k: sum(u[k] for u in usage) for k in usage[0]} if usage else None,
        rag_recall_at_5=None, rag_mrr=None, memory_retrieval_quality=None,
        context_evidence_preservation=None,
    )


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def run(tasks_path: Path, data_path: Path, out: Path, backend_factory, model_config: dict,
        budget: Budget, seed: int, react: bool, evidence_path: Path | None = None) -> dict:
    """Run fixed tasks without sharing outcomes or context across cells."""
    tasks = read_tasks(tasks_path)
    market = HistoricalData(data_path)
    financial = EvidenceStore.load(evidence_path) if evidence_path is not None else None
    provenance = {
        "started_at": datetime.now(UTC).isoformat(), "code_git_sha": _git("rev-parse", "HEAD"),
        "code_worktree_dirty": bool(_git("status", "--porcelain")),
        "dependencies": {name: importlib.metadata.version(name) for name in ("langgraph", "pydantic", "openai")},
        "lock_sha256": sha256(Path("uv.lock")) if Path("uv.lock").exists() else None,
        "tasks_sha256": sha256(tasks_path), "snapshot_sha256": sha256(data_path / "manifest.json"),
        "snapshot": market.manifest, "config": model_config, "statistics_seed": seed,
        "budget": budget.model_dump(), "react": react,
        "ablation": "A0.5" if financial is not None else "A0",
        "execution": "next shared session open to horizon-th shared session close, OHLC price return",
        "alpha": "position direction × (asset price return − SPY price return); no costs",
    }
    if financial is not None:
        evidence_manifest = evidence_path / "source_manifest.json"
        if json.loads(evidence_manifest.read_text())["market_calendar_manifest_sha256"] != provenance["snapshot_sha256"]:
            raise ValueError("Financial evidence was gated with a different market calendar snapshot")
        provenance["evidence_manifest_sha256"] = sha256(evidence_manifest)
    out.mkdir(parents=True, exist_ok=False)
    (out / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    rows = []
    ledger = sqlite3.connect(out / "state.sqlite")
    try:
        ledger.execute("CREATE TABLE outcomes (task_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        with (out / "tasks.jsonl").open("x", encoding="utf-8") as stream:
            for task in tasks:
                start = perf_counter()
                backend = backend_factory()
                row = task.model_dump(mode="json")
                state = {}
                outcome = None
                try:
                    # The analyst closure receives only the clipped MarketView.
                    view = market.view(task)
                    state = build_workflow(view, backend, budget, react,
                        financial=financial.view(task) if financial is not None else None).invoke(
                        {}, {"recursion_limit": 30})
                    decision = Decision.model_validate(state["decision"])
                    status = "workflow_error" if state["errors"] else "ok"
                    outcome = market.outcome(task)
                except (ValueError, KeyError) as exc:
                    decision = Decision(action=Action.HOLD, rationale=type(exc).__name__, evidence_ids=[],
                                        risk_veto="data_or_workflow_error")
                    status = "unscored"
                direction = decision.action.direction
                row.update(status=status, decision=decision.model_dump(mode="json"), direction=direction,
                           errors=state.get("errors", []), evidence=state.get("evidence", {}),
                           tool_trace=state.get("tool_trace", []), report=state.get("report"),
                           proposal=state.get("proposal"), latency_seconds=perf_counter()-start,
                           usage=backend.usage, outcome=outcome.model_dump(mode="json") if outcome else None,
                           alpha=direction*(outcome.asset_return-outcome.benchmark_return)
                           if status == "ok" and outcome else None,
                           directional_hit=direction*outcome.asset_return > 0
                           if status == "ok" and outcome and direction else None)
                line = json.dumps(row, ensure_ascii=False)
                with ledger:
                    ledger.execute("INSERT INTO outcomes VALUES (?,?)", (task.task_id, line))
                stream.write(line + "\n")
                stream.flush()
                rows.append(row)
    finally:
        ledger.close()
    result = summary(rows, seed)
    with (out / "summary.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result))
        writer.writeheader()
        writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in result.items()})
    with (out / "REPORT.md").open("x", encoding="utf-8") as f:
        f.write("# TradingAgents-Lite historical evaluation\n\n"
                f"Backend: {model_config['backend']}. Run is Lite {provenance['ablation']}.\n\n"
                "These are OHLC snapshot research results. Historical provider revisions, corporate-action "
                "adjustments, hindsight universe and model pretraining are outside the cutoff guarantee. "
                "No transaction costs, dividend cash flows, borrow or executable portfolio are modelled. "
                "Intervals are descriptive, especially for small samples. "
                "The day bootstrap keeps same-day instruments together but does not model serial dependence. "
                "Null rates indicate absent denominators or unavailable modules.\n\n"
                "| Metric | Value |\n|---|---|\n")
        for k, v in result.items():
            f.write(f"| {k} | {json.dumps(v)} |\n")
    return result
