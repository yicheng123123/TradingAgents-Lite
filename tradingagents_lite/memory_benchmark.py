"""Offline replay and synthetic checks for episodic memory behavior."""
from collections import Counter
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import resource
from statistics import mean
from time import perf_counter

from .data import close_available, read_tasks, sha256
from .memory import Episode, MemoryConfig, MemoryReplay, MemoryStore, Reflection
from .schema import Action, Outcome
from .vectors import LocalEmbedding


def _synthetic(root, encoder):
    base = datetime(2024, 6, 1, 22, tzinfo=UTC)
    outcome = Outcome(entry_day=base.date(), exit_day=base.date(), available_at=base,
                      asset_return=-0.01, benchmark_return=0)
    common = {"ticker": "AAPL", "decision_time": base - timedelta(days=2),
              "outcome_available_at": base, "reflection_created_at": base, "eligible_at": base,
              "action": Action.HOLD, "observed_evidence_ids": ["synthetic:1", "synthetic:2"],
              "outcome": outcome, "dataset_partition": "synthetic", "experiment_arm": "check"}
    target = Episode(source_task_id="liquidity", reflection=Reflection(
        lesson="Operational liquidity, working capital and cash flow should be reviewed.",
        potential_error="Check whether short-term liabilities were supported by cash."), **common)
    unrelated = Episode(source_task_id="product", reflection=Reflection(
        lesson="Compare laptop sales before and after the release.", potential_error="Product mix drift."), **common)
    cutoff = base + timedelta(days=2)
    revised = target.model_copy(update={"reflection_created_at": cutoff, "eligible_at": cutoff})
    options = {"ticker": "AAPL", "cutoff": cutoff, "partition": "synthetic", "arm": "check", "mode": "M2"}
    query = "short term liquidity cash flow working capital"
    with MemoryStore(root, encoder) as store:
        written = [store.put(ep) for ep in (target, target, unrelated, revised)]
        early = store.recall(query, **{**options, "cutoff": base - timedelta(seconds=1)})
        foreign = [store.recall(query, **{**options, **filter_})["eligible"] for filter_ in
                   ({"arm": "other"}, {"partition": "holdout"}, {"ticker": "MSFT"})]
    with MemoryStore(root, encoder) as store:
        semantic = store.recall(query, **options)
        late = next(row for row in semantic["selected"] if row["memory_id"] == target.memory_id)
        past = store.recall(query, **{**options, "cutoff": base})
        past_target = next(row for row in past["selected"] if row["memory_id"] == target.memory_id)
        store.vectors.client.delete(store.vectors.name, [revised.point_id])
        problem = store.consistency()
        repaired = store.rebuild()
        contexts = store.recall(query, **{**options, "mode": "M3"})
    return {"synthetic_queries": 1, "recall@1": int(semantic["selected"][0]["memory_id"] == target.memory_id),
            "recall@3": int(any(row["memory_id"] == target.memory_id for row in semantic["selected"])),
            "future_visible": early["eligible"], "negative_scope_checks": len(foreign),
            "negative_scope_visible": sum(foreign), "idempotent_flags": written,
            "version_history_correct": late["version_hash"] == revised.version_hash
                                       and past_target["version_hash"] == target.version_hash,
            "reopened_records_found": len(semantic["selected"]), "index_errors_detected": len(problem["missing"]),
            "index_errors_after_rebuild": sum(map(len, repaired.values())),
            "historical_ids_in_context": sum(len(x["historical_evidence_ids"]) for x in contexts["context"])}


def benchmark(tasks_path, market_path, cache, output):
    from .runner import run
    from .schema import Budget
    from .workflow import MomentumBackend

    start = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    encoder = LocalEmbedding(cache)
    tasks = read_tasks(tasks_path)
    task_index = {task.task_id: task for task in tasks}
    task_cutoffs = {task.task_id: close_available(task.decision_date) for task in tasks}
    budget = Budget()
    rows, results = [], {}
    for mode in ("M0", "M1", "M2", "M3"):
        with MemoryStore(output / "store", encoder) as store:
            replay = MemoryReplay(store, MemoryConfig(mode, "development", mode))
            run(tasks_path, market_path, output / mode, lambda: MomentumBackend(budget),
                {"backend": "offline", "memory": mode}, budget, 42, True,
                run_label=mode, phase="A2", memory=replay)
            task_rows = [json.loads(line) for line in (output / mode / "tasks.jsonl").read_text().splitlines()]
            episode_visibility = store._all()
            numbers = Counter()
            latencies, recalls = [], {1: [], 3: []}
            for row in task_rows:
                task = task_index[row["task_id"]]
                cutoff = task_cutoffs[task.task_id]
                mem = row["memory_trace"]
                # Offline label: same-ticker eligible prior episodes with the same observed action.
                # The relevance label uses completed archived decisions and is never an input query.
                relevant = {ep.source_task_id for ep in episode_visibility
                            if ep.experiment_arm == mode and ep.ticker == task.symbol
                            and ep.decision_time < cutoff and ep.outcome_available_at <= cutoff
                            and ep.eligible_at <= cutoff and ep.action.value == row["decision"]["action"]}
                # Give the control the same relevance denominator, using M1's SQLite replay after all arms.
                selected = [item["source_task_id"] for item in mem["selected"]]
                if relevant:
                    for k in recalls:
                        recalls[k].append(len(set(selected[:k]) & relevant) / len(relevant))
                numbers["retrieved"] += len(selected)
                numbers["injected"] += len(mem["injected_ids"])
                numbers["reused_tasks"] += bool(mem["injected_ids"])
                numbers["before_bytes"] += mem["before_bytes"]
                numbers["context_bytes"] += mem["context_bytes"]
                tagged = {item["memory_id"]: set(item["historical_evidence_ids"]) for item in mem["selected"]}
                for context in mem["context"]:
                    ids = tagged[context["memory_id"]]
                    numbers["historical_ids_expected"] += len(ids)
                    numbers["historical_ids_preserved"] += len(ids & set(context["historical_evidence_ids"]))
                bad = sum(task_cutoffs[item["source_task_id"]] >= cutoff for item in mem["selected"])
                eligible_ids = {ep.source_task_id for ep in episode_visibility
                                if ep.experiment_arm == mode and ep.outcome_available_at <= cutoff
                                and ep.eligible_at <= cutoff and ep.ticker == task.symbol}
                bad += sum(task_id not in eligible_ids for task_id in selected)
                numbers["timing_violations"] += bad
                latencies.append(mem["latency_ms"])
                rows.append({"mode": mode, "task_id": task.task_id, "relevant_ids": sorted(relevant),
                             "selected": mem["selected"], "injected_ids": mem["injected_ids"],
                             "passed_to_backend": mem["passed_to_backend"],
                             "before_bytes": mem["before_bytes"], "context_bytes": mem["context_bytes"],
                             "latency_ms": mem["latency_ms"]})
            results[mode] = {**numbers, "tasks": len(task_rows), "episodes": replay.written,
                             "relevance_queries": len(recalls[3]), "mean_query_ms": mean(latencies),
                             **{f"recall@{k}": mean(vals) if vals else None for k, vals in recalls.items()}}
    control = {r["task_id"]: r["relevant_ids"] for r in rows if r["mode"] == "M1"}
    for row in rows:
        if row["mode"] == "M0":
            row["relevant_ids"] = control[row["task_id"]]
    denom = sum(bool(ids) for ids in control.values())
    results["M0"].update(relevance_queries=denom, **{"recall@1": 0 if denom else None,
                                                    "recall@3": 0 if denom else None})
    synthetic = _synthetic(output / "synthetic", encoder)
    result = {"model": encoder.metadata, "tasks_sha256": sha256(tasks_path),
              "market_sha256": sha256(market_path / "manifest.json"),
              "development": results, "synthetic": synthetic,
              "store_bytes": sum(p.stat().st_size for p in (output / "store").rglob("*") if p.is_file()),
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "elapsed_seconds": perf_counter() - start, "glm_calls": 0}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "memory_queries.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return result
