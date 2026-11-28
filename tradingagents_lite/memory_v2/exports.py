"""Publish only aggregate figures and de-identified rows."""
from collections import Counter
from pathlib import Path

from ..data import sha256
from ..rag import compact_json
from .evaluation import anon, by_group, compression_summary, paired, rank_summary


def filter_rows(rows, **where):
    return [r for r in rows if all(r[key] == value for key, value in where.items())]


def audit_rows(states, history, rows):
    result = {}
    for split, cases in states.items():
        if split == "warmup":
            continue
        for cohort in sorted({case.episode.dataset_partition for case in cases}):
            subset = [case for case in cases if case.episode.dataset_partition == cohort]
            group = filter_rows(rows, split=split, cohort=cohort, representation="E0", rank="R4")
            distribution = Counter(row["legal"] for row in group)
            cross = sum(past.episode.dataset_partition == cohort
                        and past.episode.ticker != case.episode.ticker
                        and past.episode.decision_time < case.episode.decision_time
                        and past.episode.eligible_at <= case.episode.decision_time
                        for past in history.values() for case in subset)
            result[f"{split}/{cohort}"] = {
                "queries": len(subset), "same_ticker_candidates": sum(row["legal"] for row in group),
                "cross_ticker_candidates_excluded": cross, "empty": distribution[0],
                "empty_fraction": distribution[0] / len(subset),
                "one_to_three": sum(distribution[k] for k in (1, 2, 3)),
                "more_than_three": sum(n for k, n in distribution.items() if k > 3),
                "histogram": dict(distribution), "relevant_queries": sum(row["relevant"] > 0 for row in group),
                "legacy_relevant_queries": sum(row["old_relevant"] > 0 for row in group),
                "first_decision": str(min(case.episode.decision_time for case in subset).date()),
                "last_decision": str(max(case.episode.decision_time for case in subset).date()),
            }
    return result


def export(out, history, splits, cal, val, cal_compressed, val_compressed, cal_build, val_build, protocol, freeze):
    from json import dumps

    def emit(name, payload):
        (out / name).write_text(dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ranks = [*cal, *val]
    compression = [*cal_compressed, *val_compressed]
    rank_groups = by_group(ranks, ("split", "representation", "rank"), rank_summary)
    compression_groups = by_group(compression, ("split", "mode", "budget"), compression_summary)
    pairs = {}
    for split in ("calibration", "validation"):
        baseline = filter_rows(ranks, split=split, representation="E0", rank="R4")
        for rep in ("E0", "E1", "E2"):
            for variant in ("R0", "R1", "R2", "R3", "R4"):
                group = filter_rows(ranks, split=split, representation=rep, rank=variant)
                pairs[f"{split}/{rep}/{variant}"] = {
                    metric: paired(baseline, group, metric)
                    for metric in ("ndcg@3", "recall@1", "recall@3", "delivered_recall_c1")}
    compression_pairs = {}
    for split in ("calibration", "validation"):
        baseline = filter_rows(compression, split=split, mode="C1", budget=1800)
        for mode in freeze["compression_modes"]:
            for budget in (900, 1800, 3600):
                group = filter_rows(compression, split=split, mode=mode, budget=budget)
                compression_pairs[f"{split}/{mode}/{budget}"] = paired(baseline, group, "memory_delivery_recall")

    val_scope = filter_rows(ranks, split="validation", representation="E0", rank="R4")
    enough = sum(row["legal"] > 3 and row["relevant"] > 0 for row in val_scope) >= 10
    rank_axes = {"groups": rank_groups, "pairs_vs_E0_R4": pairs,
                 "cohorts": by_group(ranks, ("split", "cohort", "representation", "rank"), rank_summary)}
    rep_axes = {"groups": {k: v for k, v in rank_groups.items() if k.endswith("/R1")},
                "index_build": {"warmup_calibration": cal_build, "validation_append": val_build}}
    comp_axes = {"groups": compression_groups, "pairs_vs_C1_1800": compression_pairs}
    audits = audit_rows(splits, history, ranks)
    audit = {"inputs": protocol["inputs"], "episode_count": len(history),
             "splits": {key: dict(Counter(c.episode.dataset_partition for c in cases))
                        for key, cases in splits.items()},
             "queries": audits, "relevance": protocol["relevance"],
             "synthetic_mixed_into_historical": False, "sealed_gold_accessed": False}
    insight = {
        "representation": rep_axes, "ranking": rank_axes, "compression": comp_axes,
        "selection": freeze, "freeze_sha256": sha256(out / "freeze.json"),
        "ranking_evidence": "EXPLORATORY" if enough else "INSUFFICIENT EVIDENCE",
        "ranking_validation_queries_gt3_and_relevant": sum(row["legal"] > 3 and row["relevant"] > 0 for row in val_scope),
        "glm_requests": 0, "glm_tokens": 0, "current_trader_efficacy_tested": False,
    }
    emit("data-audit.json", audit)
    emit("representation-ablation.json", rep_axes)
    emit("ranking-ablation.json", rank_axes)
    emit("compression-ablation.json", comp_axes)
    emit("memory-delivery-audit.json", {
        "groups": {key: {metric: group[metric] for metric in (
            "n", "selected", "injected", "relevant", "recalled_relevant", "injected_relevant",
            "references", "lesson", "structured_lesson", "uncertainty", "action",
            "eligible_date", "decision_date_present", "decision_date_exact", "outcome_present",
            "outcome_exact", "numeric_content_exact", "pit_violations", "budget_violations")}
            for key, group in compression_groups.items()},
        "pit_violations": sum(row["pit_violations"] for row in compression),
        "budget_violations": sum(row["budget_violations"] for row in compression),
    })
    emit("summary.json", insight)
    for name, rows in (("ranking-cases.jsonl", ranks), ("delivery-cases.jsonl", compression)):
        (out / name).write_text("".join(compact_json(row) + "\n" for row in rows), encoding="utf-8")
    print(dumps({"output": str(out), "episodes": len(history), "ranking_rows": len(ranks),
                 "compression_rows": len(compression), "selection": freeze}, ensure_ascii=False, indent=2))
