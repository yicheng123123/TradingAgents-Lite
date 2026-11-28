"""Scoring and paired summaries for frozen, chronologically scoped replay."""
from collections import Counter, defaultdict
from hashlib import sha256
from math import ceil, log2
import random
from statistics import mean
from time import perf_counter

from ..rag import compact_json
from .compression import BUDGETS, deliver, fidelity
from .dataset import ARM, relevance
from .retrieval import REPRESENTATIONS, RANKS, rank

SEED = 4202
TOP_K = 3


def anon(value):
    return sha256(("a2-v2:" + value).encode()).hexdigest()[:20]


def percentile(values, proportion):
    values = sorted(values)
    return values[max(0, ceil(len(values) * proportion) - 1)] if values else None


def recall(keys, relevant):
    return len(set(keys) & relevant) / len(relevant) if relevant else None


def ranking_metrics(keys, grades, old):
    relevant = {key for key, grade in grades.items() if grade >= 2}
    best = sorted(grades.values(), reverse=True)

    def dcg(values):
        return sum((2**grade - 1) / log2(index + 2) for index, grade in enumerate(values[:3]))

    ideal = dcg(best)
    return {
        **{f"recall@{k}": recall(keys[:k], relevant) for k in (1, 3, 5)},
        **{f"old_recall@{k}": recall(keys[:k], old) for k in (1, 3, 5)},
        "mrr": next((1 / (i + 1) for i, key in enumerate(keys) if key in relevant), 0) if relevant else None,
        "ndcg@3": dcg([grades[key] for key in keys]) / ideal if ideal else None,
        "oracle_recall": recall(keys, relevant),
    }


def evaluate_rank(store, cases, history, split):
    rows = []
    state = []
    for case in cases:
        ep = case.episode
        scope = {"cutoff": ep.decision_time, "partition": ep.dataset_partition, "arm": ARM}
        start = perf_counter()
        visible = store.scoped(ticker=ep.ticker, **scope)
        scope_ms = (perf_counter() - start) * 1000
        # Current outcome and current action are never read by ranking; action is only for legacy scoring.
        grades = {key: relevance(case, history[past.source_task_id]) for key, (past, _) in visible.items()}
        old = {key for key, (past, _) in visible.items() if past.action == ep.action}
        relevant = {key for key, g in grades.items() if g >= 2}
        # Prime each query once. All semantic search timings are warm on the same local cache.
        if visible:
            store.encoder.encode([case.query], query=True)
        case_row = {
            "case": anon(ep.source_task_id), "cluster": anon(ep.ticker), "cohort": ep.dataset_partition,
            "split": split, "legal": len(visible), "relevant": len(relevant), "old_relevant": len(old),
        }
        ranks = {}
        for representation in REPRESENTATIONS:
            start = perf_counter()
            cosine = store.cosine(case.query, representation, visible, **scope)
            search_ms = (perf_counter() - start) * 1000
            for mode in RANKS:
                start = perf_counter()
                ordered = rank(visible, cosine, ep.decision_time, mode)
                algorithm_ms = (perf_counter() - start) * 1000
                recalled = ordered[:TOP_K]
                legacy = deliver([visible[key] for key in recalled], case.query, 1800, "C1")
                by_id = {mem.memory_id: key for key, (mem, _) in visible.items()}
                ranks[representation, mode] = recalled
                rows.append({
                    **case_row, "representation": representation, "rank": mode,
                    **ranking_metrics(ordered, grades, old),
                    "latency_ms": scope_ms + algorithm_ms + (search_ms if mode != "R0" else 0),
                    "scope_ms": scope_ms, "vector_ms": search_ms if mode != "R0" else 0,
                    "recalled": [anon(visible[key][0].memory_id) for key in recalled],
                    "delivered_recall_c1": recall([by_id[mid] for mid in legacy["injected"]], relevant),
                })
        state.append((case, case_row, visible, grades, ranks))
    return rows, state


def evaluate_compression(state, configuration, tokenizer, modes):
    rows = []
    for case, case_row, visible, grades, ranks in state:
        keys = ranks[tuple(configuration)]
        recalled = [visible[key] for key in keys]
        by_id = {mem.memory_id: key for key, (mem, _) in visible.items()}
        relevant = {key for key, grade in grades.items() if grade >= 2}
        for budget in BUDGETS:
            for mode in modes:
                output = deliver(recalled, case.query, budget, mode)
                checked = fidelity(recalled, output)
                injected = [by_id[mid] for mid in output["injected"]]
                pit = sum(not (
                    visible[key][0].decision_time < case.episode.decision_time
                    and visible[key][0].outcome_available_at <= case.episode.decision_time
                    and visible[key][1].eligible_at <= case.episode.decision_time
                ) for key in injected)
                delivered = output["context_bytes"]
                rows.append({
                    **case_row, "mode": mode, "budget": budget,
                    "selected_ids": [anon(mid) for mid in output["selected"]],
                    "injected_ids": [anon(mid) for mid in output["injected"]],
                    "recalled_relevant": len(set(keys) & relevant),
                    "injected_relevant": len(set(injected) & relevant),
                    "memory_delivery_recall": recall(injected, relevant),
                    "raw_bytes": output["raw_bytes"], "context_bytes": delivered,
                    "byte_reduction": 1 - delivered / output["raw_bytes"],
                    "proxy_tokens": len(tokenizer.encode(compact_json(output["context"]), add_special_tokens=False).ids),
                    "latency_ms": output["latency_ms"], "budget_violations": int(delivered > budget),
                    "pit_violations": pit, **checked,
                })
    return rows


def avg_nonnull(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def rank_summary(rows):
    fields = ("recall@1", "recall@3", "recall@5", "old_recall@1", "old_recall@3", "old_recall@5",
              "mrr", "ndcg@3", "oracle_recall", "delivered_recall_c1", "latency_ms", "scope_ms", "vector_ms")
    return {
        "n": len(rows), "positive_n": sum(bool(row["relevant"]) for row in rows),
        "ndcg_defined_n": sum(row["ndcg@3"] is not None for row in rows),
        **{field: avg_nonnull(rows, field) for field in fields},
        "p95_ms": percentile([row["latency_ms"] for row in rows], .95),
    }


def compression_summary(rows):
    pre = sum(len(row["selected_ids"]) for row in rows)
    post = sum(len(row["injected_ids"]) for row in rows)
    fields = ("lesson", "references", "action", "eligible_date", "decision_date_present",
              "decision_date_exact", "outcome_present", "outcome_exact", "uncertainty",
              "structured_lesson", "numeric_content_exact", "budget_violations", "pit_violations",
              "recalled_relevant", "injected_relevant", "relevant")
    return {"n": len(rows), "selected": pre, "injected": post,
            "delivery_fraction": post / pre if pre else None,
            **{field: sum(row[field] for row in rows) for field in fields},
            **{field: avg_nonnull(rows, field) for field in ("memory_delivery_recall", "raw_bytes",
                                                            "context_bytes", "byte_reduction",
                                                            "proxy_tokens", "latency_ms")},
            "p95_ms": percentile([row["latency_ms"] for row in rows], .95)}


def by_group(rows, keys, summarize):
    groups = defaultdict(list)
    for row in rows:
        groups["/".join(str(row[key]) for key in keys)].append(row)
    return {key: summarize(group) for key, group in sorted(groups.items())}


def paired(first, second, metric):
    baseline = {row["case"]: row for row in first}
    clusters = defaultdict(list)
    counts = Counter(improved=0, declined=0, tied=0)
    for row in second:
        previous = baseline[row["case"]].get(metric)
        value = row.get(metric)
        if previous is not None and value is not None:
            delta = value - previous
            counts["improved" if delta > 0 else "declined" if delta < 0 else "tied"] += 1
            clusters[row["cluster"]].append(delta)
    values = [value for cluster in clusters.values() for value in cluster]
    draws = []
    if len(clusters) >= 2:
        rng = random.Random(SEED)
        ticker_groups = list(clusters.values())
        for _ in range(2000):
            sample = rng.choices(ticker_groups, k=len(ticker_groups))
            draws.append(mean(value for group in sample for value in group))
    return {"n": len(values), "ticker_clusters": len(clusters), **counts,
            "mean_delta": mean(values) if values else None,
            "interval_95": [percentile(draws, .025), percentile(draws, .975)] if draws else None}
