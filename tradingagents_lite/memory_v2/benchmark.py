"""A2-V2 offline benchmark entry point. No provider is constructed in this module."""
import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sqlite3
from statistics import mean
from time import perf_counter

from tokenizers import Tokenizer

from ..data import sha256
from ..rag import compact_json
from ..vectors import LocalEmbedding
from .dataset import archived_a2, chronological, reflect, v3_dev
from .evaluation import (SEED, anon, by_group, compression_summary, evaluate_compression,
                         evaluate_rank, paired, rank_summary)
from .retrieval import REPRESENTATIONS, RANKS, ExperimentStore, representation

REPO = Path(__file__).resolve().parents[2]
BASE = "341c4414cb9e4cccc12dc2909e98183ef8479f94"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_rows(path, rows):
    path.write_text("".join(compact_json(row) + "\n" for row in rows), encoding="utf-8")


class ProfiledEncoder(LocalEmbedding):
    """Instrument per-index writes and leave timed query paths uninstrumented."""

    def __init__(self, root, tokenizer):
        super().__init__(root, threads=2, batch=8, allow_download=False)
        self.untruncated = tokenizer
        self.profile = False
        self.builds = []

    def encode(self, texts, *, query=False):
        if not self.profile:
            return super().encode(texts, query=query)
        with sqlite3.connect(self.path) as db:
            before = db.execute("SELECT count(*) FROM vectors").fetchone()[0]
            start = perf_counter()
            vectors = super().encode(texts, query=query)
            duration = (perf_counter() - start) * 1000
            after = db.execute("SELECT count(*) FROM vectors").fetchone()[0]
        lengths = [len(item.ids) for item in self.untruncated.encode_batch(texts)]
        self.builds.append({"documents": len(texts), "unique": len(set(texts)), "cold_misses": after - before,
                            "cached_or_batch_reused": len(texts) - (after - before),
                            "mean_input_tokens": mean(lengths), "max_input_tokens": max(lengths),
                            "truncated": sum(length > 512 for length in lengths), "encode_ms": duration})
        return vectors


def populate(store, cases):
    encoder = store.encoder
    encoder.profile = True
    start = perf_counter()
    store.put_many([(case.episode, reflect(case)) for case in cases])
    elapsed = (perf_counter() - start) * 1000
    encoder.profile = False
    profiles = dict(zip(REPRESENTATIONS, encoder.builds, strict=True))
    encoder.builds.clear()
    consistency = store.consistency()
    if any(ids for mode in consistency.values() for ids in mode.values()):
        raise RuntimeError("Experiment storage consistency failure")
    return {"index_and_sqlite_ms": elapsed, "representations": profiles, "consistency": consistency}


def audit_scope(cases, history, rank_rows):
    single = {row["case"]: row for row in rank_rows}
    candidates = [single[anon(case.episode.source_task_id)]["legal"] for case in cases]
    relevant = [single[anon(case.episode.source_task_id)]["relevant"] for case in cases]
    old = [single[anon(case.episode.source_task_id)]["old_relevant"] for case in cases]
    cross_count = sum(
        h.episode.dataset_partition == case.episode.dataset_partition
        and h.episode.ticker != case.episode.ticker
        and h.episode.decision_time < case.episode.decision_time
        and h.episode.eligible_at <= case.episode.decision_time
        for h in history.values() for case in cases
    )
    return {"queries": len(cases), "same_ticker_candidates": sum(candidates),
            "hypothetical_cross_ticker_candidates_excluded": cross_count,
            "queries_without_candidates": candidates.count(0),
            "queries_with_1_to_3_candidates": sum(1 <= value <= 3 for value in candidates),
            "max_candidates": max(candidates), "candidate_histogram": dict(sorted(Counter(candidates).items())),
            "positive_proxy_queries": sum(bool(v) for v in relevant), "legacy_positive_queries": sum(bool(v) for v in old)}


def select_rank(rows):
    groups = by_group(rows, ("representation", "rank"), rank_summary)
    best_repr = max(REPRESENTATIONS, key=lambda e: groups[e + "/R1"]["ndcg@3"] or -1)
    best_rank = max(RANKS, key=lambda r: groups[best_repr + "/" + r]["ndcg@3"] or -1)
    base = groups["E0/R4"]
    new = groups[best_repr + "/" + best_rank]
    pure = groups[best_repr + "/R1"]
    support = sum(row["legal"] > 3 and row["relevant"] > 0 and row["representation"] == "E0"
                  and row["rank"] == "R4" for row in rows)
    accepted = (support >= 10 and (new["ndcg@3"] or 0) >= 1.05 * (base["ndcg@3"] or 0)
                and (new["recall@3"] or 0) >= (pure["recall@3"] or 0))
    return {"best_exploratory": [best_repr, best_rank], "queries_with_gt3_and_relevant": support,
            "calibration_gate_passed": accepted,
            "config": [best_repr, best_rank] if accepted else ["E0", "R4"]}


def select_compression(rows):
    groups = by_group(rows, ("mode", "budget"), compression_summary)
    base = groups["C1/1800"]
    acceptable = []
    for name in ("C2", "C3", "C4"):
        for budget in (900, 1800):
            group = groups.get(f"{name}/{budget}")
            if not group or not group["injected"]:
                continue
            safety = (not group["budget_violations"] and not group["pit_violations"]
                      and all(group[key] == group["injected"] for key in (
                          "references", "action", "numeric_content_exact", "eligible_date", "uncertainty")))
            recall_ok = (group["memory_delivery_recall"] or 0) >= (base["memory_delivery_recall"] or 0)
            lessons_ok = group["lesson"] >= base["lesson"]
            quality_gain = (group["memory_delivery_recall"] or 0) > (base["memory_delivery_recall"] or 0) or group["lesson"] > base["lesson"]
            cheaper = group["context_bytes"] <= .75 * base["context_bytes"]
            if safety and recall_ok and lessons_ok and (quality_gain if budget == 1800 else cheaper):
                acceptable.append((name, budget, group))
    if acceptable:
        winner = max(acceptable, key=lambda item: (item[2]["memory_delivery_recall"],
                                                   item[2]["lesson"], -item[2]["context_bytes"]))
        return list(winner[:2])
    return ["C1", 1800]


def run(args):
    output, work, source = map(Path, (args.output, args.work, args.source))
    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    model_path = source / "local-models/models"
    cache = work / "embeddings"
    cache.mkdir()
    (cache / "models").symlink_to(model_path.resolve(), target_is_directory=True)
    tokenizer = Tokenizer.from_file(str(model_path / LocalEmbedding.REVISION / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    encoder = ProfiledEncoder(cache, tokenizer)
    historical = json.loads((REPO / "results/a2-memory-local-001/summary.json").read_text())
    if encoder.metadata["weight_hashes"] != historical["model"]["weight_hashes"]:
        raise ValueError("BGE-small differs from historical weights")
    a2, a2_sources = archived_a2(source, REPO)
    v3, v3_sources = v3_dev(source, REPO)
    cohorts = {"a2-development": chronological(a2, 3, 3), "v3-development": chronological(v3, 24, 28)}
    splits = {part: [case for cohort in cohorts.values() for case in cohort[part]]
              for part in ("warmup", "calibration", "validation")}
    history = {case.episode.source_task_id: case for cases in splits.values() for case in cases}
    fingerprints = {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")}
    protocol = {"base_commit": BASE, "seed": SEED, "model": encoder.metadata, "threads": 2, "concurrency": 1,
                "environment": {"load_average_start": os.getloadavg(), "logical_cpus": os.cpu_count()},
                "inputs": [a2_sources, v3_sources], "code_sha256": fingerprints,
                "reference_code_sha256": {name: sha256(REPO / "tradingagents_lite" / name)
                                          for name in ("memory.py", "workflow.py", "data.py", "vectors.py")},
                "splits": {part: [anon(case.episode.version_hash) for case in cases]
                           for part, cases in splits.items()},
                "starting_cohorts": {"a2-development": 20, "v3-development": 80},
                "relevance": {"type": "rule regime proxy", "grades": "3=same momentum and risk regimes; "
                              "2=same momentum only; 1=same risk only; 0=neither", "relevant": "grade >= 2",
                              "legacy": "same ticker and action, scoring only"},
                "rank_weights": RANKS, "scope": "separate cohort partitions, same ticker, matured historical simulation",
                "selection_rule": {"representation": "maximize R1 nDCG@3 on Calibration; ties E0/E1/E2",
                    "ranking": "maximize Calibration nDCG@3 at selected representation; ties R0 to R4",
                    "promotion": ">=10 Calibration queries with >3 candidates and a relevant result, "
                                 ">=5% relative nDCG gain vs E0/R4, Recall@3 >= selected E/R1",
                    "C4_trigger": "C2 or C3 delivers fewer whole records than C1 on >=10% of eligible "
                                  "Calibration queries at 1800 bytes",
                    "compression": "safety and C1 lesson/recall parity plus quality gain at 1800 "
                                   "or 25% bytes saved at 900; tie by recall, lessons, bytes",
                    "E3_X1_X2": "skip when candidate support or independent transfer labels are insufficient"},
                "glm_requests": 0, "glm_tokens": 0}
    write(output / "protocol-freeze.json", protocol)
    with ExperimentStore(work / "store", encoder) as store:
        cal_build = populate(store, splits["warmup"] + splits["calibration"])
        cal, cal_state = evaluate_rank(store, splits["calibration"], history, "calibration")
        selected = select_rank(cal)
        modes = ["C0", "C1", "C2", "C3"]
        compressed = evaluate_compression(cal_state, selected["config"], tokenizer, modes)
        main = {(row["case"], row["mode"]): row for row in compressed if row["budget"] == 1800}
        eligible_rows = [row for row in main.values() if row["mode"] == "C1" and row["legal"] > 0]
        trigger_fraction = sum(
            any(len(main[row["case"], mode]["injected_ids"]) < len(row["injected_ids"]) for mode in ("C2", "C3"))
            for row in eligible_rows
        ) / max(1, len(eligible_rows))
        if trigger_fraction >= .10:
            modes.append("C4")
            compressed += evaluate_compression(cal_state, selected["config"], tokenizer, ["C4"])
        compression_pick = select_compression(compressed)
        freeze = {"protocol_sha256": sha256(output / "protocol-freeze.json"),
                  "ranking": selected, "compression": compression_pick, "compression_modes": modes,
                  "c4_trigger_fraction": trigger_fraction, "validation_evaluated": False}
        write(output / "freeze.json", freeze)
        val_build = populate(store, splits["validation"])
        val, val_state = evaluate_rank(store, splits["validation"], history, "validation")
        val_compressed = evaluate_compression(val_state, selected["config"], tokenizer, modes)
    export(output, history, splits, cal, val, compressed, val_compressed, cal_build, val_build, protocol, freeze)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(REPO.parent / "tradingagents-lite/data"))
    parser.add_argument("--work", default=str(REPO / "data/a2-memory-v2/001"))
    parser.add_argument("--output", default=str(REPO / "results/a2-memory-v2"))
    run(parser.parse_args())


from .exports import export

if __name__ == "__main__":
    main()
