"""A1-V3 candidate reranking for the public FinanceBench retrieval benchmark."""
import argparse
from collections import defaultdict
from contextlib import closing
from hashlib import sha256 as digest
from importlib.metadata import distributions
from math import isfinite
from pathlib import Path
import re
import resource
from statistics import mean
from time import perf_counter

from .data import sha256
from .rag import rrf, terms
from .rag_benchmark import CUTOFF, PARTITION, relevance, split
from .reranker_v3 import BGEReranker, MiniLMReranker
from .retrieval_v2 import (
    BASE, ROOT, Ranking, bootstrap, fuse, inputs, load_split, page_key, percentiles,
    read_json, read_rows, tree_hashes, write_json, write_rows,
)
from .retrieval_v2_chunks import Audit, normalize, numeric_line
from .retrieval_v3_candidates import (
    TEMPERATURES, V2_ARCHIVE, calibration_candidates, pool, public_questions, test_candidates,
)
from .vectors import LocalReranker

V2 = Ranking(candidate_k=50, rrf_k=60, dense_weight=.8, per_page=1)
ARMS = {"S0": 24, "S1": 50, "S2": 50, "S3": 50, "S4": 100}
RANK_FILES = ("retrieval_v3.py", "retrieval_v3_candidates.py", "reranker_v3.py")
RULES = {
    "quality_order": ["recall@5", "recall@1", "mrr@10", "line_coverage_at_5", "hot_mean_latency", "arm"],
    "deploy": {"min_gain": 0, "allowed_loss": .03, "cold_and_hot_mean_ratio": 2},
    "s5": {"oracle_gap_min": .20, "recall_at_5_below": .50, "candidate_base": "better of S3 and S4",
           "logit_boosts": {"metric": .35, "years": .20, "statement": .10, "section": .10}},
    "baseline": "A1-V2 E0 and selected weights",
    "budgets": {"top_5": 5, "max_chunk_results": 10, "s2_s4_page_quota": None},
    "timing": "shared CPU components plus each arm's separately timed ranking; CPU and cross encoder "
              "do not overlap; initialization excluded",
    "cold": "first pass on a new query cache with a company-scope clear per question; no OS cache eviction",
    "hot": "repeated queries and company-local scope reuse; cross-encoder outputs recomputed",
    "bootstrap": {"unit": "company", "repetitions": 10000, "seed": 47185},
}

METRIC_ALIASES = (
    ("operating cash flow", "net cash provided by operating activities", "cash from operations"),
    ("capital expenditures", "capital expenditure", "purchases of property", "capital spending"),
    ("net income", "net earnings", "net profit"),
    ("operating income", "income from operations", "operating profit"),
    ("total revenue", "net sales", "total sales", "total net revenue", "revenues", "revenue"),
    ("total assets",), ("total liabilities",), ("earnings per share", "diluted eps", "basic eps"),
    ("accounts receivable", "trade receivables"), ("stockholders equity", "shareholders equity"),
)
STATEMENT_ALIASES = (
    ("cash flow statement", "statements of cash flows", "statement of cash flows"),
    ("balance sheet", "balance sheets", "statements of financial position"),
    ("income statement", "statements of operations", "statements of income", "statements of earnings"),
)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def financial_boost(query, chunk):
    """Single prespecified bonus using only query, document text and source metadata."""
    q = " ".join(terms(query))
    body = " ".join(terms(chunk["text"]))
    headings = " ".join(terms(chunk.get("title", "") + " " + chunk.get("section", "")))

    def matching(aliases, target):
        return any(any(f" {a} " in f" {q} " for a in family)
                   and any(f" {a} " in f" {target} " for a in family) for family in aliases)

    years = set(YEAR.findall(q))
    # A year in a filing title does not establish the period displayed in a table.
    period_hits = len(years & set(YEAR.findall(body))) / len(years) if years else 0
    content = {word for word in q.split() if len(word) >= 5 and not word.isdigit()}
    heading_fraction = len(content & set(headings.split())) / len(content) if content else 0
    weights = RULES["s5"]["logit_boosts"]
    return (weights["metric"] * matching(METRIC_ALIASES, body)
            + weights["years"] * period_hits
            + weights["statement"] * matching(STATEMENT_ALIASES, body + " " + headings)
            + weights["section"] * heading_fraction)


def ordered_scores(scores):
    if any(not isfinite(score) for score in scores.values()):
        raise ValueError("Ranker returned nonfinite scores")
    return sorted(scores, key=lambda key: (-scores[key], key))


def oracle_at_five(keys, metadata, gold):
    """Five distinct available relevant pages is the most five chunks could cover."""
    return min(5, len({page_key(metadata[k]) for k in keys} & gold)) / len(gold)


def pseudonym(*values):
    return digest(repr(values).encode()).hexdigest()[:24]


def rank_once(arm, q, rankings, chunks, rerankers):
    started = perf_counter()
    lex, dense = rankings["bm25"], rankings["dense"]
    keys = pool(lex, dense, ARMS[arm])
    reference = [key for key, _ in rrf([lex, dense])]
    usage = {"milliseconds": 0., "peak_allocated_bytes": None, "peak_reserved_bytes": None,
             "oom_batch_splits": 0}
    failure = None
    try:
        if arm in ("S0", "S1"):
            configuration = BASE if arm == "S0" else V2
            ranked = fuse(lex, dense, chunks, configuration)
            values = {}
        else:
            reranker = rerankers["mini" if arm == "S2" else "bge"]
            raw, usage = reranker.score(q["query"], [chunks[key]["text"] for key in keys])
            values = dict(zip(keys, raw, strict=True))
            ranked = ordered_scores(values)
    except (RuntimeError, ValueError) as exc:
        values, ranked, failure = {}, [], type(exc).__name__
    duration = (perf_counter() - started) * 1000
    return {"pool": keys, "prior": reference, "ranked": ranked, "scores": values, "usage": usage,
            "failure": failure, "rank_ms": duration}


def enhance(q, chunks, row):
    started = perf_counter()
    boosted = {k: score + financial_boost(q["query"], chunks[k]) for k, score in row["scores"].items()}
    return {**row, "scores": boosted, "ranked": ordered_scores(boosted),
            "rank_ms": row["rank_ms"] + (perf_counter() - started) * 1000}


def load_rank_inputs(index):
    """Only fields prespecified for the scorer and source provenance."""
    # Use the experiment copy and close its local lock before any retrieval pass.
    from qdrant_client import QdrantClient

    with closing(QdrantClient(path=str(index))) as client:
        offset = None
        rows = {}
        while True:
            points, offset = client.scroll("chunks", offset=offset, limit=512)
            for point in points:
                fields = ("document_id", "page", "text", "title", "section",
                          "ticker", "pit_level", "eligible_ts", "dataset_partition")
                rows[str(point.id)] = {key: point.payload[key] for key in fields}
            if offset is None:
                break
    return rows


def check_candidate_scope(q, entry, chunks):
    keys = {key for ranking in entry["rankings"].values() for values in ranking.values() for key, _ in values}
    for key in keys:
        candidate = chunks[key]
        if (candidate["ticker"] != q["company"] or candidate["dataset_partition"] != PARTITION
                or candidate["eligible_ts"] > CUTOFF.timestamp()
                or candidate["pit_level"] not in ("historical_proxy", "verified_known_at")):
            raise ValueError("Candidate outside the frozen pre-ranking scope")


def rank_partition(questions, records, chunks, arms, rerankers, s5_base=None):
    for q in questions:
        check_candidate_scope(q, records[q["id"]], chunks)
    all_rows = {arm: {} for arm in arms}
    for temp in TEMPERATURES:
        for count, q in enumerate(questions, 1):
            record = records[q["id"]]
            stage = {}
            for arm in arms:
                if arm == "S5":
                    base = enhance(q, chunks, stage[s5_base])
                    depth = ARMS[s5_base]
                else:
                    depth = ARMS[arm]
                    base = rank_once(arm, q, record["rankings"][str(depth)], chunks, rerankers)
                    stage[arm] = base
                history = all_rows[arm].setdefault(q["id"], {"rank": base, "latency": {}})
                cost = dict(record["timing"][temp][str(depth)])
                order_changed = cost.pop("id_order_changed")
                usage = base["usage"]
                history["latency"][temp] = {
                    **cost, "ranking_ms": base["rank_ms"], "model_ms": usage["milliseconds"],
                    "total_ms": sum(cost.values()) + base["rank_ms"],
                    "peak_allocated_bytes": usage["peak_allocated_bytes"],
                    "peak_reserved_bytes": usage["peak_reserved_bytes"],
                    "oom_batch_splits": usage["oom_batch_splits"],
                    "rank_mismatch": history["rank"]["ranked"][:10] != base["ranked"][:10],
                    "retrieval_order_changed": order_changed,
                    "failure": base["failure"],
                }
            if count % 10 == 0 or count == len(questions):
                print(f"{temp} ranking {','.join(arms)}: {count}/{len(questions)}", flush=True)
    if s5_base and "S5" not in arms:
        raise ValueError("S5 was not scheduled")
    return all_rows


def extend_s5(questions, by_arm, chunks, base_arm):
    results = {}
    for q in questions:
        base = by_arm[base_arm][q["id"]]
        enhanced = enhance(q, chunks, base["rank"])
        cost = enhanced["rank_ms"] - base["rank"]["rank_ms"]
        results[q["id"]] = {"rank": enhanced, "latency": {
            temp: {**timing, "ranking_ms": timing["ranking_ms"] + cost, "total_ms": timing["total_ms"] + cost}
            for temp, timing in base["latency"].items()
        }}
    by_arm["S5"] = results


def audit_lines(source, questions, chunks, group):
    """After rankings are written, identify exact numeric-line proxies without emitting evidence text."""
    truth = {q["id"]: q for q in load_split(source, group)}
    raw = {r["financebench_id"]: r["evidence"] for r in read_rows(
        source / "data/financebench/data/financebench_open_source.jsonl") if split(r["company"]) == group}
    reader = Audit(source, None)
    by_page = defaultdict(list)
    for key, row in chunks.items():
        by_page[page_key(row)].append((key, normalize(row["text"])))
    result = {}
    for idx, q in enumerate(questions, 1):
        gold = truth[q["id"]]["gold"]
        line_hits = []
        for doc, page in sorted(gold):
            anchors = [normalize(e["evidence_text"]) for e in raw[q["id"]]
                       if e["doc_name"] == doc and e["evidence_page_num"] == page]
            for text in reader.page(doc, page):
                if not numeric_line(text):
                    continue
                line = normalize(text)
                if any(line in anchor for anchor in anchors):
                    line_hits.append({key for key, text in by_page[(doc, page)] if line in text})
        result[q["id"]] = {"gold": gold, "lines": line_hits}
        if idx % 25 == 0:
            print(f"Post-ranking numeric-line audit {idx}/{len(questions)}", flush=True)
    return result


def _first_page_rank(keys, chunks):
    first = {}
    for i, key in enumerate(keys, 1):
        first.setdefault(page_key(chunks[key]), i)
    return first


def evaluate(q, result, truth, chunks, rankings):
    rank = result["rank"]
    pred = rank["ranked"][:10]
    head = set(pred[:5])
    candidates = set(rank["pool"])
    gold = truth["gold"]
    got_pages = [page_key(chunks[k]) for k in pred]
    documents = list(dict.fromkeys(doc for doc, _ in got_pages))
    known_lines = truth["lines"]
    previous = _first_page_rank(rank["prior"], chunks)
    current = _first_page_rank(rank["ranked"], chunks)
    bm25_ranks = _first_page_rank([key for key, _ in rankings["bm25"]], chunks)
    dense_ranks = _first_page_rank([key for key, _ in rankings["dense"]], chunks)
    return {
        "id": q["id"], "company_group": pseudonym(q["company"]),
        "page": relevance(got_pages, gold),
        "document": relevance(documents, {doc for doc, _ in gold}),
        "candidate_page_recall": len({page_key(chunks[k]) for k in candidates} & gold) / len(gold),
        "candidate_oracle_at_5": oracle_at_five(candidates, chunks, gold),
        "candidate_count": len(candidates),
        "returned_count_at_5": len(head),
        "unique_pages_at_5": len(set(got_pages[:5])),
        "line_proxy": {"identified": len(known_lines), "index_complete": sum(bool(x) for x in known_lines),
                       "pool_complete": sum(bool(x & candidates) for x in known_lines),
                       "top5_complete": sum(bool(x & head) for x in known_lines)},
        "gold_page_ranks": [{"page_hash": pseudonym(*p), "bm25": bm25_ranks.get(p),
                             "dense": dense_ranks.get(p), "before_rrf": previous.get(p),
                             "after": current.get(p)} for p in sorted(gold)],
        "top5": [pseudonym(k) for k in pred[:5]],
        "latency": result["latency"], "failure": rank["failure"],
    }


def trace(arm, q, measurement, row):
    rank = row["rank"]
    lex = {key: i for i, (key, _) in enumerate(measurement["bm25"], 1)}
    dense = {key: i for i, (key, _) in enumerate(measurement["dense"], 1)}
    final = {key: i for i, key in enumerate(rank["ranked"], 1)}
    return {"arm": arm, "id": q["id"], "failure": rank["failure"], "candidates": [
        [pseudonym(key), lex.get(key), dense.get(key), i, rank["scores"].get(key), final.get(key)]
        for i, key in enumerate(rank["prior"], 1)
    ]}


def summarize(rows):
    line_totals = {k: sum(r["line_proxy"][k] for r in rows) for k in rows[0]["line_proxy"]}
    summaries = {
        "n": len(rows), "failures": sum(r["failure"] is not None for r in rows),
        **{kind: {key: mean(row[kind][key] for row in rows) for key in rows[0][kind]}
           for kind in ("page", "document")},
        **{key: mean(r[key] for r in rows)
           for key in ("candidate_page_recall", "candidate_oracle_at_5", "candidate_count",
                       "unique_pages_at_5", "returned_count_at_5")},
        "line_proxy": line_totals,
        "line_coverage_at_5": line_totals["top5_complete"] / max(line_totals["identified"], 1),
        "latency": {},
    }
    for temp in TEMPERATURES:
        timings = [row["latency"][temp] for row in rows]
        pairs = sum(row["candidate_count"] for row in rows
                    if row["latency"][temp]["model_ms"] > 0 and not row["latency"][temp]["failure"])
        model_ms = sum(t["model_ms"] for t in timings)
        summaries["latency"][temp] = {
            **{k: percentiles([timing[k] for timing in timings])
               for k in ("total_ms", "ranking_ms", "model_ms", "scope_ms", "encode_ms", "bm25_ms", "dense_ms")},
            "peak_allocated_bytes": max((t["peak_allocated_bytes"] or 0 for t in timings)),
            "peak_reserved_bytes": max((t["peak_reserved_bytes"] or 0 for t in timings)),
            "oom_batch_splits": sum(t["oom_batch_splits"] for t in timings),
            "pairs_per_second": (pairs * 1000 / model_ms) if model_ms else None,
            "scored_pairs": pairs,
            "rank_mismatch_count": sum(t["rank_mismatch"] for t in timings),
            "retrieval_order_changed_count": sum(t["retrieval_order_changed"] for t in timings),
            "execution_failures": sum(t["failure"] is not None for t in timings),
        }
    return summaries


def score_partition(questions, records, chunks, rankings, truth, s5_base=None):
    rowsets = {}
    for arm, values in rankings.items():
        depth = str(ARMS[s5_base] if arm == "S5" else ARMS[arm])
        rowsets[arm] = [
            evaluate(q, values[q["id"]], truth[q["id"]], chunks, records[q["id"]]["rankings"][depth])
            for q in questions
        ]
        if arm != "S1":
            continue
        # Check the S1 page quota against its own uncapped ranking, not a different scorer.
        unrestricted = Ranking(50, 60, .8)
        for q, result in zip(questions, rowsets[arm], strict=True):
            ranking = records[q["id"]]["rankings"][depth]
            plain = set(fuse(ranking["bm25"], ranking["dense"], chunks, unrestricted)[:5])
            chosen = set(values[q["id"]]["rank"]["ranked"][:5])
            line_sets = truth[q["id"]]["lines"]
            result["line_proxy"]["lost_to_page_quota"] = sum(bool(k & plain) and not k & chosen for k in line_sets)
            result["line_proxy"]["gained_with_page_quota"] = sum(bool(k & chosen) and not k & plain for k in line_sets)
    return rowsets


def quality_key(item):
    arm, s = item
    p = s["page"]
    return (-p["recall@5"], -p["recall@1"], -p["mrr@10"], -s["line_coverage_at_5"],
            s["latency"]["hot"]["total_ms"]["mean"], arm)


def select(stats):
    ordered = sorted(stats.items(), key=quality_key)
    e0 = stats["S0"]
    acceptable = [
        (arm, s) for arm, s in ordered
        if not s["failures"] and s["page"]["recall@5"] > e0["page"]["recall@5"]
        and all(s["page"][k] >= e0["page"][k] - RULES["deploy"]["allowed_loss"]
                for k in ("recall@1", "recall@10", "mrr@10"))
        and all(s["latency"][temp]["total_ms"]["mean"] <= RULES["deploy"]["cold_and_hot_mean_ratio"] *
                e0["latency"][temp]["total_ms"]["mean"] for temp in TEMPERATURES)
    ]
    return ordered[0][0], acceptable[0][0] if acceptable else "S0"


def model_set(source, model, device, which):
    result = {}
    if "mini" in which:
        result["mini"] = MiniLMReranker(source / "data/local-models")
    if "bge" in which:
        result["bge"] = BGEReranker(model, device)
    return result


def local_fingerprints(source, run, model):
    mini = source / "data/local-models/models" / LocalReranker.REVISION
    return {
        "A1_V2_inputs": inputs(source),
        "index_copy": tree_hashes(run / "original-index"),
        "V2_archive": {f: sha256(V2_ARCHIVE / f) for f in
                       ("freeze.json", "summary.json", "test-E0.jsonl", "test-stage1.jsonl")},
        "source_code": {name: sha256(ROOT / "tradingagents_lite" / name) for name in RANK_FILES},
        "mini_files": tree_hashes(mini),
        "bge_files": tree_hashes(model),
        "runtime_packages": {d.metadata["Name"]: d.version for d in distributions()},
        "calibration_candidates_sha256": sha256(run / "calibration-candidates.json"),
        "preflight_sha256": sha256(run / "probe.json"),
    }


def run_probe(source, run, model, device):
    run.mkdir(parents=True, exist_ok=True)
    models = model_set(source, model, device, ("mini", "bge"))
    q = "What was operating cash flow in fiscal 2023?"
    texts = ["Cash from operating activities (in millions): 2024 62 2023 54 2022 42. " * (i % 12 + 1)
             for i in range(24)]
    results = {}
    for name, reranker in models.items():
        scores, timing = reranker.score(q, texts[:2])
        if len(scores) != 2:
            raise ValueError("Preflight shape failure")
        _, sample = reranker.score(q, texts)
        results[name] = {"metadata": reranker.metadata, "warmup": timing, "sample": sample, "pairs": len(texts),
                         "sample_pairs_per_second": len(texts) * 1000 / sample["milliseconds"]}
        print(f"{name}: {sample['milliseconds']:.1f} ms for {len(texts)} pairs", flush=True)
    write_json(run / "probe.json", results)


def archive_outputs(out, questions, records, ranks, rows, s5_base):
    out.mkdir(exist_ok=True)
    stats = {arm: summarize(rowset) for arm, rowset in rows.items()}
    stats = {arm: {**s, "relative_to_S0": bootstrap(rows[arm], rows["S0"])} for arm, s in stats.items()}
    for arm, rowset in rows.items():
        write_rows(out / f"{arm}.jsonl", rowset)
    write_json(out / "summary.json", stats)
    trace_rows = []
    for arm, indexed in ranks.items():
        depth = str(ARMS[s5_base] if arm == "S5" else ARMS[arm])
        for q in questions:
            trace_rows.append(trace(arm, q, records[q["id"]]["rankings"][depth], indexed[q["id"]]))
    write_rows(out / "ranking-trace.jsonl", trace_rows)
    return stats


def calibrate(source, run, model, device):
    archive = read_json(run / "calibration-candidates.json")
    records = archive["measurements"]
    questions = public_questions(source, "calibration", archive["question_ids"])
    chunks = load_rank_inputs(run / "original-index")
    encoders = model_set(source, model, device, ("mini", "bge"))
    probe = read_json(run / "probe.json")
    if any(encoders[name].metadata != probe[name]["metadata"] for name in encoders):
        raise ValueError("Preflight model identity mismatch")
    ranks = rank_partition(questions, records, chunks, tuple(ARMS), encoders)
    write_json(run / "calibration-unscored.json", ranks)
    truth = audit_lines(source, questions, chunks, "calibration")
    scores = score_partition(questions, records, chunks, ranks, truth)
    stats = {arm: summarize(rows) for arm, rows in scores.items()}
    best_strong, strong = min(((a, stats[a]) for a in ("S3", "S4")), key=quality_key)
    gap = strong["candidate_oracle_at_5"] - strong["page"]["recall@5"]
    s5_base = None
    if gap >= RULES["s5"]["oracle_gap_min"] and strong["page"]["recall@5"] < RULES["s5"]["recall_at_5_below"]:
        s5_base = best_strong
        extend_s5(questions, ranks, chunks, s5_base)
        scores["S5"] = score_partition(questions, records, chunks, {"S5": ranks["S5"]}, truth, s5_base)["S5"]
    stats = archive_outputs(run / "calibration", questions, records, ranks, scores, s5_base)
    quality, deploy = select(stats)
    plan = list(dict.fromkeys(("S0", "S1", quality)))
    freeze = {
        "schema": "a1-retrieval-v3", "parent_commit": "7249c9eb3e0b8cbc7eb57a1f69a289613a7f8662",
        "rules": RULES, "inputs": local_fingerprints(source, run, model),
        "models": {key: value.metadata for key, value in encoders.items()},
        "device": device, "arms": ARMS,
        "calibration_cache_parity_by_depth": archive["candidate_parity_by_depth"], "quality_winner": quality, "calibration_deploy_candidate": deploy,
        "s5_base": s5_base, "s5_trigger": {"best_strong": best_strong, "oracle_gap": gap},
        "calibration_summary_sha256": sha256(run / "calibration/summary.json"),
        "formal_plan": plan, "test_question_count": 99, "glm_requests": 0, "glm_tokens": 0,
    }
    write_json(run / "freeze.json", freeze)
    print(f"Frozen {plan}, deployment candidate {deploy}: {sha256(run / 'freeze.json')}", flush=True)


def formal(source, run, model, device):
    freeze = read_json(run / "freeze.json")
    if (freeze["inputs"] != local_fingerprints(source, run, model) or freeze["rules"] != RULES
            or freeze["device"] != device or freeze["arms"] != ARMS
            or freeze["calibration_summary_sha256"] != sha256(run / "calibration/summary.json")):
        raise ValueError("Configuration or calibration no longer matches freeze")
    plan = freeze["formal_plan"]
    base = freeze["s5_base"]
    actual_arms = list(dict.fromkeys((["S0", "S1", base, "S5"] if "S5" in plan else plan)))
    depths = tuple(sorted({ARMS[arm] for arm in actual_arms if arm != "S5"}))
    required = {("mini" if arm == "S2" else "bge") for arm in actual_arms if arm not in ("S0", "S1", "S5")}
    encoders = model_set(source, model, device, required)
    if any(freeze["models"][k] != v.metadata for k, v in encoders.items()):
        raise ValueError("Frozen Cross-Encoder changed")
    test_candidates(source, run, depths)
    archive = read_json(run / "test-candidates.json")
    questions = public_questions(source, "test")
    if [q["id"] for q in questions] != archive["question_ids"]:
        raise ValueError("Test question identity changed")
    records = archive["measurements"]
    chunks = load_rank_inputs(run / "original-index")
    all_ranks = rank_partition(questions, records, chunks, actual_arms, encoders,
                               base if "S5" in plan else None)
    ranks = {arm: all_ranks[arm] for arm in plan}
    write_json(run / "test-unscored.json", ranks)
    truth = audit_lines(source, questions, chunks, "test")
    scored = score_partition(questions, records, chunks, ranks, truth, base)
    archive_outputs(run / "test", questions, records, ranks, scored, base)
    matching = {}
    for arm, label in (("S0", "E0"), ("S1", "stage1")):
        legacy = {r["id"]: r for r in read_rows(V2_ARCHIVE / f"test-{label}.jsonl")}
        matching[arm] = sum(row["page"] == legacy[row["id"]]["page"] for row in scored[arm])
    write_json(run / "test/checks.json", {"test_n": len(questions), "A1_V2_exact_page_metrics": matching,
              "freeze_sha256": sha256(run / "freeze.json"),
              "prediction_sha256": sha256(run / "test-unscored.json"),
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "glm_requests": 0, "glm_tokens": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "probe", "calibrate", "test"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--legacy-cache", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    source, run = args.source.resolve(), args.run.resolve()
    if args.command == "prepare":
        if not args.legacy_cache:
            parser.error("prepare requires --legacy-cache")
        calibration_candidates(source, args.legacy_cache.resolve(), run)
        return
    if args.model is None:
        parser.error(f"{args.command} requires --model")
    {"probe": run_probe, "calibrate": calibrate, "test": formal}[args.command](
        source, run, args.model.resolve(), args.device)


if __name__ == "__main__":
    main()

