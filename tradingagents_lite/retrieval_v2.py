"""Offline FinanceBench retrieval experiment. Does not import a model provider."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from hashlib import sha256 as digest
from itertools import product
import json
from pathlib import Path
import platform
from random import Random
import resource
import shutil
import subprocess
from time import perf_counter

from .data import sha256
from .rag import Retriever, bm25, rrf
from .rag_benchmark import CUTOFF, PARTITION, relevance, split
from .vectors import LocalEmbedding, VectorStore


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "results/a1-rag-local-001"
BASE_INDEX = "financebench-index-financebench-run-002"
PUBLIC_REVISION = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
METRICS = ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")
DEPTHS = (24, 50, 100)
PIT = ("historical_proxy", "verified_known_at")


@dataclass(frozen=True)
class Ranking:
    candidate_k: int = 24
    rrf_k: int = 60
    dense_weight: float = .5
    per_page: int = 0

    @property
    def label(self):
        if self == BASE:
            return "E0"
        return f"k{self.candidate_k}-r{self.rrf_k}-d{self.dense_weight:.2f}-p{self.per_page}"


BASE = Ranking()
GRID = tuple(Ranking(*values) for values in product(DEPTHS, (10, 30, 60), (.5, .65, .8), (0, 1, 2)))
RULES = {
    "grid": [asdict(c) for c in GRID],
    "selection": ["recall@5", "recall@1", "mrr@10", "ndcg@10", "recall@10",
                  "smaller candidate pool", "baseline tie preference", "configuration label"],
    "minimum_calibration_recall_at_5_gain": 0,
    "allowed_calibration_loss": {"recall@1": .03, "recall@10": .03, "mrr@10": .03},
    "hot_latency_ratio_limit": 2,
    "chunk_trigger": "At least 20% of identifiable numeric lines absent from every legacy chunk, minimum 10 pages.",
    "cold_definition": "First query-cache pass, clear company-scope cache before each query. OS file cache uncontrolled.",
    "hot_definition": "Repeat after warming all query vectors, order by company to reuse legal scope.",
    "bootstrap": {"seed": 47185, "repetitions": 10000, "unit": "company", "statistic": "question-weighted paired mean"},
}


def read_rows(path):
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def load_split(source, group):
    """Only retain question and scoring labels for the requested company split."""
    dataset = source / "data/financebench/data/financebench_open_source.jsonl"
    historical = HISTORICAL / "public-four-mode-queries.jsonl"
    with historical.open() as file:
        old = {r["id"]: r for r in map(json.loads, file) if r["split"] == group}
    questions = {}
    with dataset.open() as file:
        for line in file:
            q = json.loads(line)
            if split(q["company"]) == group:
                questions[q["financebench_id"]] = q
    expected = (51, 49) if group == "calibration" else (99, 99)
    if set(old) != set(questions) or len(old) != expected[0]:
        raise ValueError("The official and historical split identities differ")
    selected = []
    for qid, previous in old.items():
        if previous["exclusions"]:
            continue
        q = questions[qid]
        pages = {tuple(p) for p in previous["anchors"]}
        if not pages or pages != {(e["doc_name"], e["evidence_page_num"]) for e in q["evidence"]}:
            raise ValueError("Historical and official page labels differ")
        selected.append({"id": qid, "company": q["company"], "query": q["question"],
                         "gold": pages, "historical": previous["modes"]["R3"]["scores"]["page_proxy"],
                         "calibration_evidence": q["evidence"] if group == "calibration" else []})
    if len(selected) != expected[1]:
        raise ValueError("Common evaluation denominator changed")
    return selected


def page_key(row):
    return row["document_id"], row["page"]


def fuse(lexical, dense, metadata, config):
    """Rank chunks. Apply page quotas during selection while retaining chunk ranks."""
    lexical, dense = lexical[:config.candidate_k], dense[:config.candidate_k]
    if config.dense_weight == .5:
        ranked = rrf([lexical, dense], k=config.rrf_k)
    else:
        scores = defaultdict(float)
        for ranking, weight in ((lexical, 1 - config.dense_weight), (dense, config.dense_weight)):
            for rank, (key, _) in enumerate(ranking, 1):
                scores[key] += weight / (config.rrf_k + rank)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    counts = Counter()
    result = []
    for key, _ in ranked:
        page = page_key(metadata[key])
        if not config.per_page or counts[page] < config.per_page:
            counts[page] += 1
            result.append(key)
            if len(result) == 10:
                break
    return result


def page_recall(keys, metadata, gold):
    return len({page_key(metadata[k]) for k in keys} & gold) / len(gold)


def score(lexical, dense, metadata, gold, config):
    keys = fuse(lexical, dense, metadata, config)
    pages = [page_key(metadata[key]) for key in keys]
    documents = list(dict.fromkeys(doc for doc, _ in pages))
    size = config.candidate_k
    candidates = [key for key, _ in lexical[:size] + dense[:size]]
    eligible = {page_key(row) for row in metadata.values()}
    return {
        "page": relevance(pages, gold),
        "document": relevance(documents, {doc for doc, _ in gold}),
        "candidate_page_recall": page_recall(candidates, metadata, gold),
        "indexed_gold_recall": len(gold & eligible) / len(gold),
        "unique_pages@5": len(set(pages[:5])),
        "returned_chunks@5": min(5, len(pages)),
    }


def retrieve(retriever, q, depth):
    rows, filters = retriever._candidates(q["company"], CUTOFF, PARTITION, PIT, False)
    start = perf_counter()
    lexical = bm25(q["query"], list(rows.values()), depth)
    bm25_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    dense = ([(str(p.id), float(p.score)) for p in retriever.store.search(q["query"], filters, depth)]
             if rows else [])
    return rows, lexical, dense, (bm25_ms, (perf_counter() - start) * 1000)


def cached_score(row, config):
    return score(row["bm25"], row["dense"], row["metadata"], {tuple(k) for k in row["gold"]}, config)


def collect(retriever, questions, output, audit=None):
    with output.open("x") as stream:
        for i, q in enumerate(questions, 1):
            metadata, lexical, dense, timing = retrieve(retriever, q, max(DEPTHS))
            seen = {key for key, _ in lexical + dense}
            # Keep page metadata for candidates and gold existence, not source text.
            seen.update(key for key, row in metadata.items() if page_key(row) in q["gold"])
            row = {"id": q["id"], "bm25": lexical, "dense": dense, "gold": sorted(q["gold"]),
                   "metadata": {k: {f: metadata[k][f] for f in ("document_id", "page")} for k in sorted(seen)},
                   "timing": {"bm25_ms_at_100": timing[0], "dense_ms_at_100": timing[1]}}
            if audit is not None:
                row["diagnostics"] = audit(q, metadata, lexical, dense)
            stream.write(json.dumps(row) + "\n")
            if i % 10 == 0:
                print(f"Calibration cached {i}/{len(questions)}", flush=True)


def average(values):
    return sum(values) / len(values)


def aggregate(rows):
    result = {scope: {metric: average([r[scope][metric] for r in rows]) for metric in METRICS}
              for scope in ("page", "document")}
    result.update({key: average([r[key] for r in rows]) for key in
                   ("candidate_page_recall", "indexed_gold_recall", "unique_pages@5", "returned_chunks@5")})
    return {"n": len(rows), **result}


def selection_key(result, config):
    p = result["page"]
    return (-p["recall@5"], -p["recall@1"], -p["mrr@10"], -p["ndcg@10"], -p["recall@10"],
            config.candidate_k, config != BASE, config.label)


def quality_ok(result, baseline):
    return (result["page"]["recall@5"] > baseline["page"]["recall@5"]
            and all(result["page"][k] >= baseline["page"][k] - loss
                    for k, loss in RULES["allowed_calibration_loss"].items()))


def percentiles(values):
    ordered = sorted(values)
    def quantile(p):
        pos = p * (len(ordered) - 1)
        i = int(pos)
        return ordered[i] + (ordered[min(i + 1, len(ordered) - 1)] - ordered[i]) * (pos - i)
    return {"mean": average(ordered), "p50": quantile(.5), "p95": quantile(.95)}


def bootstrap(results, baseline, repetitions=10000, seed=47185):
    if [r["id"] for r in results] != [r["id"] for r in baseline]:
        raise ValueError("Paired question sets do not match")
    grouped = defaultdict(list)
    differences = Counter()
    for a, b in zip(results, baseline, strict=True):
        if a["company_group"] != b["company_group"]:
            raise ValueError("Paired company groups do not match")
        delta = a["page"]["recall@5"] - b["page"]["recall@5"]
        differences["better" if delta > 0 else "worse" if delta < 0 else "tied"] += 1
        grouped[a["company_group"]].append(delta)
    clusters = [(sum(deltas), len(deltas)) for deltas in grouped.values()]
    rng = Random(seed)
    draws = []
    for _ in range(repetitions):
        sample = rng.choices(clusters, k=len(clusters))
        draws.append(sum(total for total, _ in sample) / sum(n for _, n in sample))
    draws.sort()
    return {"mean_delta": sum(total for total, _ in clusters) / len(results), "companies": len(clusters),
            "ci95": [draws[int(.025 * repetitions)], draws[min(repetitions - 1, int(.975 * repetitions))]],
            "better": differences["better"], "worse": differences["worse"], "tied": differences["tied"]}


def tree_hashes(path):
    return {p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob("*"))
            if p.is_file() and p.name != ".lock"}


def index_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and p.name != ".lock")


def inputs(source):
    dataset = source / "data/financebench"
    historical = read_json(HISTORICAL / "public-four-mode-summary.json")
    encoder_dir = source / "data/local-models/models" / LocalEmbedding.REVISION
    source_index = source / "data/local-models" / BASE_INDEX
    pdfs = {name: sha256(source / "data/financebench-pdfs" / f"{name}.pdf") for name in historical["pdfs"]}
    for name, info in historical["pdfs"].items():
        if info["status"] == "indexed" and pdfs[name] != info["sha256"]:
            raise ValueError("Historical PDF hash mismatch")
    data_hashes = {name: sha256(dataset / "data" / name) for name in
                   ("financebench_open_source.jsonl", "financebench_document_information.jsonl")}
    revision = subprocess.check_output(["git", "-C", str(dataset), "rev-parse", "HEAD"], text=True).strip()
    if (revision != PUBLIC_REVISION
            or tuple(data_hashes.values()) != (historical["questions_sha256"], historical["documents_sha256"])):
        raise ValueError("Public dataset differs from its frozen revision")
    files = ("tradingagents_lite/retrieval_v2.py", "tradingagents_lite/retrieval_v2_chunks.py",
             "tradingagents_lite/rag.py", "tradingagents_lite/vectors.py",
             "tradingagents_lite/documents.py", "tradingagents_lite/rag_benchmark.py", "uv.lock", "pyproject.toml")
    return {
        "dataset_revision": revision, "dataset_sha256": data_hashes, "pdf_sha256": pdfs,
        "original_index": tree_hashes(source_index), "original_index_bytes": index_bytes(source_index),
        "historical_labels_and_e0_sha256": sha256(HISTORICAL / "public-four-mode-queries.jsonl"),
        "historical_summary_sha256": sha256(HISTORICAL / "public-four-mode-summary.json"),
        "embedding_weights": tree_hashes(encoder_dir),
        "code": {name: sha256(ROOT / name) for name in files},
        "runtime": {"python": platform.python_version(), **{name: version(name) for name in
                    ("pypdf", "onnxruntime", "qdrant-client", "numpy", "tokenizers")}},
        "scope": {"partition": PARTITION, "cutoff": CUTOFF.isoformat(), "pit": list(PIT), "lookback": False},
    }


def copy_index(source, destination):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".lock"))
    if tree_hashes(source) != tree_hashes(destination):
        raise ValueError("Local index copy differs from frozen source")


def model_for(source, run, name):
    model_root = run / "query-caches" / name
    model_root.mkdir(parents=True, exist_ok=False)
    (model_root / "models").symlink_to((source / "data/local-models/models").resolve(), target_is_directory=True)
    return LocalEmbedding(model_root, allow_download=False)


def measure(source, run, index, cfg, questions, name):
    encoder = model_for(source, run, name)
    results = {}
    with VectorStore(index, encoder, "chunks") as store:
        retriever = Retriever(store)
        for temperature in ("cold", "hot"):
            retriever._scopes.clear()
            for q in sorted(questions, key=lambda x: (x["company"], x["id"])):
                if temperature == "cold":
                    retriever._scopes.clear()
                began = perf_counter()
                metadata, lexical, dense, timings = retrieve(retriever, q, cfg.candidate_k)
                fuse_start = perf_counter()
                fuse(lexical, dense, metadata, cfg)
                ended = perf_counter()
                row = results.setdefault(q["id"], {
                    "id": q["id"], "company_group": digest(q["company"].encode()).hexdigest()[:16],
                    **score(lexical, dense, metadata, q["gold"], cfg),
                })
                row[temperature] = {"total_ms": (ended - began) * 1000, "bm25_ms": timings[0],
                                    "dense_ms": timings[1], "fusion_ms": (ended - fuse_start) * 1000}
            print(f"{name} {temperature}: {len(questions)} queries", flush=True)
    ordered = [results[q["id"]] for q in questions]
    return ordered, {**aggregate(ordered), "latency": {
        temp: {metric: percentiles([row[temp][metric] for row in ordered])
               for metric in ("total_ms", "bm25_ms", "dense_ms", "fusion_ms")}
        for temp in ("cold", "hot")}}


def calibrate(source, run):
    from .retrieval_v2_chunks import Audit, run_chunk_stage

    run.mkdir(parents=True, exist_ok=False)
    provenance = inputs(source)
    historical = read_json(HISTORICAL / "public-four-mode-summary.json")
    index = run / "original-index"
    copy_index(source / "data/local-models" / BASE_INDEX, index)
    questions = load_split(source, "calibration")
    audit = Audit(source, historical)
    encoder = model_for(source, run, "calibration-cache")
    with VectorStore(index, encoder, "chunks") as store:
        collect(Retriever(store), questions, run / "candidate-cache.jsonl", audit)
    cached = read_rows(run / "candidate-cache.jsonl")
    baseline_rows = [cached_score(row, BASE) for row in cached]
    baseline = aggregate(baseline_rows)
    for actual, q in zip(baseline_rows, questions, strict=True):
        # Qdrant resolves equal distances at the cutoff before RRF. The exact-k
        # timing passes below also check this on the original endpoint.
        if actual["page"] != q["historical"]:
            print("A historical page metric differs in the depth-100 cache", flush=True)
            break
    grid = [(cfg, aggregate([cached_score(row, cfg) for row in cached])) for cfg in GRID]
    grid.sort(key=lambda x: selection_key(x[1], x[0]))
    finalists = [(cfg, stats) for cfg, stats in grid if quality_ok(stats, baseline)][:2]
    base_rows, measured_base = measure(source, run, index, BASE, questions, "cal-e0")
    measured = []
    for cfg, _ in finalists:
        _, result = measure(source, run, index, cfg, questions, f"cal-{cfg.label}")
        measured.append((cfg, result))
    accepted = [(cfg, result) for cfg, result in measured if quality_ok(result, measured_base)
                and result["latency"]["hot"]["total_ms"]["mean"] <=
                RULES["hot_latency_ratio_limit"] * measured_base["latency"]["hot"]["total_ms"]["mean"]]
    best, best_metrics = min(accepted, key=lambda x: selection_key(x[1], x[0])) if accepted else (BASE, measured_base)
    diagnostics = [row["diagnostics"] for row in cached]
    by_id = {q["id"]: q for q in questions}
    for row in cached:
        d = row["diagnostics"]
        lines = d.get("lines", [])
        for cfg in dict.fromkeys((BASE, best)):
            ids = set(fuse(row["bm25"], row["dense"], row["metadata"], cfg)[:5])
            d[f"observable_lines_at_5__{cfg.label}"] = sum(bool(ids & set(line["chunk_ids"])) for line in lines)
        d.pop("lines", None)
    chunk_stage = run_chunk_stage(source, run, historical, questions, best, best_metrics, diagnostics)
    clean_audit = [{"id": row["id"], **row["diagnostics"], "baseline_historical_match":
                    actual["page"] == by_id[row["id"]]["historical"]}
                   for row, actual in zip(cached, base_rows, strict=True)]
    write_rows(run / "calibration-audit.jsonl", clean_audit)
    calibration = {
        "n": len(questions), "full_split_n": 51, "historical_exact_e0_matches":
        sum(row["baseline_historical_match"] for row in clean_audit),
        "grid": [{"config": asdict(cfg), "label": cfg.label, **row} for cfg, row in grid],
        "measured": [{"config": asdict(cfg), "label": cfg.label, **row}
                     for cfg, row in [(BASE, measured_base), *measured]],
        "stage1_best": {"config": asdict(best), "label": best.label, **best_metrics},
        "chunk_stage": chunk_stage, "glm_requests": 0,
    }
    write_json(run / "calibration.json", calibration)
    freeze = {"inputs": provenance, "rules": RULES, "baseline": asdict(BASE), "stage1": asdict(best),
              "final": asdict(best), "final_chunks": chunk_stage["selected"],
              "calibration_sha256": sha256(run / "calibration.json"),
              "calibration_audit_sha256": sha256(run / "calibration-audit.jsonl"),
              "original_index_copy": tree_hashes(index),
              "test_scope": {"n": 99, "label_source": "historical R3 public page anchors"},
              "embedding": encoder.metadata, "reranker": None}
    write_json(run / "freeze.json", freeze)
    print(f"Frozen: {sha256(run / 'freeze.json')} ({best.label}, {freeze['final_chunks']})", flush=True)


def formal(source, run):
    from .retrieval_v2_chunks import build_variant

    freeze = read_json(run / "freeze.json")
    index = run / "original-index"
    if (freeze["inputs"] != inputs(source) or freeze["original_index_copy"] != tree_hashes(index)
            or freeze["calibration_sha256"] != sha256(run / "calibration.json")
            or freeze["calibration_audit_sha256"] != sha256(run / "calibration-audit.jsonl")
            or freeze["rules"] != RULES):
        raise ValueError("Frozen inputs, source, or calibration changed")
    test_output = run / "test"
    test_output.mkdir()
    questions = load_split(source, "test")
    plans = [("E0", BASE, "legacy"), ("stage1", Ranking(**freeze["stage1"]), "legacy"),
             ("final", Ranking(**freeze["final"]), freeze["final_chunks"])]
    recorded, groups, aliases = {}, {}, {}
    for label, cfg, chunks in plans:
        key = (cfg, chunks)
        if key in aliases:
            groups[label] = {"alias_of": aliases[key]}
            continue
        aliases[key] = label
        build = None
        if chunks != "legacy":
            historical = read_json(HISTORICAL / "public-four-mode-summary.json")
            target, build = build_variant(source, run, historical, chunks, companies=None)
        else:
            target = index
        rows, stats = measure(source, run, target, cfg, questions, "test-" + label)
        recorded[label] = rows
        stats["config"] = asdict(cfg)
        stats["chunks"] = chunks
        stats["index_bytes"] = index_bytes(target)
        stats["build"] = build
        stats["relative_to_e0"] = bootstrap(rows, recorded["E0"])
        for row, old in zip(rows, recorded["E0"], strict=True):
            row["recall_at_5_delta"] = row["page"]["recall@5"] - old["page"]["recall@5"]
        groups[label] = stats
    for label, rows in recorded.items():
        write_rows(test_output / f"{label}.jsonl", rows)
    result = {"groups": groups, "n": len(questions), "glm_requests": 0,
              "original_e0_matching_questions": sum(r["page"] == q["historical"] for r, q in
                                                   zip(recorded["E0"], questions, strict=True)),
              "freeze_sha256": sha256(run / "freeze.json"),
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(test_output / "summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "test"))
    parser.add_argument("--source", type=Path, required=True, help="Original checkout with frozen local data")
    parser.add_argument("--run", type=Path, required=True, help="A fresh output directory for calibration")
    args = parser.parse_args()
    action = calibrate if args.command == "calibrate" else formal
    action(args.source.resolve(), args.run.resolve())


if __name__ == "__main__":
    main()
