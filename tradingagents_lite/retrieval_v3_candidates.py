"""Shared retrieval and timing for A1-V3. Query paths contain no scoring labels."""
from pathlib import Path
from time import perf_counter

from .data import sha256
from .rag import Retriever, bm25
from .rag_benchmark import CUTOFF, PARTITION, split
from .retrieval_v2 import (
    BASE_INDEX, PIT, ROOT, copy_index, inputs, model_for, read_json, read_rows, tree_hashes, write_json,
)
from .vectors import VectorStore

TEMPERATURES = ("cold", "hot")
DEPTHS = (24, 50, 100)
V2_ARCHIVE = ROOT / "results/a1-retrieval-v2"


def public_questions(source, group, calibration_ids=None):
    file = source / "data/financebench/data/financebench_open_source.jsonl"
    result = {
        obj["financebench_id"]: {"id": obj["financebench_id"], "company": obj["company"], "query": obj["question"]}
        for obj in read_rows(file) if split(obj["company"]) == group
    }
    if group == "test":
        if len(result) != 99:
            raise ValueError("Unexpected Test denominator")
        return list(result.values())
    if len(result) != 51 or len(calibration_ids) != 49 or len(set(calibration_ids)) != 49:
        raise ValueError("Unexpected Calibration partition")
    return [result[key] for key in calibration_ids]


def pool(lexical, dense, depth):
    return sorted({key for key, _ in lexical[:depth]} | {key for key, _ in dense[:depth]})


def rank_ids(ranking):
    return {arm: [key for key, _ in values] for arm, values in ranking.items()}


def _time_retrieval(retriever, questions, depths):
    store = retriever.store
    measurements = {}
    for temperature in TEMPERATURES:
        retriever._scopes.clear()
        for i, q in enumerate(sorted(questions, key=lambda item: (item["company"], item["id"])), 1):
            if temperature == "cold":
                retriever._scopes.clear()
            began = perf_counter()
            rows, filters = retriever._candidates(q["company"], CUTOFF, PARTITION, PIT, False)
            scoped = perf_counter()
            lex = bm25(q["query"], list(rows.values()), max(depths))
            lexical_done = perf_counter()
            query_vector = store.encoder.encode([q["query"]], query=True)[0] if rows else None
            encoded = perf_counter()
            rankings, costs = {}, {}
            for depth in depths:
                d0 = perf_counter()
                dense = store.client.query_points(store.name, query=query_vector, query_filter=filters,
                                                  limit=depth, with_payload=True).points if rows else []
                d1 = perf_counter()
                rankings[str(depth)] = {"bm25": lex[:depth],
                                        "dense": [(str(p.id), float(p.score)) for p in dense]}
                costs[str(depth)] = {
                    "scope_ms": (scoped - began) * 1000,
                    "bm25_ms": (lexical_done - scoped) * 1000,
                    "encode_ms": (encoded - lexical_done) * 1000,
                    "dense_ms": (d1 - d0) * 1000,
                }
            cold = measurements.setdefault(q["id"], {"rankings": rankings, "timing": {}})
            for depth in depths:
                k = str(depth)
                costs[k]["id_order_changed"] = rank_ids(rankings[k]) != rank_ids(cold["rankings"][k])
            cold["timing"][temperature] = costs
            if i % 20 == 0 or i == len(questions):
                print(f"{temperature} CPU retrieval: {i}/{len(questions)}", flush=True)
    return measurements


def calibration_candidates(source, old_cache, run):
    run.mkdir(parents=True, exist_ok=True)
    archive = V2_ARCHIVE / "freeze.json"
    old_freeze = read_json(archive)
    if old_freeze["inputs"] != inputs(source):
        raise ValueError("Original corpus no longer matches A1-V2")
    cached = read_rows(old_cache)
    ids = [r["id"] for r in cached]
    prior_ids = [r["id"] for r in read_rows(V2_ARCHIVE / "calibration-audit.jsonl")]
    if ids != prior_ids:
        raise ValueError("Candidate cache does not cover the frozen 49 Calibration questions")
    copy = run / "original-index"
    if not copy.exists():
        copy_index(source / "data/local-models" / BASE_INDEX, copy)
    if tree_hashes(copy) != old_freeze["original_index_copy"]:
        raise ValueError("Copied index is not the A1-V2 frozen index")
    encoder = model_for(source, run, "calibration-cpu")
    questions = public_questions(source, "calibration", ids)
    with VectorStore(copy, encoder, "chunks") as store:
        measurements = _time_retrieval(Retriever(store), questions, DEPTHS)
    parity_by_depth = {str(depth): 0 for depth in DEPTHS}
    for old in cached:
        for depth in DEPTHS:
            k = str(depth)
            prior = {arm: old[arm][:depth] for arm in ("bm25", "dense")}
            fresh = measurements[old["id"]]["rankings"][k]
            parity_by_depth[k] += rank_ids(fresh) == rank_ids(prior)
            # Use the V2 candidates for inference. The independent pass times retrieval.
            measurements[old["id"]]["rankings"][k] = prior
    write_json(run / "calibration-candidates.json", {
        "a1_v2_freeze_sha256": sha256(archive), "a1_v2_cache_sha256": sha256(old_cache),
        "question_ids": ids, "measurements": measurements, "candidate_parity_by_depth": parity_by_depth,
    })


def test_candidates(source, run, depths):
    index = run / "original-index"
    encoder = model_for(source, run, "test-cpu")
    questions = public_questions(source, "test")
    with VectorStore(index, encoder, "chunks") as store:
        result = _time_retrieval(Retriever(store), questions, depths)
    write_json(run / "test-candidates.json", {"question_ids": [q["id"] for q in questions], "measurements": result})

