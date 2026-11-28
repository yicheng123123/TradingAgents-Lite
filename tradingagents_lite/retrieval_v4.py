"""A1-V4 BGE-M3 retrieval ablation. Offline and isolated from historical indexes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import statistics
import struct
import time
from random import Random
from time import perf_counter

from qdrant_client import QdrantClient, models

from .data import sha256 as file_sha256
from .embedding_v4 import BGEM3Encoder, DIMENSION, MODEL_ID, MODEL_REVISION
from .rag import Retriever, bm25, rrf
from .rag_benchmark import CUTOFF, PARTITION, relevance, split
from .reranker_v3 import BGEReranker
from .retrieval_v2 import index_bytes, read_json, read_rows, tree_hashes
from .retrieval_v2_chunks import Audit, normalize, numeric_line
from .vectors import LocalEmbedding, VectorStore

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/gaoyi/projects/tradingagents-lite")
V3_ROOT = Path("/home/gaoyi/projects/tradingagents-a1-retrieval-v3")
V3_RUN = V3_ROOT / "data/a1-retrieval-v3/002"
V3_ARCHIVE = V3_ROOT / "results/a1-retrieval-v3"
V2_ARCHIVE = V3_ROOT / "results/a1-retrieval-v2"
RUN = ROOT / "data/a1-retrieval-v4/001"
RESULTS = ROOT / "results/a1-retrieval-v4"
MODEL_ROOT = RUN / "models/baai-bge-m3"
INDEX_ROOT = RUN / "bge-m3-index"
COLLECTION = "chunks_bge_m3_v4"
RERANKER_ROOT = V3_ROOT / "data/a1-retrieval-v3/bge-reranker-v2-m3"
DEPTHS = (1, 5, 10, 24, 50, 100)
CANDIDATE_DEPTHS = (24, 50, 100)
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RERANKER_SHA256 = "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286"
SOURCE_CODE = (
    "tradingagents_lite/embedding_v4.py", "tradingagents_lite/retrieval_v4.py",
    "tradingagents_lite/rag.py", "tradingagents_lite/rag_benchmark.py",
    "tradingagents_lite/retrieval_v2_chunks.py", "tradingagents_lite/reranker_v3.py",
    "tradingagents_lite/vectors.py",
)
BOOTSTRAP = {"unit": "company", "repetitions": 10000, "seed": 47185,
             "statistic": "question-weighted paired mean page Recall@5"}


def json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, value, *, overwrite: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def jsonl_write(path: Path, rows, *, overwrite: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def jsonl_read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_tree_hashes(path: Path):
    return {item.relative_to(path).as_posix(): file_sha256(item) for item in sorted(path.rglob("*"))
            if item.is_file() and item.name != ".lock" and ".cache" not in item.parts}


def digest_obj(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def query_hash(qid: str) -> str:
    return sha256(qid.encode()).hexdigest()[:24]


def company_hash(company: str) -> str:
    # Match V3's company-cluster pseudonymization exactly.
    return sha256(repr((company,)).encode()).hexdigest()[:24]


def page_hash(page) -> str:
    return sha256(repr(tuple(page)).encode()).hexdigest()[:24]


def calibration_questions(source: Path):
    dataset = source / "data/financebench/data/financebench_open_source.jsonl"
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    calibration_ids = json_read(V3_RUN / "calibration-candidates.json")["question_ids"]
    audit_ids = [row["id"] for row in jsonl_read(V2_ARCHIVE / "calibration-audit.jsonl")]
    if calibration_ids != audit_ids or len(calibration_ids) != 49 or len(set(calibration_ids)) != 49:
        raise ValueError("Calibration identities do not match the frozen A1-V2 common set")
    by_id = {row["financebench_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate FinanceBench question IDs")
    calibration = []
    for qid in calibration_ids:
        raw = by_id[qid]
        if split(raw["company"]) != "calibration":
            raise ValueError("Calibration question outside the frozen company split")
        calibration.append(question(raw))
    if len(calibration) != 49:
        raise ValueError("Unexpected Calibration denominator")
    return calibration


def test_questions_after_freeze(source: Path):
    dataset = source / "data/financebench/data/financebench_open_source.jsonl"
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = [question(row) for row in rows if split(row["company"]) == "test"]
    if len(result) != 99:
        raise ValueError("Unexpected public Test denominator")
    return result


def question(raw):
    gold = sorted({(entry["doc_name"], entry["evidence_page_num"]) for entry in raw["evidence"]})
    if not gold:
        raise ValueError("Public FinanceBench question has no page anchors")
    return {"id": raw["financebench_id"], "company": raw["company"],
            "query": raw["question"], "gold": set(map(tuple, gold)),
            "evidence": raw["evidence"]}


def scroll_points(client: QdrantClient, collection: str, *, with_vectors: bool = False):
    offset, result = None, []
    while True:
        points, offset = client.scroll(collection, offset=offset, limit=256,
                                       with_payload=True, with_vectors=with_vectors)
        result.extend(points)
        if offset is None:
            return result


def original_index_rows(path: Path):
    with closing(QdrantClient(path=str(path))) as client:
        if not client.collection_exists("chunks"):
            raise ValueError("Frozen V3 source index lacks chunks collection")
        points = scroll_points(client, "chunks")
    rows = [{"point_id": str(point.id), **point.payload} for point in points]
    rows.sort(key=lambda row: row["point_id"])
    if len(rows) != 36622 or len({row["chunk_id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected 36,622 unique original chunks, found {len(rows)}")
    if any(row["point_id"] != row["chunk_id"] for row in rows):
        raise ValueError("Qdrant point ID and original Chunk ID differ")
    docs = {row["document_id"] for row in rows}
    pdfs = list((SOURCE / "data/financebench-pdfs").glob("*.pdf"))
    frozen_pdf_ids = set(json_read(V3_RUN / "freeze.json")["inputs"]["A1_V2_inputs"]["pdf_sha256"])
    pdf_ids = {p.stem for p in pdfs}
    if len(docs) != 83 or docs - frozen_pdf_ids or pdf_ids != frozen_pdf_ids or not docs <= pdf_ids:
        raise ValueError("The frozen corpus does not match the 83 source PDFs")
    required = {"chunk_id", "document_id", "page", "text", "ticker", "section", "pit_level",
                "eligible_ts", "eligible_at", "dataset_partition", "version_id"}
    if any(not required <= row.keys() for row in rows):
        raise ValueError("An original chunk is missing required retrieval metadata")
    return rows


def source_fingerprint(rows):
    digestor = sha256()
    for row in rows:
        payload = {key: value for key, value in row.items() if key != "point_id"}
        digestor.update(json.dumps([row["point_id"], payload], sort_keys=True,
                                  ensure_ascii=False, separators=(",", ":")).encode())
        digestor.update(b"\n")
    return digestor.hexdigest()


def fetch_model(model_root: Path):
    model_root.parent.mkdir(parents=True, exist_ok=True)
    weight_file = model_root / "pytorch_model.bin"
    if not weight_file.is_file():
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(model_root),
                          allow_patterns=["*.json", "pytorch_model.bin", "*.model", "*.txt"])
    weights = file_tree_hashes(model_root)
    if "pytorch_model.bin" not in weights:
        raise ValueError("Pinned BGE-M3 PyTorch checkpoint is missing")
    return weights


def token_audit(encoder: BGEM3Encoder, rows):
    lengths = []
    texts = [row["text"] for row in rows]
    for start in range(0, len(texts), 128):
        lengths.extend(encoder.token_lengths(texts[start:start + 128]))
        if (start + 128) % 2048 == 0 or start + 128 >= len(texts):
            print(f"Token audit: {min(start + 128, len(texts))}/{len(texts)}", flush=True)
    ordered = sorted(lengths)
    at = lambda p: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]
    return {
        "chunks": len(lengths), "max_length": MAX_AUDIT_LENGTH,
        "truncated_at_512": sum(value > 512 for value in lengths),
        "truncated_at_1024": sum(value > 1024 for value in lengths),
        "truncation_rate_at_512": sum(value > 512 for value in lengths) / len(lengths),
        "truncation_rate_at_1024": sum(value > 1024 for value in lengths) / len(lengths),
        "mean": statistics.fmean(lengths), "p50": at(.5), "p90": at(.9),
        "p95": at(.95), "p99": at(.99), "max": ordered[-1],
    }


MAX_AUDIT_LENGTH = 8192


def smoke(encoder: BGEM3Encoder, rows):
    texts = [row["text"] for row in rows[:8]]
    if len(texts) < 2:
        raise ValueError("BGE-M3 smoke requires several real source chunks")
    batch, stats = encoder.encode_with_stats(texts)
    single = encoder.encode([texts[0]])[0]
    repeated = encoder.encode([texts[0]])[0]
    max_single_batch_delta = max(abs(a - b) for a, b in zip(single, batch[0], strict=True))
    max_repeat_delta = max(abs(a - b) for a, b in zip(single, repeated, strict=True))
    cosine = [sum(a * b for a, b in zip(single, vector, strict=True)) for vector in batch]
    if max_single_batch_delta > 5e-4 or max_repeat_delta > 2e-5:
        raise ValueError("BGE-M3 encoding is not stable between repeated or batched runs")
    if max(cosine) < .999 or cosine.index(max(cosine)) != 0:
        raise ValueError("BGE-M3 smoke retrieval did not rank the exact source chunk first")
    return {
        "real_chunks": len(texts), "dimension": len(batch[0]), "finite": True,
        "l2_norms": [math.sqrt(sum(v * v for v in vector)) for vector in batch],
        "repeat_max_abs_delta": max_repeat_delta,
        "single_batch_max_abs_delta": max_single_batch_delta,
        "exact_chunk_retrieval_top1": 0,
        "self_similarity": cosine[0],
        "milliseconds": stats["milliseconds"],
        "chunks_per_second": len(texts) * 1000 / max(stats["milliseconds"], 1e-9),
        "peak_allocated_bytes": stats["peak_allocated_bytes"],
        "peak_reserved_bytes": stats["peak_reserved_bytes"],
        "oom_batch_splits": stats["oom_batch_splits"],
    }


def create_independent_index(rows, encoder: BGEM3Encoder, index_root: Path, source_hash: str):
    if index_root.exists():
        raise FileExistsError(f"V4 index path already exists; refusing to overwrite: {index_root}")
    index_root.mkdir(parents=True)
    with closing(QdrantClient(path=str(index_root))) as client:
        client.create_collection(COLLECTION, vectors_config=models.VectorParams(
            size=DIMENSION, distance=models.Distance.COSINE))
        encoded_count, peak_allocated, peak_reserved, oom_splits = 0, 0, 0, 0
        token_count = 0
        started = perf_counter()
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            vectors, stats = encoder.encode_with_stats([row["text"] for row in batch])
            client.upsert(COLLECTION, [models.PointStruct(
                id=row["point_id"], vector=vector,
                payload={key: value for key, value in row.items() if key != "point_id"},
            ) for row, vector in zip(batch, vectors, strict=True)], wait=True)
            encoded_count += len(vectors)
            token_count += stats["tokens"]
            peak_allocated = max(peak_allocated, stats["peak_allocated_bytes"] or 0)
            peak_reserved = max(peak_reserved, stats["peak_reserved_bytes"] or 0)
            oom_splits += stats["oom_batch_splits"]
            if encoded_count % 1024 < len(batch) or encoded_count == len(rows):
                print(f"BGE-M3 index: {encoded_count}/{len(rows)} chunks", flush=True)
        elapsed = perf_counter() - started
        count = client.count(COLLECTION, exact=True).count
        if count != len(rows) or encoded_count != len(rows):
            raise ValueError(f"Incomplete BGE-M3 index: encoded={encoded_count}, stored={count}")
        output_points = scroll_points(client, COLLECTION, with_vectors=True)
        output_points.sort(key=lambda point: str(point.id))
        payload_rows = [{"point_id": str(point.id), **point.payload} for point in output_points]
        payload_hash = source_fingerprint(payload_rows)
        vector_hash = sha256()
        for point in output_points:
            vector = point.vector
            if isinstance(vector, dict):
                vector = vector.get("dense")
            if vector is None or len(vector) != DIMENSION or any(not math.isfinite(v) for v in vector):
                raise ValueError("Invalid vector found during Qdrant index integrity audit")
            norm = math.sqrt(sum(v * v for v in vector))
            if abs(norm - 1.0) > 2e-4:
                raise ValueError("Stored vector failed L2 normalization audit")
            vector_hash.update(str(point.id).encode())
            vector_hash.update(struct.pack(f"<{DIMENSION}f", *vector))
        if payload_hash != source_hash:
            raise ValueError("Original chunk payload or Chunk ID changed while rebuilding V4 index")
        vector_hash_before_close = vector_hash.hexdigest()
        result = {
            "collection": COLLECTION, "dimensions": DIMENSION, "distance": "Cosine",
            "documents": len({row["document_id"] for row in rows}), "chunks": count,
            "encoded_chunks": encoded_count, "encoding_failures": 0,
            "original_chunk_payload_sha256": source_hash,
            "indexed_chunk_payload_sha256": payload_hash,
            "indexed_vector_sha256": vector_hash_before_close,
            "vector_sha256_before_qdrant_close": vector_hash_before_close,
            "chunk_ids_and_text_metadata_match": True,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "oom_batch_splits": oom_splits,
            "tokens_encoded": token_count,
            "chunks_per_second": encoded_count / elapsed,
            "build_seconds": elapsed,
            "index_bytes": index_bytes(index_root),
        }
    persisted = index_semantic_fingerprint(index_root, expected_chunks=len(rows))
    if persisted["payload_sha256"] != source_hash:
        raise ValueError("Persisted Qdrant payload differs from original chunks")
    result["indexed_vector_sha256"] = persisted["vector_sha256"]
    result["vector_hash_changed_on_reopen"] = result["vector_sha256_before_qdrant_close"] != persisted["vector_sha256"]
    result["index_bytes"] = index_bytes(index_root)
    json_write(index_root / "v4-index-manifest.json", result)
    return result


def exact_cosine_audit(index_root: Path, encoder: BGEM3Encoder, rows_by_id: dict, questions):
    """Compare Qdrant top-100 with exact dot-product top-100 on sampled Calibration scopes."""
    import numpy as np

    chosen = sorted(questions, key=lambda q: (q["company"], q["id"]))[:10]
    audits = []
    with closing(QdrantClient(path=str(index_root))) as client:
        retriever = Retriever(_StoreAdapter(client, encoder))
        for q in chosen:
            scoped, filters = retriever._candidates(q["company"], CUTOFF, PARTITION,
                                                      ("historical_proxy", "verified_known_at"), False)
            query = encoder.encode([q["query"]], query=True)[0]
            points = client.retrieve(COLLECTION, ids=list(scoped), with_payload=False, with_vectors=True)
            points.sort(key=lambda point: str(point.id))
            matrix = np.asarray([point.vector for point in points], dtype=np.float32)
            scores = matrix @ np.asarray(query, dtype=np.float32)
            exact = [str(points[i].id) for i in np.argsort(-scores, kind="stable")[:100]]
            approx = [str(p.id) for p in client.query_points(
                COLLECTION, query=query, query_filter=filters, limit=100, with_payload=False).points]
            common = sum(key in set(exact) for key in approx)
            audits.append({"question_sha256": query_hash(q["id"]), "scope_size": len(scoped),
                           "top100_overlap": common / max(len(exact), 1),
                           "top10_overlap": len(set(exact[:10]) & set(approx[:10])) / min(10, len(exact)),
                           "top1_equal": bool(exact and approx and exact[0] == approx[0])})
    return {"sample_n": len(audits), "mean_top100_overlap": statistics.fmean(x["top100_overlap"] for x in audits),
            "mean_top10_overlap": statistics.fmean(x["top10_overlap"] for x in audits),
            "top1_equal_count": sum(x["top1_equal"] for x in audits), "per_query": audits}


class _StoreAdapter:
    """Minimal Retriever contract for exact Qdrant audits; it never writes to the collection."""
    def __init__(self, client, encoder):
        self.client, self.encoder = client, encoder
        self.name, self.revision = COLLECTION, 0

    def scan(self, filters=None):
        offset = None
        while True:
            points, offset = self.client.scroll(COLLECTION, scroll_filter=filters, limit=256,
                                                offset=offset, with_payload=True)
            yield from points
            if offset is None:
                return


def prepare(source: Path, run: Path, device: str, max_length: int):
    if (run / "prepare.json").exists():
        raise FileExistsError("V4 prepare record already exists; refusing to silently replace it")
    source = source.resolve()
    source_index = V3_RUN / "original-index"
    source_hashes = tree_hashes(source_index)
    v3_freeze = json_read(V3_RUN / "freeze.json")
    if source_hashes != v3_freeze["inputs"]["A1_V2_inputs"]["original_index"]:
        raise ValueError("Read-only V3 source index no longer matches its frozen hash")
    rows = original_index_rows(source_index)
    raw_hash = source_fingerprint(rows)
    weight_hashes = fetch_model(MODEL_ROOT)
    model_load_started = perf_counter()
    encoder = BGEM3Encoder(MODEL_ROOT, device=device, max_length=max_length)
    model_load_seconds = perf_counter() - model_load_started
    smoke_result = smoke(encoder, rows)
    lengths = token_audit(encoder, rows)
    index_manifest_path = INDEX_ROOT / "v4-index-manifest.json"
    if index_manifest_path.is_file():
        index_result = json_read(index_manifest_path)
        if index_result["original_chunk_payload_sha256"] != raw_hash:
            raise ValueError("Existing V4 index was built from different original chunks")
        persisted = index_semantic_fingerprint(INDEX_ROOT, expected_chunks=index_result["chunks"])
        if persisted["payload_sha256"] != raw_hash:
            raise ValueError("Existing V4 Qdrant payload differs from original chunks")
        if persisted["vector_sha256"] != index_result["indexed_vector_sha256"]:
            index_result["vector_sha256_before_qdrant_close"] = index_result["indexed_vector_sha256"]
            index_result["indexed_vector_sha256"] = persisted["vector_sha256"]
            index_result["vector_hash_changed_on_reopen"] = True
            json_write(index_manifest_path, index_result, overwrite=True)
    else:
        index_result = create_independent_index(rows, encoder, INDEX_ROOT, raw_hash)
    calibration = calibration_questions(source)
    exact = exact_cosine_audit(INDEX_ROOT, encoder, {r["chunk_id"]: r for r in rows}, calibration)
    del encoder
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    result = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "files_sha256": weight_hashes},
        "source_index": {"path": str(source_index), "tree_sha256": source_hashes,
                         "chunks": len(rows), "documents": len({r["document_id"] for r in rows}),
                         "payload_sha256": raw_hash},
        "encoder": {"device": device, "maximum_input_tokens": max_length,
                    "pooling": "CLS", "normalization": "L2", "query_prefix": None,
                    "model_load_seconds": model_load_seconds},
        "token_length_audit": lengths, "smoke": smoke_result,
        "index": index_result, "qdrant_exact_cosine_audit": exact,
        "bm25": "Unchanged A1-V3 bm25() over original chunk payload text",
        "source_pdfs_reparsed": 0,
    }
    json_write(run / "prepare.json", result)
    return result


def collect_candidates(source: Path, index_root: Path, output: Path, group: str,
                       questions, device: str, max_length: int):
    if output.exists():
        raise FileExistsError(f"Candidate archive already exists: {output}")
    encoder = BGEM3Encoder(MODEL_ROOT, device=device, max_length=max_length)
    results = {}
    with closing(QdrantClient(path=str(index_root))) as client:
        retriever = Retriever(_StoreAdapter(client, encoder))
        ordered = sorted(questions, key=lambda item: (item["company"], item["id"]))
        for temperature in ("cold", "hot"):
            retriever._scopes.clear()
            for count, q in enumerate(ordered, 1):
                if temperature == "cold":
                    retriever._scopes.clear()
                started = perf_counter()
                rows, filters = retriever._candidates(q["company"], CUTOFF, PARTITION,
                    ("historical_proxy", "verified_known_at"), False)
                scope_done = perf_counter()
                lexical = bm25(q["query"], list(rows.values()), max(CANDIDATE_DEPTHS))
                bm25_done = perf_counter()
                query_vector, encode_stats = encoder.encode_with_stats([q["query"]])
                encoded_done = perf_counter()
                by_depth, timing = {}, {}
                for depth in CANDIDATE_DEPTHS:
                    search_start = perf_counter()
                    found = client.query_points(COLLECTION, query=query_vector[0], query_filter=filters,
                        limit=depth, with_payload=True).points if rows else []
                    search_done = perf_counter()
                    dense = [(str(point.id), float(point.score)) for point in found]
                    by_depth[str(depth)] = dense
                    timing[str(depth)] = {
                        "scope_ms": (scope_done - started) * 1000,
                        "bm25_ms": (bm25_done - scope_done) * 1000,
                        "query_encode_ms": encode_stats["milliseconds"],
                        "qdrant_search_ms": (search_done - search_start) * 1000,
                        "retrieval_total_ms": ((scope_done - started) + (bm25_done - scope_done)
                                                + (encoded_done - bm25_done) + (search_done - search_start)) * 1000,
                        "peak_allocated_bytes": encode_stats["peak_allocated_bytes"],
                        "peak_reserved_bytes": encode_stats["peak_reserved_bytes"],
                        "oom_batch_splits": encode_stats["oom_batch_splits"],
                    }
                entry = results.setdefault(q["id"], {"rankings": {}, "timing": {}})
                for depth in CANDIDATE_DEPTHS:
                    entry["rankings"][str(depth)] = {
                        "bm25": lexical[:depth], "dense": by_depth["100"][:depth]}
                for depth in CANDIDATE_DEPTHS:
                    entry["timing"].setdefault(temperature, {})[str(depth)] = timing[str(depth)]
                if count % 10 == 0 or count == len(ordered):
                    print(f"BGE-M3 {group} {temperature}: {count}/{len(ordered)}", flush=True)
    ids = [q["id"] for q in questions]
    json_write(output, {"question_ids": ids, "measurements": results,
                        "model_revision": MODEL_REVISION, "collection": COLLECTION,
                        "depths": list(CANDIDATE_DEPTHS)})
    return results


def read_index_payloads(index_root: Path):
    with closing(QdrantClient(path=str(index_root))) as client:
        points = scroll_points(client, COLLECTION)
    result = {str(point.id): point.payload for point in points}
    if len(result) != 36622 or any(key != value["chunk_id"] for key, value in result.items()):
        raise ValueError("V4 index payload count or Chunk ID identity changed")
    return result


def small_model_candidates(group: str):
    path = V3_RUN / f"{group}-candidates.json"
    result = json_read(path)
    return result["measurements"], file_sha256(path)


def dense_top_k(rankings, depth):
    available = [int(key) for key, value in rankings.items()
                 if str(key).isdigit() and isinstance(value, dict) and "dense" in value]
    if not available or depth > max(available):
        return None
    source_depth = max(available)
    return rankings[str(source_depth)]["dense"][:depth]


def rank_candidates(questions, small, m3, chunks, reranker: BGEReranker | None, *, run_e4: bool):
    """Produce fixed E1/E2/E3 arms; rerank the entire deduplicated candidate union."""
    computed = {}
    rerank_usage = {"model_load_seconds": None, "warmup": None}
    by_id = {q["id"]: q for q in questions}
    if set(by_id) != set(m3) or set(by_id) != set(small):
        raise ValueError("Small/M3 candidate question identities differ")
    if reranker is not None:
        warm_query = "What was revenue in fiscal year 2023?"
        sample_text = next(iter(chunks.values()))["text"][:800]
        _, warmup = reranker.score(warm_query, [sample_text])
        rerank_usage["warmup"] = warmup

    for q in questions:
        qid = q["id"]
        sm, mm = small[qid]["rankings"], m3[qid]["rankings"]
        computed[qid] = {"arms": {}, "measurements": {"small": small[qid], "m3": mm}}
        for model_name, rankings in (("small", sm), ("m3", mm)):
            for depth in DEPTHS:
                selected = dense_top_k(rankings, depth)
                if selected is None:
                    continue
                computed[qid]["arms"][f"E1-{model_name}@{depth}"] = {
                    "ranked": [key for key, _ in selected], "before": [key for key, _ in selected],
                    "candidate": [key for key, _ in selected],
                    "latency": _latency_record(small[qid] if model_name == "small" else m3[qid], depth),
                }
        for model_name, rankings in (("small", sm), ("m3", mm)):
            lexical, dense = rankings["50"]["bm25"], rankings["50"]["dense"]
            fusion_start = perf_counter()
            fused = rrf([lexical, dense], k=60)
            fusion_ms = (perf_counter() - fusion_start) * 1000
            candidate = list(dict.fromkeys([key for key, _ in lexical] + [key for key, _ in dense]))
            computed[qid]["arms"][f"E2-{model_name}"] = {
                "ranked": [key for key, _ in fused], "before": [key for key, _ in fused],
                "candidate": candidate,
                "latency": _latency_record(small[qid] if model_name == "small" else m3[qid], 50,
                                           ranking_ms=fusion_ms),
            }
        if reranker is not None:
            for arm, depth in (("E3-m3", 50), *(((("E4-m3", 100),) if run_e4 else ()))):
                rankings = mm[str(depth)]
                lexical, dense = rankings["bm25"], rankings["dense"]
                candidates = list(dict.fromkeys([key for key, _ in lexical] + [key for key, _ in dense]))
                texts = [chunks[key]["text"] for key in candidates]
                total_scores, usages = {}, {}
                for temperature in ("cold", "hot"):
                    scores, usage = reranker.score(q["query"], texts)
                    total_scores[temperature] = scores
                    usages[temperature] = usage
                if len(total_scores["cold"]) != len(candidates):
                    raise ValueError("Reranker score count differs from candidate union")
                cold_order = sorted(range(len(candidates)),
                    key=lambda i: (-total_scores["cold"][i], candidates[i]))
                hot_order = sorted(range(len(candidates)),
                    key=lambda i: (-total_scores["hot"][i], candidates[i]))
                ranked = [candidates[i] for i in cold_order]
                output = {
                    "ranked": ranked,
                    "before": [key for key, _ in rrf([lexical, dense], k=60)],
                    "candidate": candidates,
                    "scores": dict(zip(candidates, total_scores["cold"], strict=True)),
                    "cold_hot_order_equal": [candidates[i] for i in cold_order] ==
                                              [candidates[i] for i in hot_order],
                    "reranker_usage": usages,
                    "latency": _latency_record(m3[qid], depth,
                        reranker_ms={temp: usages[temp]["milliseconds"] for temp in usages},
                        reranker_usage=usages),
                }
                computed[qid]["arms"][arm] = output
    return computed, rerank_usage


def _latency_record(measurement, depth, *, ranking_ms=0.0, reranker_ms=None, reranker_usage=None):
    reranker_ms = reranker_ms or {"cold": 0.0, "hot": 0.0}
    result = {"cold": {}, "hot": {}}
    for temperature in ("cold", "hot"):
        measured_depth = next(value for value in CANDIDATE_DEPTHS if value >= depth)
        timing = measurement["timing"][temperature][str(measured_depth)]
        # Stored Retrieval V3 small-model measurements use encode_ms/dense_ms fields.
        source = timing
        scope = source.get("scope_ms", 0.0)
        lexical = source.get("bm25_ms", 0.0)
        query = source.get("query_encode_ms", source.get("encode_ms", 0.0))
        dense = source.get("qdrant_search_ms", source.get("dense_ms", 0.0))
        rr = reranker_ms[temperature]
        result[temperature] = {
            "scope_ms": scope, "bm25_ms": lexical, "query_encode_ms": query,
            "qdrant_search_ms": dense, "fusion_ms": ranking_ms,
            "reranker_ms": rr,
            "total_ms": scope + lexical + query + dense + ranking_ms + rr,
            "peak_allocated_bytes": (reranker_usage or {}).get(temperature, {}).get(
                "peak_allocated_bytes", source.get("peak_allocated_bytes")),
            "peak_reserved_bytes": (reranker_usage or {}).get(temperature, {}).get(
                "peak_reserved_bytes", source.get("peak_reserved_bytes")),
            "oom_batch_splits": (reranker_usage or {}).get(temperature, {}).get(
                "oom_batch_splits", source.get("oom_batch_splits", 0)),
        }
    return result


def line_truth(source: Path, questions, chunks):
    """V3-equivalent page-level numeric financial line coverage proxy."""
    audit = Audit(source, None)
    chunks_by_page = defaultdict(list)
    for key, row in chunks.items():
        chunks_by_page[(row["document_id"], row["page"])].append((key, normalize(row["text"])))
    truth = {}
    for count, q in enumerate(questions, 1):
        lines = []
        for doc, page in sorted(q["gold"]):
            anchors = [normalize(ev["evidence_text"]) for ev in q["evidence"]
                       if (ev["doc_name"], ev["evidence_page_num"]) == (doc, page)]
            for text in audit.page(doc, page):
                if not numeric_line(text):
                    continue
                target = normalize(text)
                if any(target in anchor for anchor in anchors):
                    lines.append({key for key, chunk in chunks_by_page[(doc, page)] if target in chunk})
        truth[q["id"]] = {"gold": q["gold"], "lines": lines}
        if count % 20 == 0 or count == len(questions):
            print(f"Numeric-line audit: {count}/{len(questions)}", flush=True)
    return truth


def page_key(chunks, key):
    row = chunks[key]
    return row["document_id"], row["page"]


def candidate_oracle(candidate, chunks, gold):
    candidate_pages = {page_key(chunks, key) for key in candidate}
    return min(5, len(candidate_pages & gold)) / len(gold)


def evaluate_arm(q, arm, payload, chunks, truth):
    ranked = payload["ranked"][:10]
    candidate = set(payload["candidate"])
    pages = [page_key(chunks, key) for key in ranked]
    documents = list(dict.fromkeys(doc for doc, _ in pages))
    gold = q["gold"]
    candidate_pages = {page_key(chunks, key) for key in candidate}
    first_ranks = {}
    for rank, key in enumerate(payload["ranked"], 1):
        first_ranks.setdefault(page_key(chunks, key), rank)
    lines = truth[q["id"]]["lines"]
    top5 = set(payload["ranked"][:5])
    line = {"identified": len(lines), "index_complete": sum(bool(items) for items in lines),
            "pool_complete": sum(bool(items & candidate) for items in lines),
            "top5_complete": sum(bool(items & top5) for items in lines)}
    categories = {"A_top5": 0, "B_in_pool_not_top5": 0, "C_not_in_pool": 0}
    top_pages = {page_key(chunks, key) for key in top5}
    for target in gold:
        if target in top_pages:
            categories["A_top5"] += 1
        elif target in candidate_pages:
            categories["B_in_pool_not_top5"] += 1
        else:
            categories["C_not_in_pool"] += 1
    return {
        "question_sha256": query_hash(q["id"]), "company_group": company_hash(q["company"]),
        "arm": arm,
        "page": relevance(pages, gold),
        "document": relevance(documents, {doc for doc, _ in gold}),
        "candidate_page_recall": len(candidate_pages & gold) / len(gold),
        "candidate_oracle_at_5": candidate_oracle(candidate, chunks, gold),
        "candidate_count": len(candidate), "unique_pages_at_5": len(top_pages),
        "returned_count_at_5": min(5, len(payload["ranked"])),
        "line_proxy": line, "loss_attribution": categories,
        "gold_page_ranks": [{"page_hash": page_hash(page),
                             "before": next((i for i, key in enumerate(payload.get("before", []), 1)
                                             if page_key(chunks, key) == page), None),
                             "after": first_ranks.get(page)}
                            for page in sorted(gold)],
        "latency": payload["latency"],
        "cold_hot_order_equal": payload.get("cold_hot_order_equal", True),
    }


def summarize_rows(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty arm")
    line_counts = {key: sum(row["line_proxy"][key] for row in rows) for key in
                   ("identified", "index_complete", "pool_complete", "top5_complete")}
    category_counts = {key: sum(row["loss_attribution"][key] for row in rows) for key in
                       ("A_top5", "B_in_pool_not_top5", "C_not_in_pool")}
    result = {
        "n": len(rows),
        "page": {metric: statistics.fmean(row["page"][metric] for row in rows)
                 for metric in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")},
        "document": {metric: statistics.fmean(row["document"][metric] for row in rows)
                     for metric in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")},
        "candidate_page_recall": statistics.fmean(row["candidate_page_recall"] for row in rows),
        "candidate_oracle_at_5": statistics.fmean(row["candidate_oracle_at_5"] for row in rows),
        "candidate_count_mean": statistics.fmean(row["candidate_count"] for row in rows),
        "unique_pages_at_5_mean": statistics.fmean(row["unique_pages_at_5"] for row in rows),
        "line_proxy": {**line_counts,
                       "coverage_at_5": line_counts["top5_complete"] / max(1, line_counts["identified"])},
        "loss_attribution": category_counts,
        "cold_hot_ranking_mismatches": sum(not row["cold_hot_order_equal"] for row in rows),
        "latency": {},
    }
    for temperature in ("cold", "hot"):
        timing_rows = [row["latency"][temperature] for row in rows]
        result["latency"][temperature] = {}
        for field in ("scope_ms", "bm25_ms", "query_encode_ms", "qdrant_search_ms",
                      "fusion_ms", "reranker_ms", "total_ms"):
            values = sorted(row[field] for row in timing_rows)
            def quantile(p):
                index = p * (len(values) - 1)
                low = int(index)
                return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)
            result["latency"][temperature][field] = {
                "mean": statistics.fmean(values), "p50": quantile(.5), "p95": quantile(.95)}
        result["latency"][temperature]["peak_allocated_bytes"] = max(
            row["peak_allocated_bytes"] or 0 for row in timing_rows)
        result["latency"][temperature]["peak_reserved_bytes"] = max(
            row["peak_reserved_bytes"] or 0 for row in timing_rows)
        result["latency"][temperature]["oom_batch_splits"] = sum(
            row["oom_batch_splits"] or 0 for row in timing_rows)
    return result


def add_e4(questions, m3, chunks, reranker, computed):
    for count, q in enumerate(questions, 1):
        entry = m3[q["id"]]
        lexical, dense = entry["rankings"]["100"]["bm25"], entry["rankings"]["100"]["dense"]
        candidate = list(dict.fromkeys([key for key, _ in lexical] + [key for key, _ in dense]))
        texts = [chunks[key]["text"] for key in candidate]
        scores_by_temp, usage = {}, {}
        for temperature in ("cold", "hot"):
            scores_by_temp[temperature], usage[temperature] = reranker.score(q["query"], texts)
        order = sorted(range(len(candidate)), key=lambda i: (-scores_by_temp["cold"][i], candidate[i]))
        hot_order = sorted(range(len(candidate)), key=lambda i: (-scores_by_temp["hot"][i], candidate[i]))
        computed[q["id"]]["arms"]["E4-m3"] = {
            "ranked": [candidate[i] for i in order],
            "before": [key for key, _ in rrf([lexical, dense], k=60)],
            "candidate": candidate,
            "scores": dict(zip(candidate, scores_by_temp["cold"], strict=True)),
            "cold_hot_order_equal": [candidate[i] for i in order] == [candidate[i] for i in hot_order],
            "reranker_usage": usage,
            "latency": _latency_record(entry, 100,
                reranker_ms={temp: usage[temp]["milliseconds"] for temp in usage}, reranker_usage=usage),
        }
        if count % 20 == 0 or count == len(questions):
            print(f"E4 reranker: {count}/{len(questions)}", flush=True)


def legacy_rows(path):
    result = []
    for row in jsonl_read(path):
        timings = {}
        for temperature in ("cold", "hot"):
            old = row["latency"][temperature]
            timings[temperature] = {
                "scope_ms": old["scope_ms"], "bm25_ms": old["bm25_ms"],
                "query_encode_ms": old["encode_ms"], "qdrant_search_ms": old["dense_ms"],
                "fusion_ms": 0.0, "reranker_ms": old["model_ms"], "total_ms": old["total_ms"],
                "peak_allocated_bytes": old["peak_allocated_bytes"],
                "peak_reserved_bytes": old["peak_reserved_bytes"],
                "oom_batch_splits": old["oom_batch_splits"],
            }
        result.append({
            "question_sha256": query_hash(row["id"]), "company_group": row["company_group"],
            "arm": "E0-S3", "page": row["page"], "document": row["document"],
            "candidate_page_recall": row["candidate_page_recall"],
            "candidate_oracle_at_5": row["candidate_oracle_at_5"],
            "candidate_count": row["candidate_count"],
            "unique_pages_at_5": row["unique_pages_at_5"],
            "returned_count_at_5": row["returned_count_at_5"],
            "line_proxy": row["line_proxy"], "gold_page_ranks": row["gold_page_ranks"],
            "latency": timings, "cold_hot_order_equal": True,
        })
    if len(result) not in {49, 99}:
        raise ValueError(f"Historical S3 baseline has unexpected denominator: {len(result)}")
    return result


def paired_comparison(rows, baseline):
    base = {row["question_sha256"]: row for row in baseline}
    if set(base) != {row["question_sha256"] for row in rows}:
        raise ValueError("Paired comparison question sets do not match")
    deltas, grouped, paired = [], defaultdict(list), []
    for row in rows:
        prior = base[row["question_sha256"]]
        delta = row["page"]["recall@5"] - prior["page"]["recall@5"]
        grouped[row["company_group"]].append(delta)
        paired.append({"question_sha256": row["question_sha256"],
                       "delta_recall_at_5": delta,
                       "status": "improved" if delta > 0 else "regressed" if delta < 0 else "tied"})
        deltas.append(delta)
    clusters = [(sum(values), len(values)) for values in grouped.values()]
    rng = Random(BOOTSTRAP["seed"])
    draws = []
    for _ in range(BOOTSTRAP["repetitions"]):
        sample = rng.choices(clusters, k=len(clusters))
        draws.append(sum(total for total, _ in sample) / sum(count for _, count in sample))
    draws.sort()
    return {
        "n": len(rows), "mean_delta_recall_at_5": statistics.fmean(deltas),
        "companies": len(clusters),
        "bootstrap_95_ci": [draws[int(.025 * len(draws))],
                             draws[min(len(draws) - 1, int(.975 * len(draws)))]],
        "improved": sum(delta > 0 for delta in deltas),
        "regressed": sum(delta < 0 for delta in deltas), "tied": sum(delta == 0 for delta in deltas),
        "paired": paired,
    }


def summarize_legacy(rows):
    metrics = ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")
    result = {"n": len(rows),
              "page": {key: statistics.fmean(row["page"][key] for row in rows) for key in metrics},
              "document": {key: statistics.fmean(row["document"][key] for row in rows) for key in metrics},
              "candidate_page_recall": statistics.fmean(row["candidate_page_recall"] for row in rows),
              "candidate_oracle_at_5": statistics.fmean(row["candidate_oracle_at_5"] for row in rows),
              "candidate_count_mean": statistics.fmean(row["candidate_count"] for row in rows),
              "unique_pages_at_5_mean": statistics.fmean(row["unique_pages_at_5"] for row in rows),
              "line_proxy": {}, "loss_attribution": None, "latency": {}}
    fields = ("scope_ms", "bm25_ms", "query_encode_ms", "qdrant_search_ms",
              "fusion_ms", "reranker_ms", "total_ms")
    line = ("identified", "index_complete", "pool_complete", "top5_complete")
    line_counts = {key: sum(row["line_proxy"][key] for row in rows) for key in line}
    result["line_proxy"] = {**line_counts,
        "coverage_at_5": line_counts["top5_complete"] / max(1, line_counts["identified"])}
    for temperature in ("cold", "hot"):
        result["latency"][temperature] = {}
        for field in fields:
            values = sorted(row["latency"][temperature][field] for row in rows)
            def quantile(p):
                pos = p * (len(values) - 1)
                low = int(pos)
                return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)
            result["latency"][temperature][field] = {
                "mean": statistics.fmean(values), "p50": quantile(.5), "p95": quantile(.95)}
        result["latency"][temperature]["peak_allocated_bytes"] = max(
            row["latency"][temperature]["peak_allocated_bytes"] or 0 for row in rows)
        result["latency"][temperature]["peak_reserved_bytes"] = max(
            row["latency"][temperature]["peak_reserved_bytes"] or 0 for row in rows)
        result["latency"][temperature]["oom_batch_splits"] = sum(
            row["latency"][temperature]["oom_batch_splits"] or 0 for row in rows)
    return result


def candidate_stage_summary(questions, small, m3, chunks):
    output = {}
    for label, values in (("BGE-small", small), ("BGE-M3", m3)):
        per_stage = {"BM25@50": [], "Dense@50": [], "Hybrid union@50": []}
        dense_at_k = {depth: [] for depth in DEPTHS}
        for q in questions:
            rank = values[q["id"]]["rankings"]["50"]
            lexical = [key for key, _ in rank["bm25"]]
            dense = [key for key, _ in rank["dense"]]
            union = list(dict.fromkeys(lexical + dense))
            for stage, keys in (("BM25@50", lexical), ("Dense@50", dense), ("Hybrid union@50", union)):
                pages = {page_key(chunks, key) for key in keys}
                gold = q["gold"]
                per_stage[stage].append({"page_recall": len(pages & gold) / len(gold),
                    "candidate_oracle_at_5": min(5, len(pages & gold)) / len(gold),
                    "candidate_count": len(keys)})
            for depth in DEPTHS:
                dense = dense_top_k(values[q["id"]]["rankings"], depth)
                if dense is None:
                    dense_at_k[depth].append(None)
                    continue
                dense_keys = [key for key, _ in dense]
                dense_pages = {page_key(chunks, key) for key in dense_keys}
                dense_at_k[depth].append(len(dense_pages & q["gold"]) / len(q["gold"]))
        output[label] = {stage: {
            "n": len(stage_rows), "page_recall": statistics.fmean(row["page_recall"] for row in stage_rows),
            "candidate_oracle_at_5": statistics.fmean(row["candidate_oracle_at_5"] for row in stage_rows),
            "candidate_count_mean": statistics.fmean(row["candidate_count"] for row in stage_rows),
        } for stage, stage_rows in per_stage.items()}
        output[label]["dense_page_recall_at_k"] = {
            str(depth): {"n": sum(value is not None for value in values_at_depth),
                         "mean": statistics.fmean(value for value in values_at_depth if value is not None)
                         if any(value is not None for value in values_at_depth) else None,
                         "available": all(value is not None for value in values_at_depth)}
            for depth, values_at_depth in dense_at_k.items()
        }
    return output


def arm_rows(questions, computed, chunks, truth):
    result = defaultdict(list)
    for q in questions:
        for arm, payload in computed[q["id"]]["arms"].items():
            result[arm].append(evaluate_arm(q, arm, payload, chunks, truth))
    return dict(result)


def arm_sort_key(item):
    name, summary = item
    page = summary["page"]
    return (-page["recall@5"], -page["recall@1"], -page["recall@10"],
            -page["mrr@10"], -page["ndcg@10"], -summary["line_proxy"]["coverage_at_5"],
            summary["latency"]["hot"]["total_ms"]["mean"], name)


def choose_calibration(summary):
    arms = summary["arms"]
    hybrid_names = [name for name in ("E2-small", "E2-m3") if name in arms]
    ranked_names = [name for name in ("E2-small", "E2-m3", "E3-m3", "E4-m3") if name in arms]
    if not hybrid_names or not ranked_names:
        raise ValueError("Calibration did not produce required hybrid/reranker arms")
    best_hybrid = min(((name, arms[name]) for name in hybrid_names), key=arm_sort_key)[0]
    quality = min(((name, arms[name]) for name in ranked_names), key=arm_sort_key)[0]
    baseline = arms["E0-S3"]
    eligible = []
    for name in ranked_names:
        current = arms[name]
        if (current["page"]["recall@5"] <= baseline["page"]["recall@5"]
                or any(current["page"][key] < baseline["page"][key] - .03
                       for key in ("recall@1", "recall@10", "mrr@10"))):
            continue
        if any(current["latency"][temp]["total_ms"]["mean"] >
               2 * baseline["latency"][temp]["total_ms"]["mean"] for temp in ("cold", "hot")):
            continue
        eligible.append(name)
    deploy = min(((name, arms[name]) for name in eligible), key=arm_sort_key)[0] if eligible else "E0-S3"
    return {"best_hybrid_no_reranker": best_hybrid, "quality_best": quality,
            "deployment_candidate": deploy,
            "selection_rules": {"primary": "Calibration page Recall@5",
                "tie_breaks": ["Recall@1", "Recall@10", "MRR@10", "nDCG@10",
                               "numeric-line coverage@5", "hot latency"],
                "deployment_gate": "Recall@5 gain; Recall@1/10 and MRR loss <=0.03; cold/hot mean latency <=2x E0"}}


def run_calibration(source: Path, run: Path, device: str, max_length: int):
    if (RESULTS / "freeze.json").exists() or (run / "calibration-summary.json").exists():
        raise FileExistsError("Calibration or freeze already exists; refusing to overwrite")
    questions = calibration_questions(source.resolve())
    chunks = read_index_payloads(INDEX_ROOT)
    small, small_hash = small_model_candidates("calibration")
    m3_path = run / "calibration-m3-candidates.json"
    if m3_path.is_file():
        cached = json_read(m3_path)
        expected_ids = [q["id"] for q in questions]
        if (cached.get("collection") != COLLECTION or cached.get("model_revision") != MODEL_REVISION
                or cached.get("question_ids") != expected_ids
                or cached.get("depths") != list(CANDIDATE_DEPTHS)):
            raise ValueError("Existing Calibration candidate cache does not match the frozen retrieval setup")
        m3 = cached["measurements"]
    else:
        m3 = collect_candidates(source, INDEX_ROOT, m3_path, "Calibration", questions, device, max_length)
    free_cuda()
    truth = line_truth(source.resolve(), questions, chunks)
    load_started = perf_counter()
    reranker = BGEReranker(RERANKER_ROOT, device)
    reranker_load = perf_counter() - load_started
    computed, warmup = rank_candidates(questions, small, m3, chunks, reranker, run_e4=False)
    rows = arm_rows(questions, computed, chunks, truth)
    e3 = summarize_rows(rows["E3-m3"])
    attribution = e3["loss_attribution"]
    run_e4 = (e3["candidate_oracle_at_5"] < .90
              and attribution["C_not_in_pool"] > attribution["B_in_pool_not_top5"])
    if run_e4:
        add_e4(questions, m3, chunks, reranker, computed)
        rows = arm_rows(questions, computed, chunks, truth)
    legacy = legacy_rows(V3_RUN / "calibration/S3.jsonl")
    if {row["question_sha256"] for row in legacy} != {query_hash(q["id"]) for q in questions}:
        raise ValueError("V3 S3 Calibration baseline does not match the frozen 49 questions")
    summaries = {name: summarize_rows(values) for name, values in rows.items()}
    summaries["E0-S3"] = summarize_legacy(legacy)
    comparisons = {name: paired_comparison(values, legacy) for name, values in rows.items()}
    output = {
        "dataset": {"split": "Calibration", "n": 49,
                    "question_hashes_sha256": digest_obj(sorted(query_hash(q["id"]) for q in questions))},
        "arms": summaries,
        "candidate_stages": candidate_stage_summary(questions, small, m3, chunks),
        "paired_vs_E0": {name: {key: value for key, value in result.items() if key != "paired"}
                         for name, result in comparisons.items()},
        "selection": None,
        "e4": {"executed": run_e4, "trigger": "E3 oracle@5 < 0.90 and C > B on Calibration",
               "e3_oracle_at_5": e3["candidate_oracle_at_5"],
               "e3_C_not_in_pool": attribution["C_not_in_pool"],
               "e3_B_in_pool_not_top5": attribution["B_in_pool_not_top5"]},
        "reranker": {"id": "BAAI/bge-reranker-v2-m3", "revision": RERANKER_REVISION,
                     "weights_sha256": file_tree_hashes(RERANKER_ROOT).get("model.safetensors"),
                     "load_seconds": reranker_load, "warmup": warmup["warmup"]},
        "bootstrap": BOOTSTRAP, "glm_requests": 0, "glm_tokens": 0,
    }
    output["selection"] = choose_calibration(output)
    raw_rows = {arm: [{**row, "question_id": q["id"]}
                      for q, row in zip(questions, values, strict=True)] for arm, values in rows.items()}
    json_write(run / "calibration-scores.json", raw_rows)
    json_write(run / "calibration-summary.json", output)
    jsonl_write(RESULTS / "calibration-rows.jsonl",
                [row for values in rows.values() for row in values], overwrite=True)
    return output


def free_cuda():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def freeze_calibration(source: Path, run: Path):
    if (RESULTS / "freeze.json").exists():
        raise FileExistsError("V4 freeze already exists")
    summary_path, candidate_path, prepare_path = (
        run / "calibration-summary.json", run / "calibration-m3-candidates.json", run / "prepare.json")
    if not all(path.is_file() for path in (summary_path, candidate_path, prepare_path)):
        raise FileNotFoundError("Prepare and Calibration artifacts are required before freeze")
    calibration = json_read(summary_path)
    choices = calibration["selection"]
    reranked = ["E3-m3", *( ["E4-m3"] if calibration["e4"]["executed"] else [] )]
    reranked_best = min(((name, calibration["arms"][name]) for name in reranked), key=arm_sort_key)[0]
    model_hashes = file_tree_hashes(MODEL_ROOT)
    if model_hashes != json_read(prepare_path)["model"]["files_sha256"]:
        raise ValueError("BGE-M3 weights changed after prepare")
    code_hashes = {name: file_sha256(ROOT / name) for name in SOURCE_CODE}
    index_manifest = json_read(INDEX_ROOT / "v4-index-manifest.json")
    if index_manifest["indexed_chunk_payload_sha256"] != index_manifest["original_chunk_payload_sha256"]:
        raise ValueError("V4 index payload differs from original chunks")
    v3_freeze = json_read(V3_RUN / "freeze.json")
    freeze = {
        "experiment": "A1-V4 Stronger Embedding Retrieval Ablation",
        "base_commit": "ee7998438ffeee4d786eed46dcb925d1798afaf9",
        "branch": "experiment/a1-retrieval-v4-embedding",
        "formal_test_accessed": False,
        "test_split_note": "The 99-question set is a repeatedly evaluated public benchmark; it is not an unseen test.",
        "v3_freeze_sha256": file_sha256(V3_RUN / "freeze.json"),
        "public_dataset_sha256_from_v3_freeze": v3_freeze["inputs"]["A1_V2_inputs"]["dataset_sha256"],
        "calibration": {"n": 49,
            "question_hashes_sha256": calibration["dataset"]["question_hashes_sha256"],
            "candidate_cache_sha256": file_sha256(candidate_path),
            "summary_sha256": file_sha256(summary_path),
            "legacy_s3_calibration_sha256": file_sha256(V3_RUN / "calibration/S3.jsonl")},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "files_sha256": model_hashes,
            "dimension": DIMENSION, "pooling": "CLS", "normalization": "L2", "query_prefix": None,
            "input_limit": json_read(prepare_path)["encoder"]["maximum_input_tokens"],
            "device": json_read(prepare_path)["encoder"]["device"]},
        "index": {"collection": COLLECTION, "path_local_only": "data/a1-retrieval-v4/001/bge-m3-index",
            "manifest_sha256": file_sha256(INDEX_ROOT / "v4-index-manifest.json"),
            "payload_sha256": index_manifest["indexed_chunk_payload_sha256"],
            "vector_sha256": index_manifest["indexed_vector_sha256"],
            "chunks": index_manifest["chunks"], "documents": index_manifest["documents"]},
        "reranker": {"id": "BAAI/bge-reranker-v2-m3", "revision": RERANKER_REVISION,
            "weights_sha256": RERANKER_SHA256, "files_sha256": file_tree_hashes(RERANKER_ROOT)},
        "source_code_sha256": code_hashes,
        "selection": {**choices, "best_hybrid_with_reranker": reranked_best,
                      "e4_executed": calibration["e4"]["executed"]},
        "formal_plan": ["E0-S3", "E1-small@1/5/10/24/50 (archived Test cache ends at @50)",
                        "E1-m3@1/5/10/24/50/100",
                        "E2-small", "E2-m3", reranked_best],
        "test_cache_constraint": "The archived V3 BGE-small Test cache contains only @24/@50; @100 is reported Unavailable, not reconstructed from the original index/cache.",
        "candidate_config": {"bm25_k": 50, "dense_k": 50, "rrf_k": 60, "rrf_weights": "equal",
            "candidate_union_dedup_key": "Chunk ID", "reranker_before_truncation": True,
            "final_top_k": 5, "reranker": "same BGE-reranker-v2-m3 as S3"},
        "e4": calibration["e4"], "bootstrap": BOOTSTRAP,
        "glm_requests": 0, "glm_tokens": 0,
    }
    json_write(RESULTS / "freeze.json", freeze)
    json_write(run / "freeze.json", freeze)
    return freeze


def verify_frozen_inputs(freeze, run):
    if freeze["base_commit"] != "ee7998438ffeee4d786eed46dcb925d1798afaf9":
        raise ValueError("V4 freeze base commit is not A1-V3")
    current_code = {name: file_sha256(ROOT / name) for name in SOURCE_CODE}
    if current_code != freeze["source_code_sha256"]:
        raise ValueError("Source code changed after Calibration freeze")
    if file_tree_hashes(MODEL_ROOT) != freeze["model"]["files_sha256"]:
        raise ValueError("BGE-M3 weights changed after Calibration freeze")
    if file_tree_hashes(RERANKER_ROOT) != freeze["reranker"]["files_sha256"]:
        raise ValueError("V3 reranker weights changed after Calibration freeze")
    if file_sha256(run / "calibration-m3-candidates.json") != freeze["calibration"]["candidate_cache_sha256"]:
        raise ValueError("Calibration candidates changed after freeze")
    if file_sha256(run / "calibration-summary.json") != freeze["calibration"]["summary_sha256"]:
        raise ValueError("Calibration summary changed after freeze")
    if file_sha256(INDEX_ROOT / "v4-index-manifest.json") != freeze["index"]["manifest_sha256"]:
        raise ValueError("V4 index manifest changed after freeze")
    return True


def index_semantic_fingerprint(index_root: Path, *, expected_chunks: int):
    payload_hash, vector_hash = sha256(), sha256()
    with closing(QdrantClient(path=str(index_root))) as client:
        points = scroll_points(client, COLLECTION, with_vectors=True)
    points.sort(key=lambda point: str(point.id))
    if len(points) != expected_chunks:
        raise ValueError("Frozen index point count changed")
    for point in points:
        payload_hash.update(json.dumps([str(point.id), point.payload], sort_keys=True,
            ensure_ascii=False, separators=(",", ":")).encode())
        payload_hash.update(b"\n")
        vector = point.vector.get("dense") if isinstance(point.vector, dict) else point.vector
        if vector is None or len(vector) != DIMENSION or any(not math.isfinite(v) for v in vector):
            raise ValueError("Frozen index contains invalid vectors")
        if abs(math.sqrt(sum(v * v for v in vector)) - 1.0) > 2e-4:
            raise ValueError("Frozen index contains nonnormalized vectors")
        vector_hash.update(str(point.id).encode())
        vector_hash.update(struct.pack(f"<{DIMENSION}f", *vector))
    return {"chunks": len(points), "payload_sha256": payload_hash.hexdigest(),
            "vector_sha256": vector_hash.hexdigest(), "verified": True}


def verify_index_semantics(index_root: Path, expected):
    fingerprint = index_semantic_fingerprint(index_root, expected_chunks=expected["chunks"])
    if (fingerprint["payload_sha256"] != expected["payload_sha256"]
            or fingerprint["vector_sha256"] != expected["vector_sha256"]):
        raise ValueError("V4 index semantic fingerprint changed after freeze")
    return fingerprint


def _deployment_passed(summaries, selection):
    arm = selection["deployment_candidate"]
    if arm == "E0-S3":
        return True
    baseline, current = summaries["E0-S3"], summaries[arm]
    return all(current["latency"][temp]["total_ms"]["mean"] <=
               2 * baseline["latency"][temp]["total_ms"]["mean"] for temp in ("cold", "hot"))


def run_formal_test(source: Path, run: Path, device: str, max_length: int):
    freeze_path = RESULTS / "freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("Formal Test requires results/a1-retrieval-v4/freeze.json")
    freeze = json_read(freeze_path)
    verify_frozen_inputs(freeze, run)
    index_check = verify_index_semantics(INDEX_ROOT, freeze["index"])
    # Test questions and labels are only loaded after the frozen inputs pass verification.
    questions = test_questions_after_freeze(source.resolve())
    small, small_hash = small_model_candidates("test")
    if set(small) != {q["id"] for q in questions}:
        raise ValueError("Frozen BGE-small Test candidate cache does not match all 99 questions")
    m3_path = run / "test-m3-candidates.json"
    if m3_path.is_file():
        cached = json_read(m3_path)
        expected_ids = [q["id"] for q in questions]
        if (cached.get("collection") != COLLECTION or cached.get("model_revision") != MODEL_REVISION
                or cached.get("question_ids") != expected_ids
                or cached.get("depths") != list(CANDIDATE_DEPTHS)):
            raise ValueError("Existing BGE-M3 Test candidate cache does not match the frozen retrieval setup")
        m3 = cached["measurements"]
    else:
        m3 = collect_candidates(source, INDEX_ROOT, m3_path, "Test", questions, device, max_length)
    free_cuda()
    chunks = read_index_payloads(INDEX_ROOT)
    truth = line_truth(source.resolve(), questions, chunks)
    load_started = perf_counter()
    reranker = BGEReranker(RERANKER_ROOT, device)
    reranker_load_seconds = perf_counter() - load_started
    computed, warmup = rank_candidates(questions, small, m3, chunks, reranker,
                                       run_e4=freeze["e4"]["executed"])
    rows = arm_rows(questions, computed, chunks, truth)
    legacy = legacy_rows(V3_RUN / "test/S3.jsonl")
    ids = {query_hash(q["id"]) for q in questions}
    if len(legacy) != 99 or {row["question_sha256"] for row in legacy} != ids:
        raise ValueError("Historical S3 Test denominator or public question set differs")
    summaries = {name: summarize_rows(values) for name, values in rows.items()}
    summaries["E0-S3"] = summarize_legacy(legacy)
    comparisons = {name: paired_comparison(values, legacy) for name, values in rows.items()}
    result = {
        "dataset": {"split": "Public Test", "n": 99,
                    "question_hashes_sha256": digest_obj(sorted(ids)),
                    "independent_unseen_test": False},
        "arms": summaries,
        "candidate_stages": candidate_stage_summary(questions, small, m3, chunks),
        "paired_vs_E0": {name: {key: value for key, value in comparison.items() if key != "paired"}
                         for name, comparison in comparisons.items()},
        "paired_questions": {name: comparison["paired"] for name, comparison in comparisons.items()},
        "formal_selection_from_freeze": freeze["selection"],
        "deployment_latency_gate": {"baseline": "E0-S3 archived same-host A1-V3 run",
                                    "limit_multiplier": 2,
                                    "passed": _deployment_passed(summaries, freeze["selection"])},
        "index_integrity": index_check,
        "small_candidate_cache_sha256": small_hash,
        "m3_candidate_cache_sha256": file_sha256(m3_path),
        "reranker": {"id": "BAAI/bge-reranker-v2-m3", "revision": RERANKER_REVISION,
                     "load_seconds": reranker_load_seconds, "warmup": warmup["warmup"]},
        "bootstrap": BOOTSTRAP, "glm_requests": 0, "glm_tokens": 0,
    }
    raw_rows = {arm: [{**row, "question_id": q["id"]}
                      for q, row in zip(questions, arm_result, strict=True)]
                for arm, arm_result in rows.items()}
    json_write(run / "test-scores.json", raw_rows)
    json_write(run / "test-summary.json", result)
    jsonl_write(RESULTS / "test-rows.jsonl", [row for arm_result in rows.values() for row in arm_result])
    json_write(run / "test-checks.json", {
        "test_denominator": len(questions), "all_99_scored": True,
        "freeze_sha256": file_sha256(freeze_path), "code_freeze_verified": True,
        "index_integrity": index_check, "glm_requests": 0, "glm_tokens": 0,
        "historical_s3_test_baseline_sha256": file_sha256(V3_RUN / "test/S3.jsonl"),
    })
    final = {"calibration": json_read(run / "calibration-summary.json"), "test": result}
    json_write(RESULTS / "summary.json", final)
    write_deliverables(final, json_read(run / "prepare.json"), freeze)
    return final


def _pct(value):
    return "Unavailable" if value is None else f"{100 * value:.2f}%"


def _ms(value):
    return f"{value:,.1f} ms"


def _svg_text(x, y, value, *, size=14, fill="#dbe6f3", anchor="start", weight="normal"):
    from html import escape
    return (f'<text x="{x}" y="{y}" font-family="Inter,Arial,sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(str(value))}</text>')


def write_candidate_chart(stats):
    names = [name for name in ("E0-S3", "E1-small@50", "E1-m3@50", "E2-small", "E2-m3",
                                "E3-m3", "E4-m3") if name in stats]
    width, height = 1120, 120 + 56 * len(names)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#0b101a"/>',
             _svg_text(24, 34, "Public Test · Candidate Oracle vs Final Recall@5", size=20, weight="600"),
             _svg_text(24, 64, "Same 99-question denominator · page-level mean · real archived retrieval results", size=12, fill="#94a3b8")]
    x0, plot_w, y0 = 310, 650, 100
    for index in range(6):
        x = x0 + plot_w * index / 5
        parts.append(f'<line x1="{x:.1f}" y1="{y0-8}" x2="{x:.1f}" y2="{height-24}" stroke="#273244" stroke-width="1"/>')
        parts.append(_svg_text(x, y0 - 15, f"{index * 20}%", size=11, fill="#94a3b8", anchor="middle"))
    for i, name in enumerate(names):
        y = y0 + i * 56
        arm = stats[name]
        oracle = arm["candidate_oracle_at_5"]
        final = arm["page"]["recall@5"]
        parts.append(_svg_text(x0 - 16, y + 19, name, size=13, anchor="end"))
        for offset, value, color, label in ((0, oracle, "#38bdf8", "Oracle"), (22, final, "#34d399", "Final")):
            bar = plot_w * value
            parts.append(f'<rect x="{x0}" y="{y+offset}" width="{bar:.1f}" height="14" rx="3" fill="{color}"/>')
            parts.append(_svg_text(x0 + bar + 8, y + offset + 12, _pct(value), size=11, fill="#e6edf7"))
        parts.append(_svg_text(1000, y + 11, "Oracle", size=10, fill="#38bdf8"))
        parts.append(_svg_text(1000, y + 33, "Final", size=10, fill="#34d399"))
    parts.append("</svg>")
    return "\n".join(parts)


def write_latency_chart(stats):
    names = [name for name in ("E0-S3", "E1-small@50", "E1-m3@50", "E2-small", "E2-m3",
                                "E3-m3", "E4-m3") if name in stats]
    width, height = 1100, 620
    left, top, plot_w, plot_h = 100, 70, 900, 470
    max_latency = max(stats[name]["latency"]["hot"]["total_ms"]["mean"] for name in names) * 1.12
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#0b101a"/>',
             _svg_text(28, 35, "Public Test · Recall@5 and Hot Latency", size=20, weight="600"),
             _svg_text(28, 57, "Latency includes scope scan, BM25, query encoding, Qdrant search and ranking", size=12, fill="#94a3b8")]
    for tick in range(6):
        recall = tick / 5
        y = top + plot_h * (1 - recall)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#273244"/>')
        parts.append(_svg_text(left - 12, y + 4, f"{recall:.0%}", size=11, fill="#94a3b8", anchor="end"))
        latency = max_latency * tick / 5
        x = left + plot_w * tick / 5
        parts.append(_svg_text(x, top + plot_h + 24, f"{latency:.0f} ms", size=11, fill="#94a3b8", anchor="middle"))
    palette = ("#fbbf24", "#a78bfa", "#38bdf8", "#94a3b8", "#60a5fa", "#34d399", "#f87171")
    for i, name in enumerate(names):
        x = left + plot_w * stats[name]["latency"]["hot"]["total_ms"]["mean"] / max_latency
        y = top + plot_h * (1 - stats[name]["page"]["recall@5"])
        color = palette[i % len(palette)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="#0b101a" stroke-width="2"/>')
        parts.append(_svg_text(x + 11, y - 5, name, size=11, fill=color))
    parts.append(_svg_text(left + plot_w / 2, height - 18, "Hot mean end-to-end latency", size=12, fill="#94a3b8", anchor="middle"))
    parts.append(_svg_text(22, top + plot_h / 2, "Recall@5", size=12, fill="#94a3b8"))
    parts.append("</svg>")
    return "\n".join(parts)


def write_deliverables(final, prepare_result, freeze):
    RESULTS.mkdir(parents=True, exist_ok=True)
    test, cal = final["test"], final["calibration"]
    charts = RESULTS / "charts"
    charts.mkdir(exist_ok=True)
    (charts / "candidate-oracle-vs-final.svg").write_text(
        write_candidate_chart(test["arms"]), encoding="utf-8")
    (charts / "recall-latency.svg").write_text(write_latency_chart(test["arms"]), encoding="utf-8")
    arms = test["arms"]
    table_names = [name for name in ("E0-S3", "E1-small@50", "E1-m3@50", "E2-small", "E2-m3",
                                      "E3-m3", "E4-m3") if name in arms]
    table = ["| Arm | Candidate Oracle@5 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Hot mean | Hot P95 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in table_names:
        arm = arms[name]
        hot = arm["latency"]["hot"]
        table.append(f"| {name} | {_pct(arm['candidate_oracle_at_5'])} | {_pct(arm['page']['recall@1'])} | "
            f"{_pct(arm['page']['recall@5'])} | {_pct(arm['page']['recall@10'])} | "
            f"{arm['page']['mrr@10']:.3f} | {arm['page']['ndcg@10']:.3f} | "
            f"{_ms(hot['total_ms']['mean'])} | {_ms(hot['total_ms']['p95'])} |")
    candidate_table = [
        "| Embedding | Dense R@1 | Dense R@5 | Dense R@10 | Dense R@24 | Dense R@50 | Dense R@100 | BM25@50 page recall | Dense@50 page recall | Hybrid union page recall | Hybrid Oracle@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("BGE-small", "BGE-M3"):
        stage = test["candidate_stages"][label]
        dense = stage["dense_page_recall_at_k"]
        candidate_table.append(
            f"| {label} | " + " | ".join(_pct(dense[str(depth)]['mean']) for depth in DEPTHS) +
            f" | {_pct(stage['BM25@50']['page_recall'])} | {_pct(stage['Dense@50']['page_recall'])} | "
            f"{_pct(stage['Hybrid union@50']['page_recall'])} | "
            f"{_pct(stage['Hybrid union@50']['candidate_oracle_at_5'])} |"
        )
    prep = prepare_result
    token = prep["token_length_audit"]
    exact = prep["qdrant_exact_cosine_audit"]
    selection = freeze["selection"]
    limitations = [
        "FinanceBench's 99-question public Test is reused and is not an independent unseen test.",
        "Financial-line coverage is the same V3 numeric-line proxy; it is not a manually adjudicated XBRL metric.",
        "Cold timing follows V3's first retrieval pass with company scope cache cleared; OS page cache is not evicted.",
        "Model loading and reranker warm-up are reported separately and excluded from per-query cold/hot latency.",
        "Retrieval uses the original chunk payloads, BM25 implementation, Qdrant filters, and page-level mean recall.",
    ]
    if not cal["e4"]["executed"]:
        limitations.append("E4 Top-100 was skipped because the Calibration trigger did not identify candidate coverage as the dominant bottleneck.")
    if not test["candidate_stages"]["BGE-small"]["dense_page_recall_at_k"]["100"]["available"]:
        limitations.append("The archived V3 BGE-small Test cache ends at @50. BGE-small Dense@100 is reported Unavailable; the original Qdrant index and embedding cache were not accessed or modified to reconstruct it.")
    report = f"""# A1-V4 Stronger Embedding Retrieval Ablation

## Result

The experiment uses the frozen A1-V3 base commit `{freeze['base_commit']}` on branch `{freeze['branch']}`. It uses BGE-M3 dense embeddings at revision `{MODEL_REVISION}`, the unchanged BGE-small S3 control, the same BGE reranker as S3, and the existing 83-PDF / 36,622-chunk corpus. No model was trained and no GLM request was made.

The 49 shared Calibration questions select the configuration before formal Test. The 99-question FinanceBench public Test has been evaluated repeatedly in prior work and is not an unseen test set.

## Frozen setup and integrity

- Model: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), 1,024 dimensions, CLS pooling, L2 normalization, no query instruction prefix, dense only, `{freeze['model']['input_limit']}` token limit, `{freeze['model']['device']}` inference.
- Pinned model revision: `{MODEL_REVISION}`. Per-file SHA256 hashes are in `freeze.json`.
- Token lengths: mean `{token['mean']:.1f}`, P50 `{token['p50']}`, P95 `{token['p95']}`, P99 `{token['p99']}`, max `{token['max']}`. `{token['truncated_at_512']}/{token['chunks']}` chunks exceed 512 tokens ({_pct(token['truncation_rate_at_512'])}); `{token['truncated_at_1024']}/{token['chunks']}` exceed 1,024 ({_pct(token['truncation_rate_at_1024'])}).
- New independent Qdrant collection: `{COLLECTION}`, cosine distance, `{prep['index']['chunks']:,}` encoded chunks and `{prep['index']['documents']}` documents. Failed encodings: `{prep['index']['encoding_failures']}`. Payload/Chunk ID equality audit: `{prep['index']['chunk_ids_and_text_metadata_match']}`.
- Index size: `{prep['index']['index_bytes']:,}` bytes. Build: `{prep['index']['build_seconds']:.1f}s`, `{prep['index']['chunks_per_second']:.1f}` chunks/s. Peak embedding VRAM allocated/reserved: `{prep['index']['peak_allocated_bytes']:,}` / `{prep['index']['peak_reserved_bytes']:,}` bytes.
- Qdrant versus exact cosine audit, `{exact['sample_n']}` Calibration queries: top-100 overlap `{_pct(exact['mean_top100_overlap'])}`, top-10 overlap `{_pct(exact['mean_top10_overlap'])}`, top-1 equal `{exact['top1_equal_count']}/{exact['sample_n']}`.
- The smoke test used `{prep['smoke']['real_chunks']}` actual chunks: `{prep['smoke']['dimension']}`-d vectors, unit norms, repeat max absolute delta `{prep['smoke']['repeat_max_abs_delta']:.2g}`, batch/single max delta `{prep['smoke']['single_batch_max_abs_delta']:.2g}`, exact self-retrieval top-1 passed. Peak smoke VRAM `{prep['smoke']['peak_allocated_bytes']:,}` allocated / `{prep['smoke']['peak_reserved_bytes']:,}` reserved bytes.
- Query encoding uses CUDA; BM25 and local Qdrant search run on CPU. Qdrant query timing includes the synchronized GPU-to-host vector transfer. Model loading and reranker warm-up are recorded separately.

## Calibration selection

Calibration denominator: `{cal['dataset']['n']}`. E4 was `{'executed' if cal['e4']['executed'] else 'skipped'}`. Selection was fixed as follows: best hybrid without reranker `{selection['best_hybrid_no_reranker']}`, quality candidate `{selection['quality_best']}`, deployment candidate `{selection['deployment_candidate']}`, best reranked candidate `{selection['best_hybrid_with_reranker']}`.

| Arm | Candidate Oracle@5 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Hot mean |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(f"| {name} | {_pct(cal['arms'][name]['candidate_oracle_at_5'])} | {_pct(cal['arms'][name]['page']['recall@1'])} | {_pct(cal['arms'][name]['page']['recall@5'])} | {_pct(cal['arms'][name]['page']['recall@10'])} | {cal['arms'][name]['page']['mrr@10']:.3f} | {cal['arms'][name]['page']['ndcg@10']:.3f} | {_ms(cal['arms'][name]['latency']['hot']['total_ms']['mean'])} |" for name in ("E0-S3", "E1-small@50", "E1-m3@50", "E2-small", "E2-m3", "E3-m3", "E4-m3") if name in cal['arms'])}

## Public Test results

Test denominator: `{test['dataset']['n']}` / 99. Metrics use the same page-level mean recall, MRR@10 and nDCG@10 calculation as A1-V3. All 99 questions remain in the denominator. RRF is fixed at k=60 and equal weights; no weight grid was searched. E3 reranks the entire de-duplicated BM25@50 + dense@50 union before returning Top-5.

{chr(10).join(table)}

### Candidate-stage recall

Dense Recall@K is page-level recall among the first K Dense-ranked chunks, matching the A1-V3 page-level mean-recall convention. Candidate-pool page recall and the Top-5 oracle are shown separately for BM25, dense, and their de-duplicated union.

{chr(10).join(candidate_table)}

![Candidate oracle compared with final Recall@5](charts/candidate-oracle-vs-final.svg)

![Recall and latency comparison](charts/recall-latency.svg)

### Separating the three effects

- **Embedding retrieval:** compare `E1-small@50` with `E1-m3@50` in the pure Dense curve and `candidate_stages`. This isolates Dense candidate quality.
- **Candidate coverage:** compare the BM25@50, Dense@50 and de-duplicated hybrid union page recall / Oracle@5. `candidate_stages` reports each independently.
- **Final ranking:** compare E2 hybrid RRF (no reranker) with E3 BGE-M3 hybrid plus the unchanged reranker. A gap from Oracle@5 to final Recall@5 measures how much candidate coverage the final ordering fails to realize.

Per-arm paired improved/regressed/tied counts and company-cluster bootstrap 95% intervals are in `summary.json`; deidentified per-question deltas are in `summary.json` and `test-rows.jsonl`. Loss is attributed by gold page: A entered final Top-5, B was present in the candidate pool but missed Top-5, C was absent from the pool.

### Latency and resource details

`summary.json` reports cold and hot mean/P50/P95 for scope scan, BM25, query encoding, Qdrant search, fusion, reranking and end-to-end latency; each arm includes peak allocated/reserved GPU memory and OOM batch splits. The pinned model load time is `{prep['encoder']['model_load_seconds']:.1f}s`; reranker load and warm-up are in `summary.json`. Cold follows the A1-V3 first retrieval pass and does not flush the OS cache. Model startup/warm-up costs are excluded from per-query latency.

### Limits and reproducibility

""" + "\n".join(f"- {item}" for item in limitations) + f"""

GLM requests: `{test['glm_requests']}`. GLM tokens: `{test['glm_tokens']}`. New Python package dependencies: none; the run reused the A1-V3 benchmark CUDA environment (PyTorch, Transformers, Hugging Face Hub, Qdrant client).

Run from the V4 worktree with the A1-V3 benchmark environment:

```bash
PY=/home/gaoyi/projects/tradingagents-a1-retrieval-v3/data/a1-retrieval-v3/benchmark-env/bin/python
$PY -m tradingagents_lite.retrieval_v4 prepare
$PY -m tradingagents_lite.retrieval_v4 calibrate
$PY -m tradingagents_lite.retrieval_v4 freeze
$PY -m tradingagents_lite.retrieval_v4 test
```

The Test command verifies the frozen source-code, model, calibration and index fingerprints before loading the 99 Test questions. Local model files, vectors, index and raw candidate caches stay under ignored `data/a1-retrieval-v4/001/`; they are not committed.
"""
    (RESULTS / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "calibrate", "freeze", "test"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.run, args.device, args.max_length)
    elif args.command == "calibrate":
        result = run_calibration(args.source, args.run, args.device, args.max_length)
    elif args.command == "freeze":
        result = freeze_calibration(args.source, args.run)
    else:
        result = run_formal_test(args.source, args.run, args.device, args.max_length)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
