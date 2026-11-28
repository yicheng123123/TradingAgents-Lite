"""Read frozen indexes without opening Qdrant's writable local client."""
from collections import Counter, defaultdict
from contextlib import closing
from functools import lru_cache
from math import log
from pathlib import Path
import pickle
import sqlite3
from time import perf_counter

import numpy as np

from ..data import sha256
from ..evidence import retain_versions
from ..rag import terms
from ..rag_benchmark import CUTOFF, PARTITION

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ROOT.parent
SOURCE = PROJECTS / "tradingagents-lite"
V3 = PROJECTS / "tradingagents-a1-retrieval-v3/data/a1-retrieval-v3"
V4 = PROJECTS / "tradingagents-a1-retrieval-v4/data/a1-retrieval-v4"
V3_INDEX = V3 / "002/original-index/collection/chunks/storage.sqlite"
V4_INDEX = V4 / "001/bge-m3-index/collection/chunks_bge_m3_v4/storage.sqlite"
DATASET = SOURCE / "data/financebench/data/financebench_open_source.jsonl"
M3 = V4 / "001/models/baai-bge-m3"
RERANKER = V3 / "bge-reranker-v2-m3"
PIT = {"verified_known_at", "historical_proxy"}


def read_points(database):
    # Only repository-owned, previously frozen Qdrant databases are supported.
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        for (value,) in db.execute("SELECT point FROM points"):
            point = pickle.loads(value)
            if str(point.id) != point.payload["chunk_id"]:
                raise ValueError("Index ID differs from source chunk locator")
            yield str(point.id), point.payload, point.vector


def scope(rows, company, cutoff=CUTOFF, partition=PARTITION):
    stamp = cutoff.timestamp()
    allowed = [row for row in rows if row["ticker"] == company and row["dataset_partition"] == partition
               and row["pit_level"] in PIT and row["eligible_ts"] <= stamp]
    versions = {row["version_id"]: row for row in allowed}
    winners = {row["version_id"] for row in retain_versions(versions.values(),
               key=lambda r: (r["source_type"], r["document_id"], r["eligible_ts"]),
               verified=lambda r: r["pit_level"] == "verified_known_at")}
    return [row["chunk_id"] for row in allowed if row["version_id"] in winners]


class Search:
    def __init__(self, records, *, legacy=None):
        self.keys = sorted(records)
        self.rows = {key: records[key][0] for key in self.keys}
        self.positions = {key: i for i, key in enumerate(self.keys)}
        self.matrix = np.asarray([records[key][1] for key in self.keys], dtype=np.float32)
        if not np.isfinite(self.matrix).all() or not np.allclose(np.linalg.norm(self.matrix, axis=1), 1, atol=.002):
            raise ValueError("Nonfinite or non-normalized frozen index")
        self.legacy = legacy
        self.company_rows = defaultdict(list)
        self.counts = {}
        for key, row in self.rows.items():
            self.company_rows[row["ticker"]].append(row)
            self.counts[key] = Counter(terms(row["text"]))

    @classmethod
    def from_sqlite(cls, db):
        return cls({key: (row, vector) for key, row, vector in read_points(db)})

    @lru_cache(maxsize=8)
    def legal(self, company):
        keys = scope(self.company_rows[company], company)
        if not keys:
            return keys, np.empty((0, self.matrix.shape[1]), dtype=np.float32), {}, {}
        vectors = self.matrix[[self.positions[key] for key in keys]]
        postings = defaultdict(list)
        lengths = [sum(self.counts[key].values()) for key in keys]
        average = sum(lengths) / len(keys)
        scales = {key: 1.2 * (.25 + .75*length/average) for key, length in zip(keys, lengths, strict=True)}
        for key in keys:
            for term, count in self.counts[key].items():
                postings[term].append((key, count))
        return keys, vectors, postings, scales

    def lexical(self, query, company, k):
        keys, _, postings, scales = self.legal(company)
        scores = defaultdict(float)
        for term in sorted(set(terms(query))):
            hits = postings.get(term, ())
            idf = log(1 + (len(keys)-len(hits)+.5)/(len(hits)+.5))
            for key, count in hits:
                scores[key] += idf*2.2*count/(count+scales[key])
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]

    def dense(self, vector, company, k):
        keys, vectors, _, _ = self.legal(company)
        vector = np.asarray(vector, dtype=np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        scores = vectors @ vector
        return sorted(zip(keys, map(float, scores), strict=True), key=lambda pair: (-pair[1], pair[0]))[:k]

    def check(self, q, keys):
        scope_keys = set(self.legal(q["company"])[0])
        if not set(keys) <= scope_keys:
            raise ValueError("Prediction outside the pre-ranking scope")

    def retrieve(self, q, forms, encoder, *, expansion=True):
        started = perf_counter()
        self.legal(q["company"])
        scoped = perf_counter()
        lex = [self.lexical(query, q["company"], 50 if i == 0 else 35) for i, query in enumerate(forms)]
        lexical = perf_counter()
        vectors = encoder.encode(forms, query=True)
        encoded = perf_counter()
        dense = [self.dense(vector, q["company"], 50 if i == 0 else 35) for i, vector in enumerate(vectors)]
        end = perf_counter()
        return lex, dense, {"scope_ms": 1000*(scoped-started), "bm25_ms": 1000*(lexical-scoped),
                            "encode_ms": 1000*(encoded-lexical), "dense_ms": 1000*(end-encoded)}


def fused(groups, cap=None):
    scores = defaultdict(float)
    for weight, group in groups:
        for i, (key, _) in enumerate(group, 1):
            scores[key] += weight/(60+i)
    order = sorted(scores, key=lambda key: (-scores[key], key))
    return order[:cap], order


def union(bm, dense):
    return list(dict.fromkeys(key for key, _ in [*bm, *dense]))


def scope_fingerprint(database):
    return {"sha256": sha256(database), "bytes": database.stat().st_size}
