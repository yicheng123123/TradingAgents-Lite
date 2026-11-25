"""Bounded lexical/dense retrieval with version-aware filters before ranking."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from math import log
import re
from time import perf_counter

from qdrant_client import models

from .data import close_available
from .documents import Document, chunk_document, frozen_sources, parse_document
from .evidence import LOOKBACK, retain_versions
from .vectors import VectorStore, date_limit, match


def terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def bm25(query: str, rows: list[dict], limit: int) -> list[tuple[str, float]]:
    counts = [Counter(terms(row["text"])) for row in rows]
    average = sum(sum(count.values()) for count in counts) / max(1, len(rows))
    if not average:
        return []
    ranked = []
    query_terms = sorted(set(terms(query)))
    idf = {}
    for term in query_terms:
        df = sum(term in count for count in counts)
        idf[term] = log(1 + (len(rows) - df + 0.5) / (df + 0.5))
    for row, count in zip(rows, counts, strict=True):
        scale = 1.2 * (0.25 + 0.75 * sum(count.values()) / average)
        score = sum(idf[t] * 2.2 * count[t] / (count[t] + scale)
                    for t in query_terms if t in count)
        if score:
            ranked.append((row["chunk_id"], score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))[:limit]


def rrf(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (key, _) in enumerate(ranking, 1):
            scores[key] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def ranked_trace(ranking):
    return [{"id": key, "rank": rank, "score": float(score)}
            for rank, (key, score) in enumerate(ranking, 1)]


@dataclass(frozen=True)
class RagConfig:
    mode: str = "R3"
    top_k: int = 4
    candidate_k: int = 24
    context_bytes: int = 3200
    dispatch: str = "agentic"

    def __post_init__(self):
        if (self.mode not in {"R1", "R2", "R3", "R4"}
                or not 1 <= self.top_k <= min(10, self.candidate_k)
                or not 1 <= self.candidate_k <= 100
                or not 256 <= self.context_bytes <= 6400
                or self.dispatch not in {"fixed", "agentic"}):
            raise ValueError("Invalid RAG config")


class Retriever:
    def __init__(self, store: VectorStore, reranker=None):
        self.store = store
        self.reranker = reranker
        self._scopes = {}

    def _candidates(self, ticker: str, cutoff: datetime, partition: str, pit: tuple[str, ...], lookback: bool):
        supported = ["historical_proxy", "verified_known_at"]
        if cutoff.utcoffset() is None or not pit or not set(pit) <= set(supported):
            raise ValueError("Invalid cutoff or PIT policy")
        key = (self.store.revision, ticker, cutoff, partition, pit, lookback)
        if key in self._scopes:
            return self._scopes[key]
        base = [match("ticker", ticker), match("dataset_partition", partition),
                date_limit("eligible_ts", cutoff.timestamp())]
        if lookback:
            base.append(models.Filter(should=[
                models.Filter(must=[match("source_type", source),
                    date_limit("eligible_ts", cutoff.timestamp(), minimum=(cutoff - timedelta(days=days)).timestamp())])
                for source, days in LOOKBACK.items()]))
        pit_gate = lambda levels: models.FieldCondition(key="pit_level", match=models.MatchAny(any=levels))
        candidates = [p.payload for p in self.store.scan(models.Filter(must=[*base, pit_gate(supported)]))]
        versions = {row["version_id"]: row for row in candidates}
        allowed = {
            row["version_id"] for row in retain_versions(
                versions.values(), key=lambda r: (r["source_type"], r["document_id"], r["eligible_ts"]),
                verified=lambda r: r["pit_level"] == "verified_known_at") if row["pit_level"] in pit}
        rows = {r["chunk_id"]: r for r in candidates if r["version_id"] in allowed}
        filters = models.Filter(must=[*base, pit_gate(list(pit)), models.FieldCondition(
            key="version_id", match=models.MatchAny(any=sorted(allowed)))])
        if len(self._scopes) >= 8:
            self._scopes.pop(next(iter(self._scopes)))
        self._scopes[key] = (rows, filters)
        return rows, filters

    def retrieve(self, query: str, *, ticker: str, cutoff: datetime, partition: str, mode: str,
                 top_k: int, candidate_k: int, pit=("historical_proxy", "verified_known_at"),
                 lookback=True) -> dict:
        if mode not in {"R1", "R2", "R3", "R4"} or not 1 <= top_k <= candidate_k <= 100:
            raise ValueError("Invalid retrieval mode or limits")
        if not query.strip() or len(query.encode()) > 2048:
            raise ValueError("Invalid retrieval query")
        started = perf_counter()
        rows, filters = self._candidates(ticker, cutoff, partition, tuple(pit), lookback)
        stages = {}
        if mode in {"R1", "R3", "R4"}:
            stages["bm25"] = bm25(query, list(rows.values()), candidate_k)
        if mode in {"R2", "R3", "R4"}:
            stages["dense"] = [(str(p.id), p.score) for p in self.store.search(query, filters, candidate_k)] if rows else []
        if mode in {"R3", "R4"}:
            stages["rrf"] = rrf([stages["bm25"], stages["dense"]])[:candidate_k]
        if mode == "R4":
            if self.reranker is None:
                raise ValueError("R4 requires a local cross encoder")
            keys = [key for key, _ in stages["rrf"]]
            scores = self.reranker.score(query, [rows[key]["text"] for key in keys]) if keys else []
            stages["rerank"] = sorted(zip(keys, scores, strict=True), key=lambda item: -item[1])
        final = next(reversed(stages.values()))[:top_k]
        return {"items": [rows[key] for key, _ in final], "stages": {
            name: ranked_trace(ranking) for name, ranking in stages.items()},
            "eligible_chunks": len(rows), "latency_ms": (perf_counter() - started) * 1000}


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class RagView:
    def __init__(self, retriever: Retriever, task, partition: str, config: RagConfig, *, provenance=False):
        self.retriever, self.task, self.partition, self.config = retriever, task, partition, config
        self.remaining = config.context_bytes
        self.provenance = provenance

    def search(self, query: str, limit: int | None = None) -> dict:
        cfg = self.config
        result = self.retriever.retrieve(query, ticker=self.task.symbol,
            cutoff=close_available(self.task.decision_date), partition=self.partition, mode=cfg.mode,
            top_k=min(limit or cfg.top_k, cfg.top_k), candidate_k=cfg.candidate_k)
        payload = {"status": "available", "items": []}
        for row in result["items"]:
            item = {"evidence_id": "chunk:" + row["chunk_id"], "source_type": row["source_type"],
                    "eligible_at": row["eligible_at"], "pit_level": row["pit_level"],
                    "document_id": row["document_id"], "page": row["page"], "excerpt": row["text"][:720]}
            if self.provenance:
                item.update({key: row[key] for key in (
                    "source_url", "version_id", "raw_hash", "ticker", "dataset_partition")})
            proposal = [*payload["items"], item]
            if len(compact_json({**payload, "items": proposal}).encode()) > self.remaining:
                break
            payload["items"] = proposal
        if not payload["items"]:
            payload["status"] = "unavailable"
        else:
            self.remaining -= len(compact_json(payload).encode())
        trace = {k: v for k, v in result.items() if k != "items"}
        return {**payload, "retrieval_trace": trace}


def manifest_sources(manifest):
    root = manifest.resolve().parent
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = (root / row.pop("path")).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Document path must stay inside the manifest directory")
        yield Document.model_validate(row), path.read_bytes()


def build_index(args):
    from .vectors import LocalEmbedding

    if args.output.exists():
        raise FileExistsError(args.output)
    sources = (manifest_sources(args.manifest) if args.manifest
               else frozen_sources(args.evidence, args.cache))
    encoder = LocalEmbedding(args.model_cache, device=args.device)
    documents = []
    with VectorStore(args.output, encoder, "chunks") as store:
        for doc, raw in sources:
            chunks = chunk_document(doc, parse_document(doc, raw), size=args.chunk_size,
                                    overlap=args.overlap, strategy=args.strategy)
            store.put(chunks, id_key="chunk_id", text_key="text")
            documents.append({"id": doc.document_id, "hash": doc.raw_hash, "chunks": len(chunks)})
    summary = {"documents": documents, "encoder": encoder.metadata, "strategy": args.strategy,
               "chunk_size": args.chunk_size, "overlap": args.overlap}
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
