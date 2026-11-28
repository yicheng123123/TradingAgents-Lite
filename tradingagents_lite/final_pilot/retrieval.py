"""Reuse the existing corpus scope and expose S3 as task-bound RAG."""
from dataclasses import dataclass
from time import perf_counter

from ..context_selection import _render_segments, _selection_segments
from ..data import close_available
from ..rag import Retriever, bm25, compact_json, rrf
from ..verifier import verify_citations
from .data import PARTITION

QUERY = "Recent financial results, risks and company outlook"
PIT = ("historical_proxy", "verified_known_at")
EVIDENCE_BYTES = 3200


class ProductionS3(Retriever):
    def __init__(self, store, reranker):
        super().__init__(store, reranker)

    def search(self, query, task):
        tick = perf_counter()
        rows, filters = self._candidates(task.symbol, close_available(task.decision_date), PARTITION, PIT, True)
        lexical = bm25(query, list(rows.values()), 50)
        dense = [(str(item.id), item.score) for item in self.store.search(query, filters, 50)] if rows else []
        pool = list(dict.fromkeys(key for key, _ in [*lexical, *dense]))
        if not set(pool) <= rows.keys():
            raise ValueError("Retriever candidate outside eligible task scope")
        scores, resources = self.reranker.score(query, [rows[key]["text"] for key in pool])
        order = sorted(zip(pool, scores, strict=True), key=lambda item: (-item[1], item[0]))
        return [rows[key] for key, _ in order[:5]], {
            "eligible": len(rows), "bm25": len(lexical), "dense": len(dense), "pool": len(pool),
            "reranker_ms": resources["milliseconds"], "latency_ms": (perf_counter()-tick)*1000,
            "peak_reranker_gpu_bytes": resources["peak_allocated_bytes"],
        }


def default_r3(search, query, task):
    result = search.retrieve(query, ticker=task.symbol, cutoff=close_available(task.decision_date),
                             partition=PARTITION, mode="R3", top_k=4, candidate_k=24)
    return result["items"], {"eligible": result["eligible_chunks"], "latency_ms":result["latency_ms"],
                             "reranker_ms":0, "pool":len(result["stages"]["rrf"])}


def select_evidence(rows, task, query, limit, *, c1=True):
    output = []
    seen = set()
    for row in rows[:limit]:
        key = "chunk:" + row["chunk_id"]
        if key in seen:
            continue
        seen.add(key)
        fields = ("source_type","eligible_at","pit_level","document_id","page","start","end",
                  "source_url","version_id","raw_hash","ticker","dataset_partition")
        item = {field:row[field] for field in fields}
        item["evidence_id"] = key
        if not verify_citations([key], {key:item}, {key}, task, partition=PARTITION)["passed"]:
            raise ValueError("Unverified RAG evidence version or timing")
        if c1:
            options = _selection_segments(f"{task.symbol} financial results risk outlook",query,row["text"])
            to_try = [(_render_segments(row["text"],option["segments"])) for option in options]
        else:
            to_try = [(row["text"][:720], None)]
        for text,segments in to_try:
            trial = {**item,"excerpt":text}
            if segments is not None:
                trial["token_spans"] = segments
            wire = {"status":"available","items":[*output,trial]}
            if len(compact_json(wire).encode()) <= EVIDENCE_BYTES:
                output.append(trial)
                break
    return output


@dataclass(frozen=True)
class PilotRag:
    searcher: object
    task: object
    revision: str
    context: str
    config = type("FixedRagConfig",(),{"dispatch":"fixed"})()

    def search(self, query, limit=None):
        if self.revision == "S3":
            rows, audit = self.searcher.search(query,self.task)
            limit = min(limit or 5, 5)
        elif self.revision == "R3":
            rows, audit = default_r3(self.searcher,query,self.task)
            limit = min(limit or 4, 4)
        else:
            raise ValueError("Unknown frozen retriever")
        evidence = select_evidence(rows,self.task,query,limit,c1=self.context=="C1-segments")
        return {"status":"available" if evidence else "unavailable","items":evidence,
                "retrieval_trace":{**audit,"mode":self.revision,"context":self.context,
                 "selected_count":len(evidence),"selected_ids":[item["evidence_id"] for item in evidence]}}
