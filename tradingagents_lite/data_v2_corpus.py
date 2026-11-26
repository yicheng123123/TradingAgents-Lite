"""Compile V2 research evidence and a separate A1-compatible local index."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from datetime import date
from functools import partial
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
import resource
from time import perf_counter

from .data import close_available, sha256 as file_hash
from .data_v2_market import ISSUERS, ROOT, trading_sessions, write_json
from .data_v2_sec import write_rows
from .documents import Document, chunk_document, parse_document
from .evidence import Evidence, EvidenceStore, next_session_proxy
from .vectors import LocalEmbedding, VectorStore


PARTITION = "expansion-v2"
RESEARCH_THROUGH = close_available(date(2024, 12, 31))
MAX_CHUNKS = 72


def records(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def issuer_news(path: Path):
    sessions = trading_sessions()
    for row in json.loads(path.read_text()):
        if row["ticker"] not in ISSUERS or not row["source_url"].startswith("https://"):
            raise ValueError("Invalid V2 newsroom source")
        published = date.fromisoformat(row["published_date"])
        eligible = next_session_proxy(published, sessions)
        if eligible is None:
            continue
        digest = sha256(json.dumps({key: row[key] for key in (
            "ticker", "title", "published_date", "source_url")}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        item = Evidence(ticker=row["ticker"], source_type="news", document_id="issuer:" + digest,
            source_url=row["source_url"], ingested_at=row["observed_at"], eligible_at=eligible,
            availability_basis="next_session_after_publisher_calendar_date",
            content_hash=digest, pit_level="historical_proxy", excerpt=row["title"],
            source_metadata={"title": row["title"], "publisher": "issuer IR or newsroom",
                "published_date": row["published_date"], "hash_scope": "title_date_url_projection",
                "use_scope": row["use_scope"]})
        yield item


def compile_corpus(root: Path, newsroom: Path, legacy_evidence: Path | None = None,
                   *, version="data-expansion-v2-corpus-001"):
    folder = root / "evidence"
    if (folder / "source_manifest.json").exists():
        raise FileExistsError("Research corpus already published. Compile in a new --root.")
    sec = root / "sec"
    sec_manifest = json.loads((sec / "source_manifest.json").read_text())
    if file_hash(sec / "evidence.jsonl") != sec_manifest["evidence_sha256"]:
        raise ValueError("V2 SEC evidence version mismatch")
    sec_records = [Evidence.model_validate(row) for row in records(sec / "evidence.jsonl")]
    issuer_records = list(issuer_news(newsroom))
    if legacy_evidence is not None:
        issuer_records.extend(item for item in EvidenceStore.load(legacy_evidence).records
                              if item.source_type == "news")
    # Corpus releases after the 2024 research boundary remain outside the interactive corpus.
    available = [item for item in [*sec_records, *issuer_records] if item.eligible_at is not None]
    visible = [item for item in available if item.eligible_at <= RESEARCH_THROUGH]
    store = EvidenceStore(visible)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "evidence.jsonl"
    write_rows(path, [item.model_dump(mode="json") for item in store.records])
    report = {
        "version": version, "dataset_partition": PARTITION,
        "through": RESEARCH_THROUGH.isoformat(), "market_calendar_manifest_sha256": file_hash(root / "market/manifest.json"),
        "evidence_sha256": file_hash(path), "sec_manifest_sha256": file_hash(sec / "source_manifest.json"),
        "issuer_index_sha256": file_hash(newsroom), "count": len(store.records),
        "by_source": dict(Counter(item.source_type for item in store.records)),
        "by_pit": dict(Counter(item.pit_level for item in store.records)),
        "issuer_news_tickers": sorted({item.ticker for item in issuer_records}),
        "sources_after_research_boundary": len(available) - len(visible),
        "unavailable_sources": len(sec_records) + len(issuer_records) - len(available),
    }
    write_json(folder / "source_manifest.json", report)
    return report


def _document(item, cache):
    url = str(item.source_url)
    original = cache / (sha256(url.encode()).hexdigest() + ".body")
    if original.is_file():
        raw, kind, basis = original.read_bytes(), "html", "SEC accession research copy"
        raw_hash = item.content_hash
    elif item.source_type == "news":
        raw, kind, basis = item.excerpt.encode(), "headline", "Publisher title and URL metadata"
        raw_hash = sha256(raw).hexdigest()
    else:
        raise FileNotFoundError("SEC primary body missing from the local resume cache")
    return Document(document_id=item.document_id, ticker=item.ticker, source_type=item.source_type,
                    source_url=item.source_url, eligible_at=item.eligible_at, pit_level=item.pit_level,
                    dataset_partition=PARTITION, raw_hash=raw_hash,
                    accession=item.source_metadata.get("accession"), format=kind, use_basis=basis,
                    title=item.source_metadata.get("title", ""),
                    first_seen_at=item.first_seen_at,
                    version_attestation_sha256=item.source_metadata.get("version_attestation_sha256"),
                    version_attestation_source=item.source_metadata.get("version_attestation_source")), raw


def _prepare_chunks(item, *, cache: Path, cap: int):
    try:
        doc, raw = _document(item, cache)
        chunks = chunk_document(doc, parse_document(doc, raw), strategy="structure")
    except (ValueError, OSError) as error:
        return None, [], {"id": item.evidence_id, "code": type(error).__name__}
    if not chunks:
        return None, [], {"id": item.evidence_id, "code": "no_chunks"}
    original_count = len(chunks)
    if original_count > cap:
        chunks = [chunks[i * (original_count - 1) // (cap - 1)] for i in range(cap)]
    row = {"version": doc.version_id, "ticker": doc.ticker, "sector": ISSUERS[doc.ticker][0],
           "raw_sha256": doc.raw_hash, "chunks": len(chunks), "before_sampling": original_count}
    return row, chunks, None


def _open_worker(models):
    global _worker_encoder
    _worker_encoder = LocalEmbedding(models, threads=2, allow_download=False)


def _embed_chunks(item, *, cache: Path, cap: int):
    row, chunks, error = _prepare_chunks(item, cache=cache, cap=cap)
    if chunks:
        # Commit vectors to the shared content-addressed cache before the single writer reads them.
        _worker_encoder.encode([part["text"] for part in chunks])
    return row, chunks, error


def build_rag(root: Path, output: Path, models: Path, *, version="rag-expansion-v2-001",
              max_chunks: int = MAX_CHUNKS, workers: int = 1, on_progress=None) -> dict:
    started = perf_counter()
    if output.exists():
        raise FileExistsError("V2 RAG index already exists")
    if not 2 <= max_chunks <= MAX_CHUNKS or not 1 <= workers <= 4:
        raise ValueError("V2 indexing limits require 2–72 chunks and 1–4 workers")
    evidence = EvidenceStore.load(root / "evidence")
    used, errors = [], []
    cache = root / "sec/cache"
    pool = (ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_open_worker, initargs=(models,))
            if workers > 1 else nullcontext())
    encoder = LocalEmbedding(models, threads=1 if workers > 1 else 4, allow_download=False)
    with pool as executor, VectorStore(output, encoder, "chunks") as binding:
        prepared = (executor.map(partial(_embed_chunks, cache=cache, cap=max_chunks), evidence.records)
                    if executor is not None else map(partial(_prepare_chunks, cache=cache, cap=max_chunks), evidence.records))
        total_chunks = 0
        for step, (row, chunks, error) in enumerate(prepared, 1):
            if error is not None:
                errors.append(error)
            else:
                binding.put(chunks, id_key="chunk_id", text_key="text")
                used.append(row)
                total_chunks += len(chunks)
            if on_progress is not None and (step % 25 == 0 or step == len(evidence.records)):
                on_progress({"documents_processed": step, "documents_total": len(evidence.records),
                             "chunks": total_chunks, "failures": len(errors),
                             "elapsed_seconds": round(perf_counter() - started, 1)})
    summary = {
        "version": version, "partition": PARTITION, "encoder": encoder.metadata,
        "source_manifest_sha256": file_hash(root / "evidence/source_manifest.json"),
        "index_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss if workers > 1 else None,
        "document_count": len(used), "chunk_count": sum(row["chunks"] for row in used),
        "tickers": dict(Counter(row["ticker"] for row in used)),
        "sectors": dict(Counter(row["sector"] for row in used)),
        "documents": used, "failures": errors, "max_chunks_per_document": max_chunks, "workers": workers,
        "retrieval": ["BM25", "local dense cosine", "RRF"],
    }
    write_json(output / "manifest.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("compile", "index"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--news", type=Path, default=Path("tasks/data-expansion-v2/issuer-news.json"))
    parser.add_argument("--legacy-evidence", type=Path, default=Path("data/financial-evidence-dev-003"))
    parser.add_argument("--models", type=Path, default=Path("data/local-models"))
    parser.add_argument("--version", help="Immutable version label for the new corpus or index")
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    parameters = {"version": args.version} if args.version else {}
    report = (compile_corpus(args.root, args.news, args.legacy_evidence, **parameters) if args.stage == "compile"
              else build_rag(args.root, args.root / "rag", args.models, max_chunks=args.max_chunks,
                             workers=args.workers, on_progress=(lambda row: print(json.dumps(row), flush=True))
                             if args.progress else None, **parameters))
    print({name: report[name] for name in ("count", "by_source", "by_pit") if name in report}
          or {name: report[name] for name in ("document_count", "chunk_count", "elapsed_seconds", "peak_rss_kib")})
