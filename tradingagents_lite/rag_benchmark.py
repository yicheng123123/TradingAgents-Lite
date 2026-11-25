"""Reproducible retrieval evaluation against FinanceBench's public page anchors."""
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
from math import log2
from pathlib import Path
import resource
from statistics import mean
import subprocess
from time import perf_counter

from .data import sha256
from .documents import Document, chunk_document, digest, parse_document
from .rag import Retriever
from .vectors import LocalEmbedding, LocalReranker, VectorStore


PARTITION = "financebench_public"
CUTOFF = datetime(2026, 1, 1, tzinfo=UTC)
SOURCE = "https://github.com/patronus-ai/financebench"


def split(company: str) -> str:
    """Freeze company-grouped splits before relevance assessment or ranking."""
    return "calibration" if int(digest(company.casefold().encode())[:8], 16) % 5 == 0 else "test"


def relevance(ranked: list, relevant: set, k: int = 10) -> dict:
    seen = set()
    hits = []
    for key in ranked[:k]:
        hits.append(int(key in relevant and key not in seen))
        seen.add(key)
    ideal = sum(1 / log2(i + 2) for i in range(min(k, len(relevant))))
    return {**{f"recall@{n}": sum(hits[:n]) / len(relevant) for n in (1, 5, 10)},
            "mrr@10": next((1 / i for i, hit in enumerate(hits, 1) if hit), 0),
            "ndcg@10": sum(hit / log2(i + 2) for i, hit in enumerate(hits)) / ideal}


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _page_match(anchor: str, text: str) -> bool:
    import re

    def norm(value):
        return "".join(re.findall(r"[a-z0-9]+", value.casefold()))

    target, page = norm(anchor), norm(text)
    if len(target) < 24:
        return False
    if target in page:
        return True
    # PDF tables often interleave columns. Require distributed contiguous anchors.
    windows = [target[i:i+32] for i in range(0, len(target) - 31, 32)]
    return bool(windows) and sum(part in page for part in windows) >= len(windows) * .65


def benchmark(repo: Path, workspace: Path, output: Path, *, pdf_root: Path | None = None, device: str = "cpu",
              modes=("R1", "R2", "R3", "R4"), companies: set[str] | None = None,
              pdf_engine: str = "pypdf") -> dict:
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    workspace.mkdir(parents=True, exist_ok=True)
    questions_path = repo / "data/financebench_open_source.jsonl"
    metadata_path = repo / "data/financebench_document_information.jsonl"
    questions = _read(questions_path)
    metadata = {r["doc_name"]: r for r in _read(metadata_path)}
    names = {q["doc_name"] for q in questions}
    names.update(e["doc_name"] for q in questions for e in q["evidence"])
    if companies is not None:
        names = {name for name in names if metadata.get(name, {}).get("company") in companies}
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    encoder = LocalEmbedding(workspace, device=device) if set(modes) != {"R1"} else None
    try:
        reranker = LocalReranker(workspace) if "R4" in modes else None
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"R4 unavailable: {type(exc).__name__}", flush=True)
        reranker = None
    skipped_modes = ["R4"] if "R4" in modes and reranker is None else []
    modes = [mode for mode in modes if mode not in skipped_modes]
    source_rows, blocks_by_name, indexed_pages = {}, {}, {}
    index = workspace / f"financebench-index-{output.name}"
    if index.exists():
        raise FileExistsError(index)
    with VectorStore(index, encoder, "chunks") as store:
        for name in sorted(names):
            path = (pdf_root or repo / "pdfs") / f"{name}.pdf"
            raw_info = metadata.get(name)
            if raw_info is None or not path.is_file():
                source_rows[name] = {"status": "missing_pdf_or_metadata"}
                continue
            raw = path.read_bytes()
            doc = Document(document_id=name, ticker=raw_info["company"], source_type="sec_filing",
                           source_url=raw_info["doc_link"], eligible_at=CUTOFF,
                           pit_level="historical_proxy", dataset_partition=PARTITION,
                           raw_hash=digest(raw), title=name, format="pdf",
                           use_basis="Patronus public research copy; local noncommercial evaluation")
            try:
                blocks = parse_document(doc, raw, pdf_engine=pdf_engine)
                chunks = chunk_document(doc, blocks)
            except Exception as exc:
                source_rows[name] = {"status": "pdf_parse_error", "error_type": type(exc).__name__}
                continue
            store.put(chunks, id_key="chunk_id", text_key="text")
            pages = defaultdict(list)
            for block in blocks:
                pages[block.page].append(block.text)
            blocks_by_name[name] = {page: " ".join(parts) for page, parts in pages.items()}
            indexed_pages[name] = {chunk["page"] for chunk in chunks}
            source_rows[name] = {"status": "indexed", "sha256": doc.raw_hash, "pages": len(pages),
                                 "chunks": len(chunks), "source_url": str(doc.source_url)}
            print(f"Indexed {name} ({len(chunks)} chunks)", flush=True)
        retriever = Retriever(store, reranker)
        metrics = defaultdict(list)
        exclusions = Counter()
        splits = Counter()
        mapped = Counter()
        with (output / "queries.jsonl").open("x") as stream:
            for q in questions:
                group = split(q["company"])
                splits[group] += 1
                if companies is not None and q["company"] not in companies:
                    reason = "outside_pilot_company_scope"
                    exclusions[reason] += 1
                    stream.write(json.dumps({"id": q["financebench_id"], "split": group,
                                             "exclusions": [reason]}) + "\n")
                    continue
                anchors = set()
                reasons = set()
                if not q["evidence"]:
                    reasons.add("no_evidence")
                for e in q["evidence"]:
                    name = e["doc_name"]
                    pages = blocks_by_name.get(name)
                    page = e["evidence_page_num"]
                    if pages is None:
                        reasons.add("missing_or_unreadable_pdf")
                    elif metadata[name]["company"] != q["company"]:
                        reasons.add("evidence_outside_company_scope")
                    elif not isinstance(page, int) or page not in pages:
                        reasons.add("invalid_page_index")
                    elif not _page_match(e["evidence_text"], pages[page]):
                        reasons.add("evidence_text_unaligned")
                    elif page not in indexed_pages[name]:
                        reasons.add("evidence_page_without_chunk")
                    else:
                        anchors.add((name, page))
                row = {"id": q["financebench_id"], "split": group, "exclusions": sorted(reasons)}
                if reasons:
                    exclusions.update(reasons)
                else:
                    mapped[group] += 1
                    row["anchors"] = sorted(anchors)
                    row["modes"] = {}
                    for mode in modes:
                        found = retriever.retrieve(q["question"], ticker=q["company"], cutoff=CUTOFF,
                            partition=PARTITION, mode=mode, top_k=10, candidate_k=24, lookback=False)
                        pages = [(r["document_id"], r["page"]) for r in found["items"]]
                        doc_ids = list(dict.fromkeys(doc_id for doc_id, _ in pages))
                        scores = {"page_proxy": relevance(pages, anchors),
                                  "document": relevance(doc_ids, {doc_id for doc_id, _ in anchors})}
                        ms = found["latency_ms"]
                        metrics[group, mode].append((scores, ms))
                        row["modes"][mode] = {"stages": found["stages"], "pages": pages,
                                              "latency_ms": ms, "scores": scores}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    aggregates = {}
    for (group, mode), rows in sorted(metrics.items()):
        aggregates.setdefault(group, {})[mode] = {
            "n": len(rows), "mean_latency_ms": mean(ms for _, ms in rows),
            **{scope: {metric: mean(scores[scope][metric] for scores, _ in rows)
                       for metric in rows[0][0][scope]} for scope in ("page_proxy", "document")}}
    result = {"source": SOURCE, "revision": commit,
              "license_basis": "CC BY-NC 4.0 per Patronus public dataset card; source PDFs used locally",
              "license_url": "https://huggingface.co/datasets/PatronusAI/financebench",
              "pdf_engine": pdf_engine,
              "companies": sorted(companies) if companies is not None else None,
              "public_questions": len(questions), "split_counts": dict(splits), "mapped_counts": dict(mapped),
              "exclusion_reasons_nonexclusive": dict(exclusions), "pdfs": source_rows,
              "questions_sha256": sha256(questions_path), "documents_sha256": sha256(metadata_path),
              "embedding": encoder.metadata if encoder else None,
              "reranker": reranker.metadata if reranker else None,
              "skipped_modes": skipped_modes, "metrics": aggregates,
              "index_bytes": sum(p.stat().st_size for p in index.rglob("*") if p.is_file()),
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "elapsed_seconds": perf_counter() - started, "glm_calls": 0}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {k: v for k, v in result.items() if k != "pdfs"}
