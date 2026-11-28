"""Source-only chunk alternatives and Calibration evidence diagnostics."""
from collections import Counter, defaultdict
from io import BytesIO
import re
import resource
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from pypdf import PdfReader

from .data import sha256
from .documents import Document, chunk_document, clean
from .rag_benchmark import CUTOFF, PARTITION
from .vectors import LocalEmbedding, VectorStore

CHUNKS = {
    "legacy": {"size": 220, "overlap": 32, "parser": "pypdf plain", "strategy": "structure"},
    "small": {"size": 128, "overlap": 24, "parser": "pypdf plain", "strategy": "structure"},
    "large": {"size": 320, "overlap": 48, "parser": "pypdf plain", "strategy": "structure"},
    "table": {"size": 220, "overlap": 32, "parser": "pypdf layout", "strategy": "whole lines"},
}
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
UNIT = re.compile(r"\b(?:thousands?|millions?|billions?|percent|usd|eur|gbp)\b|[%$€£]", re.I)
VALUE = re.compile(r"(?<![a-zA-Z])[-(]?\d[\d,.]*(?:\s?%)?\)?")


def normalize(text):
    return "".join(re.findall(r"[a-z0-9]+", text.casefold()))


def numeric_line(text):
    values = VALUE.findall(text)
    return (len(re.findall(r"[A-Za-z]{2,}", text)) >= 2
            and any(not YEAR.fullmatch(value) for value in values)
            and len(normalize(text)) >= 24)


class Audit:
    def __init__(self, source, historical):
        self.pdf_root = source / "data/financebench-pdfs"
        self.historical = historical
        self.readers = {}
        self.pages = {}

    def page(self, doc, page):
        key = (doc, page)
        if key not in self.pages:
            if doc not in self.readers:
                self.readers[doc] = PdfReader(self.pdf_root / f"{doc}.pdf")
            self.pages[key] = (self.readers[doc].pages[page].extract_text() or "").splitlines()
        return self.pages[key]

    def __call__(self, q, metadata, lexical, dense):
        from .retrieval_v2 import BASE, DEPTHS, fuse, page_key, page_recall

        grouped = defaultdict(list)
        for key, row in metadata.items():
            grouped[page_key(row)].append((key, normalize(row["text"])))
        gold = q["gold"]
        result = {
            "gold_pages": len(gold), "legal_gold_pages": len(gold & grouped.keys()),
            "candidate_recall": {}, "line_status": [], "lines": [],
        }
        for k in DEPTHS:
            bm, de = [key for key, _ in lexical[:k]], [key for key, _ in dense[:k]]
            result["candidate_recall"][str(k)] = {
                "bm25": page_recall(bm, metadata, gold),
                "dense": page_recall(de, metadata, gold),
                "union": page_recall(bm + de, metadata, gold),
            }
        top = fuse(lexical, dense, metadata, BASE)[:5]
        result["rrf_gold_recall_at_5"] = page_recall(top, metadata, gold)
        result["duplicate_page_chunks_at_5"] = len(top) - len({page_key(metadata[k]) for k in top})
        for label, ranking in (("bm25", lexical), ("dense", dense)):
            result[f"{label}_first_gold_rank"] = next(
                (i for i, (key, _) in enumerate(ranking, 1) if page_key(metadata[key]) in gold), None)
        for doc, pg in sorted(gold):
            anchor = [normalize(e["evidence_text"]) for e in q["calibration_evidence"]
                      if (e["doc_name"], e["evidence_page_num"]) == (doc, pg)]
            page = self.page(doc, pg)
            # Exact normalized span match is a conservative line proxy, not manual ground truth.
            matched = [(i, s, normalize(s)) for i, s in enumerate(page)
                       if numeric_line(s) and any(normalize(s) in a for a in anchor)]
            chunks = grouped[(doc, pg)]
            line_keys = []
            metadata_counts = Counter()
            for i, line, target in matched:
                containing = {key: text for key, text in chunks if target in text}
                nearby = " ".join(page[max(0, i - 4):i + 1])
                years = set(YEAR.findall(nearby))
                units = {normalize(match) for match in UNIT.findall(nearby) if normalize(match)}
                metadata_counts["years_observed"] += bool(years)
                metadata_counts["years_colocated"] += bool(years) and any(
                    all(year in text for year in years) for text in containing.values())
                metadata_counts["units_observed"] += bool(units)
                metadata_counts["units_colocated"] += bool(units) and any(
                    all(unit in text for unit in units) for text in containing.values())
                line_keys.append({"chunk_ids": sorted(containing)})
            result["lines"].extend(line_keys)
            result["line_status"].append({
                "identifiable_numeric_lines": len(matched),
                "complete_in_legacy_chunk": sum(bool(line["chunk_ids"]) for line in line_keys),
                **metadata_counts,
            })
        return result


def table_blocks(document, reader):
    """Use source layout lines, with separately recorded same-page header locators."""
    meta = document.model_dump(mode="json")
    meta.update(version_id=document.version_id, eligible_ts=document.eligible_at.timestamp())
    for page_no, page in enumerate(reader.pages):
        layout = page.extract_text(extraction_mode="layout") or ""
        # Fall back to plain extraction only on a page without usable layout text.
        if not layout.strip():
            layout = page.extract_text() or ""
        lines = []
        for line_no, raw in enumerate(layout.splitlines()):
            if not raw.strip():
                continue
            words = clean(raw).split()
            for offset in range(0, len(words), 220):
                # An exceptionally long extracted line has no reliable row boundaries.
                lines.append((line_no, offset, words[offset:offset + 220]))
        start = 0
        headers = []
        end = 0
        while start < len(lines):
            size = 0
            end = start
            selected_headers = [
                (pos, kind) for pos, kind in reversed(headers)
                if 0 < start - pos <= 12
            ]
            selected_headers = list(dict((kind, pos) for pos, kind in reversed(selected_headers)).values())
            head_size = sum(len(lines[pos][2]) for pos in selected_headers)
            while end < len(lines) and size + len(lines[end][2]) <= 220:
                size += len(lines[end][2])
                end += 1
            while selected_headers and size + head_size > 220:
                removed = selected_headers.pop(0)
                head_size -= len(lines[removed][2])
            span = lines[start:end]
            # Keep source lines and contextual header lines separately traceable.
            copied = sorted(selected_headers)
            render = lambda rows: "\n".join(" ".join(words) for _, _, words in rows)
            body = render(span)
            context = render([lines[pos] for pos in copied])
            first, last = span[0], span[-1]
            key = f"{document.version_id}:table-v1:{page_no}:{first[:2]}:{last[:2]}"
            yield {
                **meta, "chunk_id": str(uuid5(NAMESPACE_URL, key)), "section": document.title,
                "page": page_no, "start": [f"pypdf-layout:{page_no}:line:{first[0]}", first[1]],
                "end": [f"pypdf-layout:{page_no}:line:{last[0]}", last[1] + len(last[2])],
                "text": f"{context}\n{body}" if context else body, "parsed_body": body,
                "retrieval_header_lines": [lines[pos][0] for pos in copied],
            }
            if end == len(lines):
                break
            nxt, overlap = end, 0
            while nxt > start + 1 and overlap + len(lines[nxt - 1][2]) <= 32:
                nxt -= 1
                overlap += len(lines[nxt][2])
            for pos in range(start, nxt):
                s = " ".join(lines[pos][2])
                if len(lines[pos][2]) <= 20 and not numeric_line(s):
                    if len(YEAR.findall(s)) >= 2:
                        headers.append((pos, "columns"))
                    if UNIT.search(s):
                        headers.append((pos, "units"))
                    if re.search(r"\b(?:statements? of|balance sheets?|years? ended|quarters? ended)\b", s, re.I):
                        headers.append((pos, "title"))
            headers = [(pos, kind) for pos, kind in headers if nxt - pos <= 12]
            start = nxt


def build_variant(source, run, historical, variant, companies=None):
    from .documents import parse_document
    from .retrieval_v2 import index_bytes, read_rows, tree_hashes

    config = CHUNKS[variant]
    data = source / "data/financebench/data/financebench_document_information.jsonl"
    details = {r["doc_name"]: r for r in read_rows(data)}
    root = run / ("chunk-index-cal" if companies is not None else "chunk-index-full")
    target = root / variant
    if target.exists():
        raise FileExistsError(target)
    # A single local cache allows identical document windows to share vectors.
    import sqlite3
    model = run / "chunk-embedding-cache"
    if not model.exists():
        model.mkdir()
        (model / "models").symlink_to((source / "data/local-models/models").resolve())
        original_cache = source / "data/local-models/embeddings.sqlite"
        if original_cache.is_file():
            with sqlite3.connect(original_cache.as_uri() + "?mode=ro", uri=True) as original:
                with sqlite3.connect(model / "embeddings.sqlite") as local:
                    original.backup(local)
    started = perf_counter()
    encoder = LocalEmbedding(model, allow_download=False)
    documents, skipped, total = 0, [], 0
    with VectorStore(target, encoder, "chunks") as store:
        for name, info in sorted(historical["pdfs"].items()):
            raw_info = details[name]
            if info["status"] != "indexed" or (companies is not None and raw_info["company"] not in companies):
                continue
            raw = (source / "data/financebench-pdfs" / f"{name}.pdf").read_bytes()
            doc = Document(document_id=name, ticker=raw_info["company"], source_type="sec_filing",
                           source_url=raw_info["doc_link"], eligible_at=CUTOFF, pit_level="historical_proxy",
                           dataset_partition=PARTITION, raw_hash=info["sha256"], title=name, format="pdf",
                           use_basis="Patronus public research copy; local noncommercial evaluation")
            try:
                if variant == "table":
                    rows = list(table_blocks(doc, PdfReader(BytesIO(raw))))
                else:
                    rows = chunk_document(doc, parse_document(doc, raw), size=config["size"], overlap=config["overlap"])
                if not rows:
                    skipped.append({"id": name, "reason": "empty_parse"})
                    continue
            except (ValueError, RuntimeError, OSError) as exc:
                skipped.append({"id": name, "reason": type(exc).__name__})
                continue
            store.put(rows, id_key="chunk_id", text_key="text")
            documents += 1
            total += len(rows)
            print(f"{variant}: {documents} documents, {total} chunks", flush=True)
    return target, {"config": config, "index_bytes": index_bytes(target), "documents": documents,
                    "chunks": total, "unindexed": skipped, "sha256": tree_hashes(target),
                    "build_seconds": perf_counter() - started,
                    "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def run_chunk_stage(source, run, historical, questions, best, legacy, diagnostics):
    from .retrieval_v2 import RULES, measure, quality_ok, selection_key

    pages = [page for row in diagnostics for page in row["line_status"]]
    identifiable = [p for p in pages if p["identifiable_numeric_lines"]]
    misses = sum(p["identifiable_numeric_lines"] - p["complete_in_legacy_chunk"] for p in identifiable)
    total = sum(p["identifiable_numeric_lines"] for p in identifiable)
    trigger = len(identifiable) >= 10 and misses / total >= .2
    result = {"triggered": trigger, "identifiable_pages": len(identifiable),
              "audited_pages": len(pages), "missing_complete_lines": misses, "identifiable_lines": total,
              "selected": "legacy", "protocol": CHUNKS}
    if not trigger:
        return result
    companies = {q["company"] for q in questions}
    candidates = []
    for variant in ("small", "large", "table"):
        index, build = build_variant(source, run, historical, variant, companies)
        _, scores = measure(source, run, index, best, questions, f"cal-chunk-{variant}")
        candidates.append({"label": variant, "build": build, **scores})
    # Calibration-only. Legacy must be beaten without worsening the original safeguards.
    matched = [c for c in candidates if quality_ok(c, legacy)
               and c["latency"]["hot"]["total_ms"]["mean"] <= RULES["hot_latency_ratio_limit"] *
               legacy["latency"]["hot"]["total_ms"]["mean"]]
    if matched:
        result["selected"] = min(matched, key=lambda c: (selection_key(c, best), c["label"]))["label"]
    result["candidates"] = candidates
    return result
