"""Conservative statement-page evidence from the original PDF layout."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from hashlib import sha256
from pathlib import Path
import re
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from ..data import sha256 as file_hash
from .engine import SOURCE, Search, read_points
from .features import NUMBER, YEAR, find_metrics, statements

VERSION = "v6-pypdf-layout-v2"
PERIOD = re.compile(r"\b(?:years? ended|months? ended|as of|fiscal year)\b", re.I)
SCALE = re.compile(r"\b(?:in (?:thousands|millions|billions)|USD|EUR|GBP)\b|[$£€]", re.I)
NUMERIC_CELL = re.compile(
    r"(?:[$£€]\s*)?\(\s*[$£€]?\s*[−+\-]?\d[\d,]*(?:\.\d+)?\s*%?\s*\)"
    r"|[$£€]?\s*[−+\-]?\d[\d,]*(?:\.\d+)?\s*%?|[—–-]"
)


def fields(line):
    return [(match.group().strip(), match.start(), match.end())
            for match in re.finditer(r"\S(?:.*?\S)?(?=\s{2,}|$)", line)]


def positioned_lines(layout):
    positions = []
    offset = 0
    for line in layout.splitlines(keepends=True):
        without_end = line.rstrip("\r\n")
        if without_end.strip():
            positions.append((offset, offset+len(without_end), without_end))
        offset += len(line)
    return positions


def header_binding(header, row, *, tolerance=9):
    """Require a one-to-one ordered alignment of cell right edges in layout text."""
    header_cells = [(value, start, end) for value, start, end in fields(header) if YEAR.search(value)]
    row_cells = [(value, start, end) for value, start, end in fields(row)[1:] if NUMERIC_CELL.fullmatch(value)]
    years = [YEAR.findall(value) for value,_,_ in header_cells]
    if (len(row_cells) < 2 or len(row_cells) != len(header_cells)
            or any(len(found)!=1 for found in years) or len({y[0] for y in years}) != len(years)):
        return []
    for index, ((_,_,header_end), (_,_,value_end)) in enumerate(zip(header_cells,row_cells,strict=True)):
        offsets = [abs(value_end-end) for _,_,end in header_cells]
        if offsets[index] > tolerance or offsets.count(min(offsets))!=1 or offsets[index]!=min(offsets):
            return []
    return [{"period": year[0], "header_columns": [a,b], "row_columns": [c,d]}
            for year,(_,a,b),(_,c,d) in zip(years,header_cells,row_cells,strict=True)]


def source_units(layout):
    lines = positioned_lines(layout)
    if not lines:
        return []
    heading = next((i for i,(_,_,text) in enumerate(lines[:30]) if statements(text) and len(text.strip())<135),None)
    if heading is None:
        return []
    heading_line = lines[heading]
    page_statement = sorted(statements(heading_line[2]))
    rows = []
    for index,(start,end,line) in enumerate(lines):
        cells = fields(line)
        if len(cells)<3 or not find_metrics(cells[0][0]):
            continue
        if len(line.split())>120 or sum(bool(NUMERIC_CELL.fullmatch(value)) for value,_,_ in cells[1:])<2:
            continue
        prior = lines[max(heading,index-12):index]
        header = None
        binding = []
        for hs,he,htext in reversed(prior):
            check = header_binding(htext,line)
            if check and (PERIOD.search(htext) or any(PERIOD.search(ln) for _,_,ln in prior)):
                header = [hs,he]
                binding = check
                break
        scale = next(([a,b] for a,b,ln in reversed(prior) if SCALE.search(ln)
                      and re.search(r"\b(?:thousands|millions|billions)\b",ln,re.I)), None)
        meta = {"statement":page_statement, "header_span":header, "column_binding":binding,
                "unit_span":scale, "binding_status":"verified_layout_alignment" if binding else "uncertain"}
        rows.append(([start,end],"row",meta,index))
    units = []
    # Bounded line windows keep whitespace and original source order.
    offset = 0
    while offset<len(lines):
        end = offset+1
        while end<len(lines) and len(layout[lines[offset][0]:lines[end][1]].split())<=190:
            end+=1
        units.append(([lines[offset][0],lines[end-1][1]],"paragraph",{"statement":page_statement},None))
        offset = max(offset+1,end-3) if end<len(lines) else end
    units.extend(rows)
    for (left,_,meta,index),(right,_,_,next_index) in zip(rows,rows[1:]):
        if next_index-index<=4 and len(layout[left[0]:right[1]].split())<=100:
            units.append(([left[0],right[1]],"block",{"statement":page_statement,
                          "binding_status":"uncertain","header_span":meta["header_span"]},None))
    return units


def extract_doc(job):
    pdf_root, document_id, template, expected, suggested_pages = job
    path = Path(pdf_root)/(document_id+".pdf")
    if file_hash(path)!=expected or template["raw_hash"]!=expected:
        raise ValueError("PDF bytes differ from frozen citation")
    pdf = PdfReader(path)
    result, stats = [], Counter()
    stats["pdf_pages"] = len(pdf.pages)
    for page_no in suggested_pages:
        page = pdf.pages[page_no]
        stats["pages_reparsed"] += 1
        plain = page.extract_text() or ""
        if not any(statements(line) and len(line.strip())<135 for line in plain.splitlines()[:35]):
            continue
        layout = page.extract_text(extraction_mode="layout") or ""
        if not layout:
            continue
        units = source_units(layout)
        if not units:
            continue
        stats["statement_pages"] += 1
        for span,kind,annotation,_ in units:
            start,end = span
            original = layout[start:end]
            if not original.strip():
                continue
            identifier = str(uuid5(NAMESPACE_URL,f"{VERSION}:{template['version_id']}:{page_no}:{start}:{end}:{kind}"))
            spans = [span]
            if kind=="row":
                spans.extend(s for s in (annotation["header_span"], annotation["unit_span"]) if s)
            if kind=="block" and annotation["header_span"]:
                spans.append(annotation["header_span"])
            spans = sorted({tuple(s) for s in spans})
            pieces = [layout[a:b] for a,b in spans]
            result.append({
                **template, "chunk_id":identifier, "page":page_no,
                "start":[f"pdf:{page_no}:layout",start],"end":[f"pdf:{page_no}:layout",end],
                "text":"\n[… omitted …]\n".join(pieces),
                "unit_type":kind,"parser":VERSION,"source_spans":[list(s) for s in spans],
                "source_span_hashes":[sha256(value.encode()).hexdigest() for value in pieces],
                "page_layout_sha256":sha256(layout.encode()).hexdigest(),**annotation,
            })
            stats[kind]+=1
            stats["verified_bindings"] += bool(annotation.get("column_binding"))
    return result,dict(stats)


def structure_jobs(legacy, known, companies=None):
    documents = {row["document_id"]:row for row in legacy.rows.values()}
    if len(documents)!=83 or not set(documents)<=set(known):
        raise ValueError("Structure run must preserve the frozen 83-PDF manifest")
    pages = defaultdict(set)
    for row in legacy.rows.values():
        if statements(row["text"]):
            pages[row["document_id"]].add(row["page"])
    return [(str(SOURCE/"data/financebench-pdfs"),doc,template,known[doc],sorted(pages[doc]))
            for doc,template in sorted(documents.items()) if companies is None or template["ticker"] in companies]



def build_index(jobs, work, encoder):
    index = work/"structure-index"
    index.mkdir(exist_ok=False)
    stats = Counter()
    peak = 0
    start = perf_counter()
    collection = "financial_layout"
    with closing(QdrantClient(path=str(index))) as client:
        client.create_collection(collection,vectors_config=models.VectorParams(size=encoder.metadata["dimension"],distance=models.Distance.COSINE))
        with ProcessPoolExecutor(max_workers=4) as workers:
            for rows,audit in workers.map(extract_doc,jobs,chunksize=1):
                stats.update(audit)
                stats["documents"]+=1
                for i in range(0,len(rows),128):
                    batch=rows[i:i+128]
                    lengths=encoder.token_lengths([row["text"] for row in batch])
                    stats["over_512_tokens"] += sum(length>512 for length in lengths)
                    stats["max_input_tokens"] = max(stats["max_input_tokens"],max(lengths,default=0))
                    vectors, usage = encoder.encode_with_stats([row["text"] for row in batch])
                    peak = max(peak, usage["peak_allocated_bytes"] or 0)
                    client.upsert(collection,[models.PointStruct(id=row["chunk_id"],vector=vector,payload=row)
                                              for row,vector in zip(batch,vectors,strict=True)],wait=True)
                print("Structure PDFs",stats["documents"],"chunks",stats["paragraph"]+stats["row"]+stats["block"],flush=True)
    database = index/"collection/financial_layout/storage.sqlite"
    stats["indexed"] = stats["paragraph"]+stats["row"]+stats["block"]
    stats["bytes"] = sum(file.stat().st_size for file in index.rglob("*") if file.is_file())
    stats["elapsed_seconds"] = perf_counter()-start
    stats["peak_allocated_bytes"] = peak
    return database,dict(stats)
