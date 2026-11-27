"""Project the pinned FinanceBench corpus into disjoint question, oracle and scoring inputs."""
import json
from pathlib import Path
import subprocess

from pydantic import Field

from . import b1, rag_benchmark
from .data import sha256
from .data_v2_market import atomic_write, write_json
from .schema import Contract


SOURCE = Path("data/financebench")
PDFS = Path("data/financebench-pdfs")
RETRIEVAL_ARCHIVE = Path("results/a1-rag-local-001")
WORKSPACE = Path("data/local-models")
INDEX = WORKSPACE / "financebench-index-financebench-run-002"
SMOKE_IDS = ("financebench_id_04735", "financebench_id_07507")
BUNDLE = Path("data/final-financebench-v1")
GOLD = Path("data/final-financebench-gold-v1.jsonl")


class Case(Contract):
    id: str = Field(pattern=r"^financebench_id_[0-9]+$")
    company: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=2000)
    split: str
    unavailable: tuple[str, ...] = ()


class OracleExcerpt(Contract):
    evidence_id: str
    document_id: str
    page: int = Field(ge=0)
    excerpt: str = Field(min_length=1)


class OracleCase(Contract):
    id: str
    items: tuple[OracleExcerpt, ...]


class GoldCase(Contract):
    id: str
    answer: str = Field(min_length=1)
    anchors: tuple[tuple[str, int], ...]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path, records):
    atomic_write(path, "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                               + "\n" for row in records).encode())


def index_fingerprint(path):
    if not (path / "meta.json").is_file() or not (path / "chunks-encoder.json").is_file():
        raise ValueError("Existing FinanceBench Qdrant index is required")
    files = {file.relative_to(path).as_posix(): sha256(file) for file in sorted(path.rglob("*"))
             if file.is_file() and file.name != ".lock"}
    if not any(name.startswith("collection/chunks/") for name in files):
        raise ValueError("FinanceBench Qdrant collection is missing")
    return b1._json_hash(files)


def prepare(out=BUNDLE, gold_out=GOLD, *, source=SOURCE, pdfs=PDFS, archive=RETRIEVAL_ARCHIVE, index=INDEX):
    if out.exists() or gold_out.exists() or gold_out.resolve().is_relative_to(out.resolve()):
        raise ValueError("Use a new bundle and a separate new scoring path")
    summary_path = archive / "public-four-mode-summary.json"
    queries_path = archive / "public-four-mode-queries.jsonl"
    summary = json.loads(summary_path.read_text())
    questions = source / "data/financebench_open_source.jsonl"
    documents = source / "data/financebench_document_information.jsonl"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if (revision != summary["revision"] or sha256(questions) != summary["questions_sha256"]
            or sha256(documents) != summary["documents_sha256"]):
        raise ValueError("FinanceBench source no longer matches the existing Retrieval Benchmark")
    mapped = {row["id"]: row for row in rows(queries_path)}
    docs = {row["doc_name"]: row for row in rows(documents)}
    approved, availability = {}, {}
    for name, receipt in summary["pdfs"].items():
        path = pdfs / f"{name}.pdf"
        if receipt["status"] != "indexed":
            availability[name] = receipt["status"]
        elif not path.is_file():
            availability[name] = "missing_pdf"
        elif sha256(path) != receipt["sha256"]:
            availability[name] = "pdf_hash_mismatch"
        else:
            approved[name] = {"sha256": receipt["sha256"], "company": docs[name]["company"]}
    samples, oracle, gold = [], [], []
    for raw in rows(questions):
        case_id = raw["financebench_id"]
        source_row = mapped[case_id]
        split = rag_benchmark.split(raw["company"])
        if split != source_row["split"]:
            raise ValueError("Company split differs from the frozen Retrieval Benchmark")
        anchors = {(item["doc_name"], item["evidence_page_num"]) for item in raw["evidence"]}
        approved_anchors = {tuple(item) for item in source_row.get("anchors", [])}
        blockers = set(source_row["exclusions"])
        if not anchors <= approved_anchors:
            blockers.add("official_anchor_not_aligned")
        for name in {raw["doc_name"], *(name for name, _ in anchors)}:
            if name not in approved:
                blockers.add(availability.get(name, "document_not_indexed"))
            elif approved[name]["company"] != raw["company"]:
                blockers.add("document_outside_company")
        safe = Case(id=case_id, company=raw["company"], question=raw["question"],
                    split=split, unavailable=sorted(blockers))
        items = []
        if not blockers:
            for i, e in enumerate(raw["evidence"], 1):
                text = e["evidence_text"].strip()
                if text:
                    items.append(OracleExcerpt(evidence_id=f"oracle-{i}", document_id=e["doc_name"],
                                               page=e["evidence_page_num"], excerpt=text))
        samples.append(safe.model_dump(mode="json"))
        oracle.append(OracleCase(id=case_id, items=items).model_dump(mode="json"))
        gold.append(GoldCase(id=case_id, answer=raw["answer"].strip(),
                             anchors=sorted((name, page) for name, page in anchors
                                            if type(page) is int and page >= 0)).model_dump(mode="json"))
    split_counts = {group: sum(row["split"] == group for row in samples) for group in ("calibration", "test")}
    if (split_counts != {"calibration": 51, "test": 99}
            or len({row["id"] for row in samples}) != len(samples)
            or set(mapped) != {row["id"] for row in samples}
            or not set(SMOKE_IDS) <= {row["id"] for row in samples
                                     if row["split"] == "calibration" and not row["unavailable"]}):
        raise ValueError("FinanceBench cases differ from the frozen 51/99 split or Smoke availability")
    identity = index_fingerprint(index)
    out.mkdir(parents=True, exist_ok=False)
    gold_out.parent.mkdir(parents=True, exist_ok=True)
    write_rows(out / "questions.jsonl", samples)
    write_rows(out / "oracle.jsonl", oracle)
    with gold_out.open("x", encoding="utf-8") as stream:
        for row in gold:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    manifest = {
        "source_revision": revision, "source_questions_sha256": sha256(questions),
        "source_documents_sha256": sha256(documents), "retrieval_summary_sha256": sha256(summary_path),
        "retrieval_queries_sha256": sha256(queries_path), "index_sha256": identity,
        "embedding": summary["embedding"], "split_counts": split_counts, "smoke_ids": list(SMOKE_IDS),
        "questions_sha256": sha256(out / "questions.jsonl"), "oracle_sha256": sha256(out / "oracle.jsonl"),
        "gold_sha256": sha256(gold_out), "approved_documents": approved, "unavailable_documents": availability,
        "case_unavailability": {row["id"]: row["unavailable"] for row in samples if row["unavailable"]},
    }
    write_json(out / "manifest.json", manifest)
    return manifest
