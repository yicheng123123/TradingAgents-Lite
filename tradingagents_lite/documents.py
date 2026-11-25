"""Parse local, permission-cleared sources into versioned retrieval chunks."""
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from uuid import NAMESPACE_URL, uuid5

from bs4 import BeautifulSoup
from pydantic import AnyHttpUrl, AwareDatetime, Field, model_validator
from pypdf import PdfReader

from .evidence import EvidenceStore, PITLevel, SourceType
from .schema import Contract


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Document(Contract):
    document_id: str
    ticker: str
    source_type: SourceType
    source_url: AnyHttpUrl
    eligible_at: AwareDatetime
    pit_level: PITLevel
    dataset_partition: str
    raw_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    accession: str | None = None
    title: str = ""
    format: str
    use_basis: str = Field(min_length=1)
    first_seen_at: AwareDatetime | None = None
    version_attestation_sha256: str | None = None
    version_attestation_source: str | None = None

    @model_validator(mode="after")
    def verify_version(self):
        if self.pit_level == "verified_known_at" and (
            self.first_seen_at is None or self.first_seen_at > self.eligible_at
            or self.version_attestation_sha256 != self.raw_hash or not self.version_attestation_source
        ):
            raise ValueError("Verified documents need attestation for the exact available version")
        return self

    @property
    def version_id(self) -> str:
        return digest(json.dumps(
            [self.dataset_partition, self.ticker, self.source_type, self.document_id, self.raw_hash,
             self.eligible_at.timestamp()], ensure_ascii=False).encode())


@dataclass(frozen=True)
class Block:
    text: str
    section: str
    locator: str
    page: int | None = None


def clean(text: str) -> str:
    return " ".join(text.split())


def parse_document(document: Document, raw: bytes, *, pdf_engine: str = "pypdf") -> list[Block]:
    """Locators refer to normalized block text; PDF pages are zero based."""
    if digest(raw) != document.raw_hash:
        raise ValueError("Document bytes differ from the frozen hash")
    if document.format == "pdf":
        blocks = []
        prefix = "pdf" if pdf_engine == "pypdf" else pdf_engine
        for i, page in enumerate(_pdf_pages(raw, pdf_engine)):
            paragraphs = re.split(r"\n\s*\n", page)
            blocks.extend(Block(clean(text), document.title, f"{prefix}:{i}:paragraph:{j}", i)
                          for j, text in enumerate(paragraphs))
        return blocks
    if document.format == "headline":
        return [Block(clean(raw.decode()), document.title, "headline")]
    if document.format == "text":
        return [Block(clean(text), document.title, f"paragraph:{i}")
                for i, text in enumerate(re.split(r"\n\s*\n", raw.decode()))]
    if document.format != "html":
        raise ValueError("Unsupported document format")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup.find_all(["script", "style", "ix:hidden", "noscript"]):
        tag.decompose()
    blocks = []
    section = document.title
    names = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div", "tr"}
    for i, tag in enumerate(soup.find_all(sorted(names))):
        if tag.name in {"p", "li", "div"} and (
            tag.find_parent("tr") or tag.find(sorted(names))
        ):
            continue
        if tag.name == "tr" and tag.find("tr"):
            continue
        body = clean(tag.get_text(" ", strip=True))
        if not body:
            continue
        style = tag.get("style", "").replace(" ", "").lower()
        bold = tag.find(["b", "strong"]) or "font-weight:bold" in style
        if tag.name.startswith("h") or (bold and len(body) <= 110 and tag.name != "tr"):
            section = body
        blocks.append(Block(body, section, f"html:{i}"))
    if not blocks:
        blocks.append(Block(clean(soup.get_text(" ", strip=True)), section, "html:root"))
    return blocks


def _pdf_pages(raw: bytes, engine: str):
    if engine == "pypdf":
        yield from (page.extract_text() or "" for page in PdfReader(BytesIO(raw)).pages)
        return
    if engine != "pdfium":
        raise ValueError("Unknown PDF parser")
    from pypdfium2 import PdfDocument

    with PdfDocument(raw) as document:
        for page in document:
            try:
                yield page.get_textpage().get_text_bounded().replace("\r\n", "\n")
            finally:
                page.close()


def chunk_document(document: Document, blocks: list[Block], *, size: int = 220, overlap: int = 32,
                   strategy: str = "structure", max_chunks: int = 4096) -> list[dict]:
    """Pack contiguous blocks by section and page, then slide bounded word windows."""
    if not 0 <= overlap < size <= 512 or not 1 <= max_chunks <= 10000:
        raise ValueError("Invalid chunk size, overlap or per-document limit")
    if strategy not in {"structure", "fixed"}:
        raise ValueError("Unknown chunk strategy")
    groups = []
    for block in blocks:
        if not block.text:
            continue
        key = (block.section if strategy == "structure" else "", block.page)
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        groups[-1][1].extend((word, block.locator, i) for i, word in enumerate(block.text.split()))
    result = []
    meta = document.model_dump(mode="json")
    meta.update(version_id=document.version_id, eligible_ts=document.eligible_at.timestamp())
    for (section, page), words in groups:
        for start in range(0, len(words), size - overlap):
            window = words[start:start + size]
            loc_start = [window[0][1], window[0][2]]
            loc_end = [window[-1][1], window[-1][2] + 1]
            page_ref = f"{loc_start[0].split(':', 1)[0]}:{page}"
            id_start = [page_ref, start] if page is not None else loc_start
            id_end = [page_ref, start + len(window)] if page is not None else loc_end
            key = json.dumps([document.version_id, strategy, size, overlap, id_start, id_end])
            result.append({**meta, "chunk_id": str(uuid5(NAMESPACE_URL, key)), "section": section,
                           "page": page, "start": loc_start, "end": loc_end,
                           "text": " ".join(word for word, _, _ in window)})
            if len(result) >= max_chunks or start + size >= len(words):
                break
        if len(result) >= max_chunks:
            break
    return result


def frozen_sources(evidence_path: Path, cache: Path, partition: str = "development"):
    """Replay frozen A0.5 versions without contacting SEC or news publishers."""
    for item in EvidenceStore.load(evidence_path).records:
        if item.eligible_at is None:
            continue
        title = item.source_metadata.get("title") or item.source_metadata.get("form", "")
        url = str(item.source_url)
        if item.source_type == "sec_filing":
            raw = (cache / f"{digest(url.encode())}.body").read_bytes()
            raw_hash = item.content_hash
            kind, basis = "html", "SEC accession domestic research copy"
        else:
            if not title:
                continue
            raw = title.encode()
            raw_hash = digest(raw)
            kind, basis = "headline", "Frozen publisher title metadata"
        yield Document(document_id=item.document_id, ticker=item.ticker, source_type=item.source_type,
                       source_url=item.source_url, eligible_at=item.eligible_at, pit_level=item.pit_level,
                       dataset_partition=partition, raw_hash=raw_hash, title=title, format=kind,
                       first_seen_at=item.first_seen_at,
                       version_attestation_sha256=item.source_metadata.get("version_attestation_sha256"),
                       version_attestation_source=item.source_metadata.get("version_attestation_source"),
                       accession=item.document_id if kind == "html" else None, use_basis=basis), raw
