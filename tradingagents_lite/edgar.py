"""SEC accession-level historical filings with conservative availability."""
from datetime import UTC, date, datetime
from html.parser import HTMLParser
import json
import re

from .data import NY
from .evidence import Evidence, next_session_proxy
from .evidence_http import CachedHTTP, SourceError

CIKS = {"AAPL": "0000320193", "MSFT": "0000789019"}
FORMS = {"10-K", "10-Q", "8-K"}
BASE = "https://data.sec.gov/submissions/"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/"
FIELDS = ("accessionNumber", "form", "reportDate", "filingDate", "acceptanceDateTime", "primaryDocument")


class VisibleText(HTMLParser):
    ignored = {"script", "style", "head", "ix:hidden", "ix:header", "noscript", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = []
        self.fragments = []

    def handle_starttag(self, tag, attrs):
        if tag in self.ignored:
            self.skip.append(tag)

    def handle_endtag(self, tag):
        if self.skip and self.skip[-1] == tag:
            self.skip.pop()

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.fragments.append(data.strip())


def filing_excerpt(raw: bytes, form: str) -> str:
    parser = VisibleText()
    parser.feed(raw.decode("utf-8", errors="replace"))
    text = " ".join(" ".join(parser.fragments).split())
    if len(text) < 80 or text.casefold().startswith(("your request originates", "request rate threshold")):
        raise SourceError("filing_has_no_usable_text")
    terms = ("Item 2.02", "Item 1.01", "Item 8.01", "Item 5.02") if form.startswith("8-K") else (
        "CONSOLIDATED STATEMENTS OF OPERATIONS", "Consolidated Statements of Income",
        "CONSOLIDATED BALANCE SHEETS")
    for term in terms:
        matches = list(re.finditer(re.escape(term), text, re.I))
        if matches:
            return text[matches[-1].start():matches[-1].start() + 1500]
    return text[:1500]


def parse_rows(table: dict, start: date, end: date):
    if not all(isinstance(table.get(key), list) for key in FIELDS):
        raise SourceError("invalid_edgar_submissions_columns")
    lengths = {len(table[key]) for key in FIELDS}
    if len(lengths) != 1:
        raise SourceError("inconsistent_edgar_submissions_columns")
    for values in zip(*(table[key] for key in FIELDS)):
        row = dict(zip(FIELDS, values))
        if row["form"].removesuffix("/A") in FORMS and start <= date.fromisoformat(row["filingDate"]) <= end:
            yield row


def collect_filings(http: CachedHTTP, ticker: str, start: date, end: date,
                    sessions: tuple[date, ...]) -> tuple[list[Evidence], dict]:
    records, requests, errors = [], [], []
    info = {"provider": "SEC EDGAR", "ticker": ticker, "source_type": "sec_filing",
            "status": "unavailable", "requested": [], "failures": errors}
    if ticker not in CIKS:
        errors.append("unconfigured_cik")
        return records, info
    cik = CIKS[ticker]

    def fetch(url):
        item = http.get(url)
        requests.append({"url": url, "sha256": item.content_hash, "bytes": len(item.body),
                         "ingested_at": item.ingested_at.isoformat(), "retention": "raw"})
        return item

    candidates = []
    try:
        submissions = json.loads(fetch(f"{BASE}CIK{cik}.json").body)
        candidates.extend(parse_rows(submissions["filings"]["recent"], start, end))
        for file in submissions["filings"]["files"]:
            if date.fromisoformat(file["filingFrom"]) > end or date.fromisoformat(file["filingTo"]) < start:
                continue
            name = file["name"]
            if not re.fullmatch(rf"CIK{cik}-submissions-[0-9]+\.json", name):
                raise SourceError("unexpected_edgar_segment_name")
            candidates.extend(parse_rows(json.loads(fetch(BASE + name).body), start, end))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        errors.append({"stage": "submissions", "code": str(exc) if isinstance(exc, SourceError) else
                       type(exc).__name__})

    for row in sorted({r["accessionNumber"]: r for r in candidates}.values(), key=lambda r: r["filingDate"]):
        accession = row["accessionNumber"]
        primary = row["primaryDocument"]
        try:
            if not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession) or not re.fullmatch(
                r"[A-Za-z0-9_.-]+\.html?", primary, re.I
            ):
                raise SourceError("invalid_accession_or_primary_document")
            url = f"{ARCHIVES}{int(cik)}/{accession.replace('-', '')}/{primary}"
            result = fetch(url)
            filing_date = date.fromisoformat(row["filingDate"])
            timestamp = row["acceptanceDateTime"]
            accepted = datetime.fromisoformat(timestamp) if timestamp else None
            if accepted is not None and accepted.utcoffset() is None:
                raise SourceError("acceptance_timestamp_has_no_timezone")
            anchor = max(filing_date, accepted.astimezone(NY).date()) if accepted else filing_date
            eligible = next_session_proxy(anchor, sessions)
            records.append(Evidence(
                ticker=ticker, source_type="sec_filing", document_id=accession, source_url=url,
                ingested_at=result.ingested_at, eligible_at=eligible,
                availability_basis="next_session_after_max_filing_date_and_acceptance_et_date",
                content_hash=result.content_hash, excerpt=filing_excerpt(result.body, row["form"]),
                pit_level="historical_proxy" if eligible else "unavailable",
                source_metadata={"publisher": "SEC EDGAR", "cik": cik, "form": row["form"],
                    "report_date": row["reportDate"] or None, "filing_date": row["filingDate"],
                    "accepted_at": accepted.astimezone(UTC).isoformat() if accepted else None, "public_available_at": None,
                    "hash_scope": "raw_accession_primary_document", "primary_document": primary}))
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"stage": "document", "accession": accession,
                           "code": str(exc) if isinstance(exc, SourceError) else type(exc).__name__})
    info["requested"] = requests
    info["status"] = ("partial" if errors else "available") if records else "unavailable"
    info["candidates"] = len({r["accessionNumber"] for r in candidates})
    info["records"] = len(records)
    return records, info
