"""Bounded SEC V2 acquisition and cache replay with accession provenance."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import UTC, date, datetime
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from time import perf_counter
from urllib.request import build_opener, HTTPRedirectHandler
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from .data import NY, sha256 as file_hash
from .data_v2_market import ISSUERS, ROOT, atomic_write, trading_sessions, write_json
from .edgar import ARCHIVES, BASE, FIELDS, filing_excerpt
from .evidence import Evidence, next_session_proxy
from .evidence_http import CachedHTTP, SourceError
from .fundamentals import inline_company_facts, normalize


FACTS = "https://data.sec.gov/api/xbrl/companyfacts/"
MAPPING = "https://www.sec.gov/files/company_tickers_exchange.json"
FIRST_YEAR, LAST_YEAR = 2021, 2025
PERIODIC = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
MAX_PRIMARY = 500
MAX_DOCUMENTS = 600
PROXY_BASIS = "next_session_after_max_filing_date_and_acceptance_et_date"
ALLOWED_DOC = re.compile(r"[a-zA-Z0-9_.-]+\.html?", re.I)
WARMUP_START = "2020-08-01"


@contextmanager
def collector_lock(path: Path = Path("data/.data-expansion-v2-sec.lock")):
    """One collector per checkout, including collectors with a different output root."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SourceError("sec_collector_already_running") from exc
        yield


def progress_event(stage, ticker, done, total, started, cache, receipts, errors):
    elapsed = perf_counter() - started
    return {"sec_progress": stage, "company": ticker, "completed": done, "total": total,
            "remaining": total - done, "percent": round(100 * done / total, 1) if total else 100.0,
            "eta_seconds": round(elapsed * (total - done) / done, 1) if done else None,
            "downloaded": cache.downloads, "cache_hits": cache.cache_hits,
            "documents_saved": len(receipts), "failures": len(errors), "network_stopped": cache.blocked}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class SecCache:
    def __init__(self, root: Path, legacy: Path, offline: bool, reuse: tuple[Path, ...] = (),
                 *, fresh_metadata: bool = False):
        agent = os.getenv("SEC_USER_AGENT", "").strip()
        if not offline and (("\r" in agent or "\n" in agent)
                            or not re.fullmatch(r".+\s+[^\s@]+@[^\s@]+\.[^\s@]+", agent)):
            raise ValueError("Set SEC_USER_AGENT to a real project name and authorized contact email")
        # Serial requests, at least 0.45s apart, with a stop on 403 or 429.
        kwargs = {"sec_user_agent": agent or "TradingAgents-Lite offline cache replay"}
        self.current = CachedHTTP(root, offline=offline, opener=build_opener(NoRedirect()).open,
                                  sec_interval=.45, stop_on_throttle=True, **kwargs)
        self.fallback = [CachedHTTP(path, offline=True, **kwargs)
                         for path in dict.fromkeys((*reuse, legacy)) if path.resolve() != root.resolve()]
        self.blocked = False
        self.fresh_metadata = fresh_metadata
        self.used = {}
        self.downloads = self.cache_hits = 0

    def has(self, url):
        key = sha256(url.encode()).hexdigest()
        return any((item.root / f"{key}.json").is_file() for item in (self.current, *self.fallback))

    def get(self, url, cap=30_000_000):
        key = sha256(url.encode()).hexdigest()
        cached = (self.current.root / f"{key}.json").is_file()
        refresh = self.fresh_metadata and not url.startswith(ARCHIVES)
        source = None if cached or refresh else next((item for item in self.fallback
                                           if (item.root / f"{key}.json").is_file()), None)
        if source is not None:
            payload = source.get(url, cap)
            self.current.root.mkdir(parents=True, exist_ok=True)
            atomic_write(self.current.root / f"{key}.body", payload.body)
            write_json(self.current.root / f"{key}.json", {"url": url,
                "ingested_at": payload.ingested_at.isoformat(), "content_hash": payload.content_hash})
            cached = True
        else:
            if self.blocked and not cached:
                raise SourceError("sec_network_stopped_after_access_limit")
            try:
                payload = self.current.get(url, cap)
            except SourceError as exc:
                if str(exc).startswith(("http_403", "http_429")):
                    self.blocked = True
                raise
        self.cache_hits += int(cached)
        self.downloads += int(not cached)
        self.used[url] = {"bytes": len(payload.body), "sha256": payload.content_hash,
                          "retrieved_at": payload.ingested_at.isoformat()}
        return payload


def periodic_rows(cache, ticker, cik, errors, *, first="2019-01-01", last="2025-12-31"):
    cik_text = f"{cik:010}"
    submission = json.loads(cache.get(f"{BASE}CIK{cik_text}.json").body)
    if int(submission["cik"]) != cik or ticker not in submission.get("tickers", []) and ticker != "XOM":
        raise SourceError("issuer_mapping_mismatch")
    files = submission["filings"]
    tables = [files["recent"]]
    for entry in files["files"]:
        if entry["filingTo"] < first or entry["filingFrom"] > last:
            continue
        name = entry["name"]
        if not re.fullmatch(rf"CIK{cik_text}-submissions-[0-9]+\.json", name):
            raise SourceError("invalid_segment_filename")
        try:
            tables.append(json.loads(cache.get(BASE + name).body))
        except (ValueError, OSError) as exc:
            errors.append({"ticker": ticker, "stage": "submissions_segment", "segment": name,
                           "reason": str(exc) if isinstance(exc, SourceError) else type(exc).__name__})
    rows = {}
    for table in tables:
        fields = (*FIELDS, "items")
        if len({len(table[field]) for field in fields}) != 1:
            raise SourceError("inconsistent_submissions_columns")
        for values in zip(*(table[field] for field in fields), strict=True):
            row = dict(zip(fields, values, strict=True))
            if (row["form"] not in PERIODIC | {"8-K", "8-K/A"}
                    or not first <= row["filingDate"] <= last):
                continue
            if (not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", row["accessionNumber"])
                    or not ALLOWED_DOC.fullmatch(row["primaryDocument"])):
                continue
            rows[row["accessionNumber"]] = row
    return rows


def dated_filing(ticker, cik, row, sessions):
    accession = row["accessionNumber"]
    day = date.fromisoformat(row["filingDate"])
    accepted = datetime.fromisoformat(row["acceptanceDateTime"]) if row["acceptanceDateTime"] else None
    if accepted is not None and accepted.utcoffset() is None:
        accepted = None
    anchor = max(day, accepted.astimezone(NY).date()) if accepted else day
    eligible = next_session_proxy(anchor, sessions)
    return {
        "ticker": ticker, "cik": f"{cik:010}", "accession": accession, "form": row["form"],
        "filing_date": str(day), "report_date": row["reportDate"] or None,
        "acceptance_datetime": accepted.astimezone(UTC).isoformat() if accepted else None,
        "first_seen_at": None, "eligible_at": eligible.isoformat() if eligible else None,
        "source_url": f"{ARCHIVES}{cik}/{accession.replace('-', '')}/{row['primaryDocument']}",
        "pit_level": "historical_proxy" if eligible else "unavailable", "items": row["items"],
    }


def choose_documents(filings, *, warmup=False):
    periodic = []
    earnings = {}
    for row in sorted(filings, key=lambda r: (r["filing_date"], r["accession"])):
        year = int(row["filing_date"][:4])
        if not FIRST_YEAR <= year <= LAST_YEAR:
            if warmup and WARMUP_START <= row["filing_date"] < "2021-01-01" and row["form"] in PERIODIC:
                periodic.append(row)
            continue
        if row["form"] in PERIODIC:
            periodic.append(row)
        elif row["form"] == "8-K" and "2.02" in row["items"].replace(" ", "").split(","):
            earnings.setdefault((row["ticker"], year), row)
    periodic.sort(key=lambda row: (row["form"].endswith("/A"), row["filing_date"], row["accession"]))
    periodic = periodic[:MAX_PRIMARY]
    slots = (MAX_DOCUMENTS - len(periodic)) // 2
    return [*periodic, *list(earnings.values())[:slots]]


def exhibit_reference(raw: bytes, filing: dict, *, index=False):
    primary = urlsplit(filing["source_url"])
    directory = primary.path.rsplit("/", 1)[0] + "/"
    soup = BeautifulSoup(raw, "html.parser")
    for row in soup.select("tr"):
        label = " ".join(row.stripped_strings)
        fields = {cell.get_text(" ", strip=True).upper() for cell in row.find_all("td")}
        if (index and "EX-99.1" not in fields) or (not index and not re.match(r"^99\s*\.\s*1(?=\s|$)", label)):
            continue
        for link in row.find_all("a", href=True):
            href = link["href"].strip()
            # SEC iX viewer links can wrap a document path in a doc parameter.
            if href.startswith("/ix?doc="):
                href = href.removeprefix("/ix?doc=")
            target = urlsplit(urljoin(filing["source_url"], href))
            name = target.path.removeprefix(directory)
            if (target.scheme == "https" and target.netloc == "www.sec.gov"
                    and target.path.startswith(directory) and ALLOWED_DOC.fullmatch(name)
                    and not target.query and not target.fragment and target.path != primary.path):
                return target.geturl()
    return None


def _record(filing, payload, *, exhibit=False):
    role = "news" if exhibit else "sec_filing"
    form = "EX-99.1" if exhibit else filing["form"]
    title = form
    if exhibit:
        soup = BeautifulSoup(payload.body[:65536], "html.parser")
        title = " ".join(soup.title.stripped_strings) if soup.title else "Exhibit 99.1"
        title = " ".join(title.split())[:200] or "Exhibit 99.1"
    row = {**filing, "form": form, "parent_form": filing["form"] if exhibit else None,
           "source_type": role, "source_url": payload.url, "content_hash": payload.content_hash,
           "retrieved_at": payload.ingested_at.isoformat(), "bytes": len(payload.body),
           "format": "html", "title": title, "published_at": None,
           "use_basis": "SEC public filing; local research copy; redistribution excluded"}
    evidence = None
    if row["eligible_at"]:
        evidence = Evidence(ticker=row["ticker"], source_type=role,
            document_id=row["accession"] + (":99.1" if exhibit else ""),
            source_url=payload.url, ingested_at=payload.ingested_at, eligible_at=row["eligible_at"],
            availability_basis=PROXY_BASIS, content_hash=payload.content_hash,
            pit_level="historical_proxy", excerpt=filing_excerpt(payload.body, form),
            source_metadata={"title": title, "publisher": "SEC EDGAR", "cik": row["cik"], "form": form,
                "filing_date": row["filing_date"], "acceptance_datetime": row["acceptance_datetime"],
                "accession": row["accession"]})
    return row, evidence


def write_rows(path, rows):
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                               for row in rows).encode())


def build_sec(root: Path, legacy: Path, *, offline: bool, reuse: tuple[Path, ...] = (),
              warmup=False, on_progress=None, through: date | None = None):
    clock = perf_counter()
    output = root / "sec"
    if (output / "source_manifest.json").exists():
        raise FileExistsError("SEC snapshot already published. Use a new --root with --reuse-cache.")
    fundamentals = output / "fundamentals"
    fundamentals.mkdir(parents=True, exist_ok=True)
    cache = SecCache(output / "cache", legacy, offline, reuse, fresh_metadata=through is not None)
    errors, receipts, evidence, cases, issuers = [], [], [], [], {}
    inline = defaultdict(list)
    inline_filings = defaultdict(dict)
    last_session = through or date(2026, 1, 16)
    sessions = trading_sessions(date(2019, 1, 1), last_session)
    def emit(stage, ticker, done, total, started):
        if on_progress is not None:
            on_progress(progress_event(stage, ticker, done, total, started, cache, receipts, errors))

    for i, (ticker, (_, _, cik)) in enumerate(ISSUERS.items()):
        emit("issuer metadata", ticker, i, len(ISSUERS), clock)
        try:
            rows = (periodic_rows(cache, ticker, cik, errors, last=str(through)) if through else
                    periodic_rows(cache, ticker, cik, errors))
            classified = [dated_filing(ticker, cik, row, sessions) for row in rows.values()]
            cases.extend(classified)
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"ticker": ticker, "stage": "submissions", "reason": str(exc) if isinstance(exc, SourceError) else type(exc).__name__})
            continue
        try:
            raw = cache.get(f"{FACTS}CIK{cik:010}.json")
            filings = {row["accession"]: row for row in classified if row["form"] in PERIODIC}
            facts = normalize(json.loads(raw.body), cik, filings)
            path = fundamentals / f"{ticker}.json"
            write_json(path, {"facts": facts, "filings": filings})
            issuers[ticker] = {"cik": f"{cik:010}", "rows": len(facts), "sha256": file_hash(path),
                               "company_facts_sha256": raw.content_hash, "source_url": raw.url}
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"ticker": ticker, "stage": "company_facts", "reason": str(exc) if isinstance(exc, SourceError) else type(exc).__name__})
    emit("issuer metadata", None, len(ISSUERS), len(ISSUERS), clock)
    if through:
        selected = sorted((row for row in cases if row["filing_date"] >= "2026-01-01"
                           and (row["form"] in PERIODIC or row["form"] in {"8-K", "8-K/A"}
                                and "2.02" in row["items"].replace(" ", "").split(","))),
                          key=lambda row: (row["form"] not in PERIODIC, row["filing_date"], row["accession"]))[:MAX_PRIMARY]
    else:
        selected = choose_documents(cases, warmup=warmup)
    reserved = sum(filing["form"] == "8-K" for filing in selected)
    seen = {row["accession"] for row in selected}
    supplemental = [row for row in cases if row["accession"] not in seen
                    and row["filing_date"] >= ("2026-01-01" if through else WARMUP_START if warmup else "2021-01-01")
                    and cache.has(row["source_url"])]
    selected.extend(supplemental[:max(0, MAX_DOCUMENTS - len(selected) - reserved)])
    documents_clock = perf_counter()
    for i, filing in enumerate(selected):
        emit("filing documents", filing["ticker"], i, len(selected), documents_clock)
        try:
            raw = cache.get(filing["source_url"])
            row, item = _record(filing, raw)
            receipts.append(row)
            if item:
                evidence.append(item)
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"ticker": filing["ticker"], "accession": filing["accession"], "stage": "primary",
                           "reason": str(exc) if isinstance(exc, SourceError) else type(exc).__name__})
            continue
        ticker = filing["ticker"]
        if filing["form"] in PERIODIC and ticker not in issuers:
            facts = inline_company_facts(raw.body, row)
            if facts:
                inline[ticker].extend(facts)
                inline_filings[ticker][row["accession"]] = row
        if filing["form"] != "8-K" or len(receipts) >= MAX_DOCUMENTS:
            continue
        url = exhibit_reference(raw.body, filing)
        if url is None:
            try:
                index = filing["source_url"].rsplit("/", 1)[0] + "/" + filing["accession"] + "-index.html"
                url = exhibit_reference(cache.get(index).body, filing, index=True)
            except (ValueError, KeyError, OSError) as exc:
                errors.append({"ticker": ticker, "accession": filing["accession"], "stage": "exhibit index",
                               "reason": str(exc) if isinstance(exc, SourceError) else type(exc).__name__})
            if url is None:
                errors.append({"ticker": ticker, "accession": filing["accession"],
                               "stage": "exhibit", "reason": "no_verified_99_1_link"})
                continue
        try:
            row, item = _record(filing, cache.get(url), exhibit=True)
            receipts.append(row)
            if item:
                evidence.append(item)
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"ticker": filing["ticker"], "accession": filing["accession"], "stage": "exhibit",
                           "reason": str(exc) if isinstance(exc, SourceError) else type(exc).__name__})
    emit("filing documents", None, len(selected), len(selected), documents_clock)
    inline_issuers = {}
    for ticker, facts in inline.items():
        path = fundamentals / f"{ticker}.json"
        write_json(path, {"facts": facts, "filings": inline_filings[ticker]})
        inline_issuers[ticker] = {"cik": f"{ISSUERS[ticker][2]:010}", "rows": len(facts),
                                  "sha256": file_hash(path), "source_type": "cached SEC accession iXBRL"}
    try:
        mapping = json.loads(cache.get(MAPPING).body)
        current = {row[2]: row[0] for row in mapping["data"]}
    except (SourceError, ValueError, OSError, KeyError):
        current = {}
    write_rows(output / "filings.jsonl", cases)
    write_rows(output / "documents.jsonl", receipts)
    write_rows(output / "evidence.jsonl", [item.model_dump(mode="json") for item in evidence])
    write_json(fundamentals / "manifest.json", {"version": "sec-fundamentals-v3" if through else "sec-fundamentals-v2",
                                               "issuers": issuers | inline_issuers})
    summary = {
        "version": "sec-accession-v3" if through else "sec-accession-v2",
        **({"through": str(through), "document_selection": "2026 periodic filings and earnings 8-K/99.1"} if through else {}),
        "mode": "offline_cache" if offline else "declared_agent",
        "market_sessions": "XNYS next session 16:15 ET", "selected_primary": len(selected),
        "ticker_rows": sorted({row["ticker"] for row in cases}), "ticker_mapping": {
            ticker: {"historical_cik": f"{cik:010}", "current_cik": current.get(ticker)}
            for ticker, (_, _, cik) in ISSUERS.items()},
        "document_count": len(receipts), "forms": dict(Counter(row["form"] for row in receipts)),
        "documents_by_ticker": dict(Counter(row["ticker"] for row in receipts)),
        "pit": dict(Counter(row["pit_level"] for row in receipts)),
        "distinct_document_hashes": len({row["content_hash"] for row in receipts}),
        "news_tickers": sorted({item.ticker for item in evidence if item.source_type == "news"}),
        "company_facts": issuers, "inline_xbrl_fallback": inline_issuers, "failures": errors, "source_files": cache.used,
        "filings_sha256": file_hash(output / "filings.jsonl"),
        "documents_sha256": file_hash(output / "documents.jsonl"),
        "evidence_sha256": file_hash(output / "evidence.jsonl"),
        "cache_bytes_used": sum(row["bytes"] for row in cache.used.values()),
        "elapsed_seconds": round(perf_counter() - clock, 3), "rate_limit_stop": cache.blocked,
        "downloads": cache.downloads, "cache_hits": cache.cache_hits,
        "warmup_documents": sum(row["filing_date"] < "2021-01-01" for row in receipts),
    }
    write_json(output / "source_manifest.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--legacy-cache", type=Path, default=Path("data/evidence-cache-001"))
    parser.add_argument("--reuse-cache", action="append", type=Path, default=[])
    parser.add_argument("--warmup", action="store_true", help="Acquire periodic filings from late 2020 as input context")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--through", type=date.fromisoformat, help="Build fresh 2026 SEC metadata through this date")
    args = parser.parse_args()
    with collector_lock():
        summary = build_sec(args.root, args.legacy_cache, offline=args.offline, reuse=tuple(args.reuse_cache),
                            warmup=args.warmup, through=args.through,
                            on_progress=(lambda row: print(json.dumps(row), flush=True)) if args.progress else None)
    print({key: summary[key] for key in ("document_count", "forms", "pit", "news_tickers", "cache_bytes_used")})
