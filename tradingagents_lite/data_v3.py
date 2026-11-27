"""Freeze V3 historical task inputs, with holdout access sealed at execution."""
import argparse
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256 as hash_bytes
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen

from .data import HistoricalData, NY, close_available, read_tasks, sha256
from .data_v2_market import ISSUERS, atomic_write, snapshot_market, trading_sessions, write_json
from .data_v2_sec import FACTS, SecCache, collector_lock, dated_filing, periodic_rows, write_rows
from .evidence import Evidence, EvidenceStore, next_session_proxy
from .evidence_http import CachedHTTP
from .fundamentals import FundamentalsView, normalize
from .news import collect_apple
from .schema import Task
from .temporal_audit import RELEASE


ROOT = Path("data/data-expansion-v3")
TASKS = Path("tasks/data-expansion-v3")
START = date(2025, 7, 1)
HORIZON = 5
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
PROBE_ISSUERS = {"COST": 909832, "ORCL": 1341439, "AVGO": 1730168, "ADBE": 796343, "CRM": 1108524}


def last_complete_session(now):
    today = now.astimezone(NY).date()
    return max(day for day in trading_sessions(today - timedelta(days=14), today) if close_available(day) <= now)


def collect_ir(root, through):
    out = root / "ir"
    if (out / "manifest.json").exists():
        raise FileExistsError("IR snapshot already frozen")
    start = date(2026, 1, 1)
    days = trading_sessions(start, through)
    out.mkdir(parents=True, exist_ok=True)
    records, status = collect_apple(CachedHTTP(out / "cache", sec_user_agent=""), start, through, days,
                                    pages=list(range(1, 13)))
    write_rows(out / "headlines.jsonl", [row.model_dump(mode="json") for row in records])
    manifest = {"version": "ir-v3-001", "through": str(through), "apple": status,
                "direct_ir_tickers": ["AAPL"] if records else [],
                "direct_ir_missing": [symbol for symbol in sorted(ISSUERS) if symbol != "AAPL" or not records],
                "sec_mirror": "Earnings releases for other issuers are tracked separately as SEC Exhibit 99.1",
                "headlines_sha256": sha256(out / "headlines.jsonl")}
    write_json(out / "manifest.json", manifest)


def collect_macro(root, through):
    out = root / "macro"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "receipt.json").exists():
        raise FileExistsError("Macro series already frozen")
    url = f"{FRED}?id=DFEDTARU&cosd={START}&coed={through}"
    with urlopen(url, timeout=20) as response:
        body = response.read(100_001)
    if len(body) > 100_000 or not body.startswith(b"observation_date,DFEDTARU"):
        raise ValueError("Unexpected FRED target series response")
    atomic_write(out / "DFEDTARU.csv", body)
    write_json(out / "receipt.json", {"url": url, "series": "DFEDTARU", "version": "fred-v3-001",
                                     "retrieved_at": datetime.now(UTC).isoformat(),
                                     "sha256": sha256(out / "DFEDTARU.csv")})


def collect_ir_releases(root, through):
    """Read dated official IR release metadata and discard all publisher body text."""
    from bs4 import BeautifulSoup
    from pypdf import PdfReader

    out = root / "ir-releases"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("IR release supplement already frozen")
    sessions = trading_sessions(date(2026, 1, 1), through)
    base = "https://www.microsoft.com/en-us/Investor/earnings"
    sources = [
        ("MSFT", f"{base}/{quarter}/press-release-webcast", label, day, date_text, False)
        for quarter, label, day, date_text in (
            ("FY-2026-Q2", "FY2026 second quarter results", date(2026, 1, 28), "January 28, 2026"),
            ("FY-2026-Q3", "FY2026 third quarter results", date(2026, 4, 29), "April 29, 2026"),
            ("FY-2026-Q4", "FY2026 fourth quarter results", date(2026, 7, 29), "July 29, 2026"),
            ("FY-2027-Q1", "FY2027 first quarter results", date(2026, 10, 28), "October 28, 2026"),
        )
    ] + [("PFE", "https://www.pfizer.com/print/pdf/node/563670",
          "Third quarter 2026 earnings release", date(2026, 11, 3), "November 03, 2026", True)]
    http = CachedHTTP(out / "cache", sec_user_agent="")
    records, receipts, issues = [], [], []
    for ticker, url, title, day, date_text, pdf in sources:
        if day > through:
            continue
        try:
            payload = http.get(url, cap=3_000_000, store_body=False)
            if pdf:
                body = PdfReader(BytesIO(payload.body)).pages[0].extract_text()
            else:
                body = BeautifulSoup(payload.body, "html.parser").get_text(" ", strip=True)
            if date_text not in " ".join(body.split()):
                raise ValueError("Release publication date not verified")
            eligible = next_session_proxy(day, sessions)
            records.append(Evidence(
                ticker=ticker, source_type="news", document_id=url, source_url=url, excerpt=title,
                ingested_at=payload.ingested_at, eligible_at=eligible, content_hash=payload.content_hash,
                availability_basis="date_only_verified_issuer_release_next_session",
                pit_level="historical_proxy" if eligible else "unavailable",
                source_metadata={"title": title, "published_date": str(day), "hash_scope": "publisher_response"},
            ).model_dump(mode="json"))
            receipts.append({"ticker": ticker, "url": url, "published_date": str(day),
                             "retrieved_at": payload.ingested_at.isoformat(), "sha256": payload.content_hash})
        except (OSError, ValueError) as exc:
            issues.append({"ticker": ticker, "url": url, "code": type(exc).__name__})
    write_rows(out / "headlines.jsonl", records)
    write_json(out / "manifest.json", {"version": "direct-ir-releases-v3-001", "receipts": receipts,
                                      "issues": issues, "headlines_sha256": sha256(out / "headlines.jsonl")})


def collect_probe_filings(root):
    """Fill the late-2025 filing stratum with additional original accessions."""
    out = root / "probe-sec"
    fundamentals = out / "fundamentals"
    fundamentals.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("Supplemental probe sources already frozen")
    days = trading_sessions(date(2025, 11, 20), date(2026, 1, 16))
    issues, issuers = [], {}
    with collector_lock():
        cache = SecCache(out / "cache", out / "unused", False, fresh_metadata=True)
        for ticker, cik in PROBE_ISSUERS.items():
            try:
                rows = periodic_rows(cache, ticker, cik, issues, first="2025-11-20", last="2025-12-31")
                filings = {key: dated_filing(ticker, cik, value, days) for key, value in rows.items()}
                payload = cache.get(f"{FACTS}CIK{cik:010}.json")
                facts = normalize(json.loads(payload.body), cik, filings)
                path = fundamentals / f"{ticker}.json"
                write_json(path, {"filings": filings, "facts": facts})
                issuers[ticker] = {"company_facts_sha256": payload.content_hash, "sha256": sha256(path)}
            except (OSError, ValueError, KeyError) as exc:
                issues.append({"ticker": ticker, "code": type(exc).__name__})
    write_json(fundamentals / "manifest.json", {"issuers": issuers})
    write_json(out / "manifest.json", {"version": "v3c-late-2025-fill", "issuers": issuers, "issues": issues,
                                       "source_files": cache.used})


def _partition(market, sessions, partition, start, stop, count, used):
    selected = []
    dates_used = Counter()
    reasons = Counter()
    idx = {day: i for i, day in enumerate(sessions)}
    positions = [i for i, day in enumerate(sessions) if start <= day <= stop]
    required_days = {ticker: {bar.day for bar in bars if bar.volume > 0} for ticker, bars in market.series.items()}
    for symbol in sorted(ISSUERS):
        _, etf, _ = ISSUERS[symbol]
        for slot in range(count):
            part = positions[len(positions) * slot // count:len(positions) * (slot + 1) // count]
            if not part:
                reasons["unfilled_slot"] += 1
                continue
            salt = int.from_bytes(hash_bytes(f"v3:{partition}:{symbol}:{slot}".encode()).digest()[:4])
            target = part[0] + salt % len(part)
            for i in sorted(part, key=lambda pos: (dates_used[sessions[pos]], abs(pos - target), pos)):
                needed = sessions[max(0, i - 20):i + HORIZON + 1]
                if len(needed) != 21 + HORIZON or any(
                        any(day not in required_days.get(ticker, ()) for day in needed)
                        for ticker in (symbol, "SPY", etf)):
                    reasons["missing_market_window"] += 1
                    continue
                if any(abs(i - prev) < HORIZON for prev in used[symbol]):
                    reasons["overlap_or_duplicate"] += 1
                    continue
                day = sessions[i]
                task = Task(task_id=f"v3-{partition}-{symbol.lower()}-{slot + 1:02d}",
                            symbol=symbol, decision_date=day, horizon_sessions=HORIZON)
                selected.append(task)
                dates_used[day] += 1
                used[symbol].append(idx[day])
                break
            else:
                reasons["unfilled_slot"] += 1
    return sorted(selected, key=lambda task: (task.decision_date, task.symbol)), dates_used, reasons


def freeze_tasks(root, out, audit, legacy_paths):
    market_path, sec_path, ir_path = root / "market", root / "sec", root / "ir"
    market = HistoricalData(market_path)
    audit_result = json.loads(audit.read_text())
    last = date.fromisoformat(market.manifest["end"])
    if last > last_complete_session(datetime.now(UTC)):
        raise ValueError("Snapshot includes a session which has not closed")
    sessions = list(trading_sessions(START, last))
    positions = {day: i for i, day in enumerate(sessions)}
    used = {symbol: [] for symbol in ISSUERS}
    for path in legacy_paths:
        for task in read_tasks(path):
            if task.decision_date in positions and task.symbol in used:
                used[task.symbol].append(positions[task.decision_date])
    probes = audit.parent / "probes.jsonl"
    if sha256(probes) != json.loads((audit.parent / "freeze.json").read_text())["probe_sha256"]:
        raise ValueError("Task eligibility requires the original frozen probe list")
    probe_controls = 0
    for line in probes.read_text().splitlines():
        row = json.loads(line)
        if row["feature"] != "return":
            continue
        symbol, first, _ = row["key"].split("/")
        day = date.fromisoformat(first)
        if day in positions and symbol in used:
            used[symbol].append(positions[day])
            probe_controls += 1
    boundary = audit_result["boundary"]
    earliest_holdout = max(date(2026, 9, 1), date.fromisoformat(RELEASE) + timedelta(days=1))
    if boundary["identified"]:
        earliest_holdout = max(earliest_holdout, date.fromisoformat(boundary["candidate"]))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol": "data-expansion-v3-001", "data_through": str(last),
        "settlement": {"entry": "next scheduled regular-session open", "exit": "fifth scheduled session close",
                       "scale": "one Yahoo snapshot share scale per ticker across entry and exit",
                       "benchmark": "SPY", "dividends": "excluded"},
        "decision_cutoff": "16:15 America/New_York on decision_date",
        "sampling": "per-symbol equal-session strata; deterministic hash target, then favor less-used dates; "
                    "complete 21-session warmup and 5-session outcome calendar for ticker, SPY and sector ETF",
        "knowledge_audit_summary_sha256": sha256(audit), "knowledge_boundary": boundary,
        "probe_sha256": sha256(probes), "known_probe_return_windows_excluded": probe_controls,
        "public_model_release": RELEASE, "holdout_label_exported": False, "holdout_agent_executed": False,
        "market_manifest_sha256": sha256(market_path / "manifest.json"),
        "sec_manifest_sha256": sha256(sec_path / "source_manifest.json"),
        "ir_manifest_sha256": sha256(ir_path / "manifest.json"),
        "ir_releases_manifest_sha256": sha256(root / "ir-releases" / "manifest.json"),
        "legacy_task_hashes": {str(path): sha256(path) for path in legacy_paths}, "partitions": {},
    }
    dev, all_tasks = [], []
    for partition, first, end, n in (
        ("development", date(2026, 1, 1), date(2026, 7, 31), 4),
        ("holdout", earliest_holdout, last, 5),
    ):
        tasks, density, reasons = _partition(market, sessions, partition, first, end, n, used)
        path = out / f"{partition}.jsonl"
        atomic_write(path, "".join(task.model_dump_json() + "\n" for task in tasks).encode())
        manifest["partitions"][partition] = {
            "requested": n * len(ISSUERS), "count": len(tasks), "sha256": sha256(path),
            "decision_range": [str(first), str(end)], "sample_dates": len(density),
            "selected_range": [str(tasks[0].decision_date), str(tasks[-1].decision_date)] if tasks else None,
            "max_tasks_on_one_date": max(density.values(), default=0), "rejections": dict(reasons),
            "sectors": dict(Counter(ISSUERS[task.symbol][0] for task in tasks)),
            "tickers": dict(Counter(task.symbol for task in tasks)),
        }
        if partition == "development":
            dev = tasks
        all_tasks.extend(tasks)
    from .data_v2_tasks import window_overlap
    manifest["outcome_window_overlap"] = window_overlap(all_tasks, sessions)
    if manifest["outcome_window_overlap"]:
        raise ValueError("V3 sample has overlapping same-stock settlement windows")
    manifest["development_pit_audit"] = audit_pit(sec_path, dev) | audit_evidence(sec_path, ir_path, market, dev)
    write_json(out / "manifest.json", manifest)
    return manifest


def audit_pit(sec, tasks):
    checked, future, covered = 0, 0, 0
    cache = {}
    for task in tasks:
        if task.symbol not in cache:
            path = sec / "fundamentals" / f"{task.symbol}.json"
            cache[task.symbol] = json.loads(path.read_text()) if path.exists() else {"facts": [], "filings": {}}
        bundle = cache[task.symbol]
        view = FundamentalsView(task, "", bundle["facts"], bundle["filings"], "development-v3")
        facts = [fact for group in view.facts.values() for fact in group]
        covered += bool(facts)
        checked += len(facts)
        for fact in facts:
            cutoff = close_available(task.decision_date)
            filing = fact.filing
            future += (fact.end > task.decision_date or datetime.fromisoformat(filing["eligible_at"]) > cutoff
                       or (filing.get("acceptance_datetime") is not None and
                           datetime.fromisoformat(filing["acceptance_datetime"]) > cutoff))
    if future:
        raise ValueError("Future accession or reporting period surfaced in a Development view")
    return {"tasks_with_eligible_xbrl": covered, "eligible_fact_instances_checked": checked,
            "future_facts": future, "timestamp_level": "historical proxy"}


def audit_evidence(sec, ir, market, tasks):
    releases = ir.parent / "ir-releases"
    files = (sec / "evidence.jsonl", ir / "headlines.jsonl", releases / "headlines.jsonl")
    expected = (json.loads((sec / "source_manifest.json").read_text())["evidence_sha256"],
                json.loads((ir / "manifest.json").read_text())["headlines_sha256"],
                json.loads((releases / "manifest.json").read_text())["headlines_sha256"])
    if tuple(sha256(path) for path in files) != expected:
        raise ValueError("Evidence snapshot SHA-256 differs")
    store = EvidenceStore([Evidence.model_validate_json(line) for path in files for line in path.read_text().splitlines()])
    summary = Counter()
    for task in tasks:
        cutoff = close_available(task.decision_date)
        view = store.view(task)
        for source in ("sec_filing", "news"):
            visible = view.visible(source)
            summary[f"tasks_with_{source}"] += bool(visible)
            for item in visible:
                if item.eligible_at is None or item.eligible_at > cutoff or item.ticker != task.symbol:
                    raise ValueError("Ineligible evidence in model view")
                summary["visible_evidence_records_checked"] += 1
        bars = market.view(task).history
        if any(bar.available_at > cutoff or bar.day > task.decision_date for bar in bars):
            raise ValueError("Future market data in Development input")
        summary["visible_market_bars_checked"] += len(bars)
        if market.outcome(task).available_at <= cutoff:
            raise ValueError("Invalid Development settlement order")
        summary["development_settlements_verified"] += 1
    return dict(summary) | {"future_evidence": 0, "future_bars": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("market", "ir", "ir-releases", "macro", "probe-sec", "freeze"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=TASKS)
    parser.add_argument("--audit", type=Path, default=Path("results/temporal-knowledge-audit-001/summary.json"))
    parser.add_argument("--through", type=date.fromisoformat)
    args = parser.parse_args()
    latest = last_complete_session(datetime.now(UTC))
    through = args.through or latest
    if through > latest:
        raise ValueError("Request exceeds the last completed session")
    if args.action == "market":
        print(snapshot_market(args.root / "market", start=START, end=through, snapshot_version="yahoo-us-v3-001")["failures"])
    elif args.action == "ir":
        collect_ir(args.root, through)
    elif args.action == "ir-releases":
        collect_ir_releases(args.root, through)
    elif args.action == "macro":
        collect_macro(args.root, through)
    elif args.action == "probe-sec":
        collect_probe_filings(args.root)
    else:
        legacy = [Path("tasks/development.jsonl"), *(Path("tasks/data-expansion-v2") / f"{name}.jsonl"
                    for name in ("offline", "development", "holdout"))]
        print(json.dumps(freeze_tasks(args.root, args.output, args.audit, legacy)["partitions"]))


if __name__ == "__main__":
    main()
