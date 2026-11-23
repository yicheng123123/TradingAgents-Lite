"""Build a versioned evidence snapshot and audit the fixed historical task grid."""
import csv
from datetime import timedelta
import json
import os
from pathlib import Path

from .data import HistoricalData, close_available, read_tasks, sha256
from .edgar import collect_filings
from .evidence import EvidenceStore, LOOKBACK
from .evidence_http import CachedHTTP
from .news import audit_microsoft, collect_apple, collect_microsoft


def collect(tasks_path: Path, market_path: Path, output: Path, cache: Path, *,
            offline: bool, apple_pages: list[int]) -> dict:
    tasks = read_tasks(tasks_path)
    market = HistoricalData(market_path)
    sessions = tuple(bar.day for bar in market.series[market.manifest["benchmark"]])
    for task in tasks:
        market.view(task)
    first, last = min(t.decision_date for t in tasks), max(t.decision_date for t in tasks)
    if sessions[-1] <= last:
        raise ValueError("Evidence lag policy requires at least one market session after the last task")
    start = max(sessions[0], first-timedelta(days=max(LOOKBACK.values())))
    news_start = first-timedelta(days=LOOKBACK["news"] + 7)
    output.mkdir(parents=True, exist_ok=False)
    ua = os.environ.get("TALITE_SEC_USER_AGENT", "").strip()
    http_ua = ua or "TradingAgents-Lite/0.1 (independent historical evidence research)"
    http = CachedHTTP(cache, offline=offline, sec_user_agent=http_ua)
    tickers = {t.symbol for t in tasks}
    records, sources = [], []
    for ticker in sorted(tickers):
        items, info = collect_filings(http, ticker, start, last, sessions)
        records.extend(items)
        sources.append(info)
    if "AAPL" in tickers:
        items, info = collect_apple(http, news_start, last, sessions, apple_pages)
        records.extend(items)
        sources.append(info)
    if "MSFT" in tickers:
        items, info = collect_microsoft(http, news_start, last, sessions)
        records.extend(items)
        sources.append(info)
        sources.append(audit_microsoft(http, [(first.year, first.month)]))
    store = EvidenceStore(records)
    data_path = output / "evidence.jsonl"
    data_path.write_text("".join(r.model_dump_json() + "\n" for r in
                                sorted(store.records, key=lambda x: x.evidence_id)), encoding="utf-8")
    manifest = {"version": 1, "tasks_sha256": sha256(tasks_path),
                "market_calendar_manifest_sha256": sha256(market_path / "manifest.json"),
                "evidence_sha256": sha256(data_path), "count": len(store.records),
                "filing_window": [str(start), str(last)], "news_window": [str(news_start), str(last)],
                "apple_archive_pages": apple_pages, "lookback_days_by_eligibility": LOOKBACK,
                "sec_user_agent": "configured" if ua else "declared_project_default",
                "availability": "Proxy at 16:15 America/New_York on the next observed session. No verified first-public timestamp authenticated.",
                "source_status": sources}
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n",
                                                  encoding="utf-8")
    return manifest


def coverage(tasks_path: Path, evidence_path: Path, output: Path) -> dict:
    tasks = read_tasks(tasks_path)
    store = EvidenceStore.load(evidence_path)
    manifest_path = evidence_path / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["tasks_sha256"] != sha256(tasks_path):
        raise ValueError("Coverage task file differs from the evidence collection task set")
    output.mkdir(parents=True, exist_ok=False)
    source_health = {}
    for source in manifest["source_status"]:
        key = (source["ticker"], source["source_type"])
        source_health.setdefault(key, {})[source["provider"]] = source["status"]
    rows = []
    for task in tasks:
        view = store.view(task)
        row = {"task_id": task.task_id, "ticker": task.symbol, "decision_date": str(task.decision_date),
               "decision_cutoff": close_available(task.decision_date).isoformat()}
        for kind in ("sec_filing", "news"):
            visible = view.visible(kind)
            row[f"{kind}_status"] = "available" if visible else "unavailable"
            row[f"{kind}_proxy_count"] = sum(item.pit_level == "historical_proxy" for item in visible)
            row[f"{kind}_verified_count"] = sum(item.pit_level == "verified_known_at" for item in visible)
            row[f"{kind}_ids"] = [item.evidence_id for item in visible]
            row[f"{kind}_forms"] = sorted({item.source_metadata["form"] for item in visible
                                             if "form" in item.source_metadata})
            health = source_health.get((task.symbol, kind), {})
            row[f"{kind}_source_status"] = health
            if visible:
                reason = None
            elif not health:
                reason = "no_provider_configured"
            elif all(status == "unavailable" for status in health.values()):
                reason = "source_or_historical_version_unavailable"
            else:
                reason = "no_eligible_evidence_in_snapshot"
            row[f"{kind}_missing_reason"] = reason
        rows.append(row)
    fields = list(rows[0])
    with (output / "coverage.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k,v in row.items()})
    (output / "coverage.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = {kind: {
        "available": sum(r[f"{kind}_status"] == "available" for r in rows),
        "verified": sum(r[f"{kind}_verified_count"] > 0 for r in rows),
        "proxy": sum(r[f"{kind}_proxy_count"] > 0 for r in rows),
        "unavailable": sum(r[f"{kind}_status"] == "unavailable" for r in rows),
    } for kind in ("sec_filing", "news")}
    result = {"tasks": len(rows), "counts": counts, "evidence_sha256": manifest["evidence_sha256"],
              "tasks_sha256": manifest["tasks_sha256"], "source_manifest_sha256": sha256(manifest_path)}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "source_manifest.json").write_bytes(manifest_path.read_bytes())
    lines = ["# Development evidence coverage", "",
             "Coverage means at least one eligible version in the 180-day SEC or 30-day news window "
             "measured from the proxy eligibility timestamp. Results describe source coverage, not model accuracy.", "",
             "| Source | Available | Proxy | Verified exact version | Unavailable |",
             "|---|---:|---:|---:|---:|"]
    for kind, row in counts.items():
        lines.append(f"| {kind} | {row['available']}/{len(rows)} | {row['proxy']} | "
                     f"{row['verified']} | {row['unavailable']} |")
    lines += ["", "| Task | Ticker | Decision date | Filing count | News count |",
              "|---|---|---|---:|---:|"]
    for row in rows:
        values = [row[f"{k}_proxy_count"] + row[f"{k}_verified_count"] for k in ("sec_filing", "news")]
        lines.append(f"| {row['task_id']} | {row['ticker']} | {row['decision_date']} | {values[0]} | {values[1]} |")
    lines += ["", "The successful news sources are issuer-controlled and sampled. Apple has archive dates only. "
              "Microsoft has a curated Security Blog subset. Missing rows do not prove an absence of company news. "
              "EDGAR uses primary accession documents only and keeps report period, filing day, acceptance and "
              "unverified public availability separate. See source_manifest.json for per-request errors and hashes."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result
