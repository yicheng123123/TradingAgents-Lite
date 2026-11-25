"""Freeze V2 task inputs independently of future prices and settlement labels."""
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import date, datetime
import hashlib
import json
from pathlib import Path

from .data import HistoricalData, NY, close_available, read_tasks, sha256
from .data_v2_market import ISSUERS, ROOT, atomic_write, trading_sessions, write_json
from .schema import Task


PROTOCOL = "equal-session-strata-v1"
PARTITIONS = {
    "offline": (2021, 2023, 10),
    "development": (2024, 2024, 4),
    "holdout": (2025, 2025, 5),
}
HORIZON = 5


def window_overlap(tasks, sessions, legacy=()):
    positions = {day: i for i, day in enumerate(sessions)}
    occupied = {}
    for task in legacy:
        if task.decision_date in positions:
            entry = positions[task.decision_date] + 1
            occupied.setdefault(task.symbol, set()).update(range(entry, entry + task.horizon_sessions))
    overlaps = 0
    for task in sorted(tasks, key=lambda task: (task.decision_date, task.symbol)):
        entry = positions[task.decision_date] + 1
        window = set(range(entry, entry + task.horizon_sessions))
        existing = occupied.setdefault(task.symbol, set())
        overlaps += bool(existing & window)
        existing.update(window)
    return overlaps


def freeze_tasks(market_path: Path, output: Path, legacy_path: Path,
                 events_path: Path | None = None) -> dict:
    if (output / "manifest.json").exists():
        raise FileExistsError("V2 task manifest is already frozen")
    market = HistoricalData(market_path)
    sessions = trading_sessions()
    positions = {day: i for i, day in enumerate(sessions)}
    legacy = read_tasks(legacy_path)
    legacy_cells = {(task.symbol, task.decision_date) for task in legacy}
    disclosures = {}
    if events_path:
        for line in events_path.read_text().splitlines():
            row = json.loads(line)
            if (row["form"] in {"10-Q", "10-K"} and row["eligible_at"]
                    and row["pit_level"] in {"historical_proxy", "verified_known_at"}):
                eligible = datetime.fromisoformat(row["eligible_at"])
                eligible_day = eligible.astimezone(NY).date()
                i = bisect_left(sessions, eligible_day)
                if i < len(sessions) and eligible <= close_available(sessions[i]):
                    disclosures.setdefault(row["ticker"], set()).add(i)

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": PROTOCOL, "selection": "per-ticker equal session strata; prefer near midpoint eligible filings on even strata",
        "horizon_sessions": HORIZON, "market_manifest_sha256": sha256(market_path / "manifest.json"),
        "legacy_tasks_sha256": sha256(legacy_path),
        "disclosure_manifest_sha256": sha256(events_path) if events_path is not None else None,
        "partitions": {},
    }
    all_tasks, previous = [], {}
    for partition, (first_year, last_year, per_ticker) in PARTITIONS.items():
        start = bisect_left(sessions, date(first_year, 1, 1))
        stop = bisect_right(sessions, date(last_year, 12, 31))
        selected, strata = [], []
        counts = Counter()
        for ticker in sorted(ISSUERS):
            stock = {bar.day: bar for bar in market.series[ticker] if bar.volume > 0}
            for slot in range(per_ticker):
                lo = start + (stop - start) * slot // per_ticker
                hi = start + (stop - start) * (slot + 1) // per_ticker
                offset = hashlib.sha256(f"{PROTOCOL}:{partition}:{ticker}:{slot}".encode()).digest()[0] % 15 - 7
                target = (lo + hi) // 2 + offset
                event_days = disclosures.get(ticker, ())
                preferred = {i + delta for i in event_days for delta in (0, 1, 2)}
                candidates = sorted(range(lo, hi), key=lambda i: (
                    not (slot % 2 == 0 and abs(i - target) <= 15 and i in preferred), abs(i - target), i,
                ))
                for i in candidates:
                    day = sessions[i]
                    if (ticker, day) in legacy_cells:
                        counts["legacy_duplicate"] += 1
                    elif i - previous.get(ticker, -HORIZON) < HORIZON:
                        counts["overlapping_v2_window"] += 1
                    elif any(session not in stock for session in sessions[i - 20:i + 1]):
                        counts["missing_or_zero_volume_in_warmup"] += 1
                    else:
                        task = Task(task_id=f"v2-{partition}-{ticker.lower()}-{slot + 1:02d}",
                                    symbol=ticker, decision_date=day, horizon_sessions=HORIZON)
                        selected.append(task)
                        previous[ticker] = i
                        if partition != "holdout":
                            v = market.view(task).indicators()["daily_volatility_20"]
                            strata.append({"task_id": task.task_id, "event_proximate": i in preferred if events_path else None,
                                           "daily_volatility_20": v, "sector": ISSUERS[ticker][0]})
                        break
                else:
                    counts["unfilled_slot"] += 1
        ordered = sorted(selected, key=lambda task: (task.decision_date, task.symbol))
        path = output / f"{partition}.jsonl"
        atomic_write(path, "".join(task.model_dump_json() + "\n" for task in ordered).encode())
        if strata:
            write_json(output / f"{partition}-strata.json", strata)
        result = {"count": len(ordered), "expected": len(ISSUERS) * per_ticker, "sha256": sha256(path),
                  "rejected_candidates": dict(counts), "tickers": dict(Counter(t.symbol for t in ordered)),
                  "sectors": dict(Counter(ISSUERS[t.symbol][0] for t in ordered))}
        manifest["partitions"][partition] = result
        all_tasks.extend(ordered)
    manifest["v2_window_overlaps"] = window_overlap(all_tasks, sessions)
    manifest["v2_or_legacy_window_overlaps"] = window_overlap(all_tasks, sessions, legacy=legacy)
    write_json(output / "manifest.json", manifest)
    return manifest


def frozen_tasks(root: Path, partition: str, market_path: Path | None = None) -> list[Task]:
    if partition not in PARTITIONS:
        raise ValueError("Unknown V2 partition")
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / f"{partition}.jsonl"
    if (manifest["protocol"] != PROTOCOL or sha256(path) != manifest["partitions"][partition]["sha256"]
            or (market_path is not None and sha256(market_path / "manifest.json") != manifest["market_manifest_sha256"])):
        raise ValueError("Frozen task input manifest mismatch")
    tasks = read_tasks(path)
    start, end, count = PARTITIONS[partition]
    if (Counter(t.symbol for t in tasks) != {ticker: count for ticker in ISSUERS}
            or any(not t.task_id.startswith(f"v2-{partition}-") or not start <= t.decision_date.year <= end
                   or t.horizon_sessions != HORIZON for t in tasks)):
        raise ValueError("Invalid frozen partition")
    return tasks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("tasks/data-expansion-v2"))
    parser.add_argument("--filings", type=Path)
    args = parser.parse_args()
    result = freeze_tasks(args.root / "market", args.output, Path("tasks/development.jsonl"), args.filings)
    print({name: row["count"] for name, row in result["partitions"].items()})
