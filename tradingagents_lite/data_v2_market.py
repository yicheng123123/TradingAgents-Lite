"""Frozen Data Expansion V2 market and US listed-company definitions."""
from datetime import UTC, date, datetime, time, timedelta
from importlib.metadata import version
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .data import BENCHMARK, ENDPOINT, NY, DataError, close_available, sha256
from .schema import Bar


# Filing entity in the 2021–2025 observation window, irrespective of later successors.
UNIVERSE = {
    "Technology": ("XLK", {"AAPL": 320193, "MSFT": 789019, "NVDA": 1045810, "AMD": 2488}),
    "Financials": ("XLF", {"JPM": 19617, "BAC": 70858, "GS": 886982}),
    "Healthcare": ("XLV", {"JNJ": 200406, "PFE": 78003, "UNH": 731766}),
    "Consumer Discretionary": ("XLY", {"AMZN": 1018724, "TSLA": 1318605}),
    "Consumer Staples": ("XLP", {"KO": 21344, "WMT": 104169}),
    "Energy": ("XLE", {"XOM": 34088, "CVX": 93410}),
    "Industrials": ("XLI", {"CAT": 18230, "BA": 12927}),
    "Communication Services": ("XLC", {"GOOGL": 1652044, "META": 1326801}),
}
ISSUERS = {ticker: (sector, etf, cik) for sector, (etf, tickers) in UNIVERSE.items()
           for ticker, cik in tickers.items()}
SYMBOLS = tuple(sorted(set(ISSUERS) | {etf for etf, _ in UNIVERSE.values()} | {BENCHMARK}))
START = date(2020, 10, 1)
END = date(2026, 1, 16)
MARKET_VERSION = "yahoo-us-v2-001"
ROOT = Path("data/data-expansion-v2")


def write_json(path: Path, value):
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def atomic_write(path: Path, contents: bytes):
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(contents)
    partial.replace(path)


def trading_sessions(start: date = START, end: date = END) -> tuple[date, ...]:
    import exchange_calendars

    calendar = exchange_calendars.get_calendar("XNYS", start=str(start), end=str(end))
    days = tuple(stamp.date() for stamp in calendar.sessions)
    # Federal day of mourning, verified against NYSE RM-2025-01.
    return tuple(day for day in days if day != date(2025, 1, 9))


def chart_url(symbol: str, end: date) -> str:
    def midnight(day: date):
        return int(datetime.combine(day, time.min, NY).timestamp())

    query = urlencode({"period1": midnight(START), "period2": midnight(end + timedelta(days=1)),
                       "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"})
    return f"{ENDPOINT}{symbol}?{query}"


def chart_rows(raw: bytes, symbol: str):
    chart = json.loads(raw)["chart"]
    if chart.get("error") or len(chart.get("result") or []) != 1:
        raise DataError("Unexpected Yahoo chart response")
    result, = chart["result"]
    meta = result["meta"]
    if (meta["symbol"], meta["currency"], meta["exchangeTimezoneName"], meta["dataGranularity"]) != (
        symbol, "USD", NY.key, "1d"
    ):
        raise DataError("Unexpected Yahoo chart metadata")
    stamps = result["timestamp"]
    quote, = result["indicators"]["quote"]
    keys = ("open", "high", "low", "close", "volume")
    adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose", [None] * len(stamps))
    if any(len(quote[key]) != len(stamps) for key in keys) or len(adjusted) != len(stamps):
        raise DataError("Misaligned chart columns")
    bars, adj = {}, {}
    dropped, duplicates, zero_volume, gaps = [], [], [], []
    for i, stamp in enumerate(stamps):
        day = datetime.fromtimestamp(stamp, NY).date()
        if not START <= day <= END:
            continue
        if day in bars:
            duplicates.append(str(day))
            continue
        try:
            bar = Bar(day=day, available_at=close_available(day), **{key: quote[key][i] for key in keys})
        except ValueError:
            dropped.append(str(day))
            continue
        bars[day] = bar
        value = adjusted[i]
        if type(value) in (int, float) and value > 0 and math.isfinite(value):
            adj[day] = value
        if bar.volume == 0:
            zero_volume.append(str(day))
    if duplicates or not bars:
        raise DataError("Duplicate dates or empty bar series")
    ordered = [bars[day] for day in sorted(bars)]
    for previous, current in zip(ordered, ordered[1:]):
        change = current.close / previous.close - 1
        if abs(change) >= .25:
            gaps.append({"day": str(current.day), "close_move": round(change, 6)})

    events = result.get("events", {})
    corporate = {"splits": [], "dividends": []}
    for kind, rows in corporate.items():
        for event in events.get(kind, {}).values():
            day = datetime.fromtimestamp(event["date"], NY).date()
            if kind == "splits":
                numerator, denominator = event["numerator"], event["denominator"]
                if not (0 < numerator and 0 < denominator):
                    raise DataError("Malformed split ratio")
                rows.append({"date": str(day), "numerator": numerator, "denominator": denominator})
            elif START <= day <= END:
                rows.append({"date": str(day), "amount": event["amount"]})
        rows.sort(key=lambda row: row["date"])
    return ordered, adj, corporate, {"invalid_rows": dropped, "duplicates": duplicates,
                                     "zero_volume_sessions": zero_volume, "large_close_moves": gaps}


def snapshot_market(root: Path, *, offline: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        from .data import HistoricalData

        return HistoricalData(root).manifest
    sessions = set(trading_sessions())
    assets, failures = {}, {}
    for symbol in SYMBOLS:
        raw = root / f"{symbol}.raw.json"
        receipt = root / f"{symbol}.download.json"
        try:
            if raw.is_file() and receipt.is_file():
                meta = json.loads(receipt.read_text())
                if meta["raw_sha256"] != sha256(raw):
                    raise DataError("Market download hash mismatch")
            elif offline:
                raise DataError("Offline market cache miss")
            else:
                # Read through the completed session before retrieval to account for
                # subsequent Yahoo split restatements without exposing later price rows.
                source_end = max(END, datetime.now(NY).date() - timedelta(days=1))
                url = chart_url(symbol, source_end)
                with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as resp:
                    body = resp.read(4_000_001)
                if not body or len(body) > 4_000_000:
                    raise DataError("Invalid market response size")
                chart_rows(body, symbol)
                atomic_write(raw, body)
                meta = {"url": url, "retrieved_at": datetime.now(UTC).isoformat(), "raw_sha256": sha256(raw)}
                write_json(receipt, meta)
            bars, adjusted, corporate, audit = chart_rows(raw.read_bytes(), symbol)
            output = root / f"{symbol}.bars.jsonl"
            supplemental = root / f"{symbol}.actions.json"
            atomic_write(output, "".join(bar.model_dump_json() + "\n" for bar in bars).encode())
            write_json(supplemental, {"adjusted_close": {str(d): x for d, x in adjusted.items()}, **corporate})
            available = {bar.day for bar in bars}
            assets[symbol] = {
                **meta, "rows": len(bars), "first_day": str(bars[0].day), "last_day": str(bars[-1].day),
                "bars_sha256": sha256(output), "actions_sha256": sha256(supplemental),
                "missing_sessions": [str(day) for day in sorted(sessions - available)],
                "off_calendar_sessions": [str(day) for day in sorted(available - sessions)],
                "adjusted_close_rows": len(adjusted), "split_events": len(corporate["splits"]),
                "dividend_events": len(corporate["dividends"]), **audit,
            }
        except (OSError, KeyError, ValueError, TypeError) as error:
            failures[symbol] = {"type": type(error).__name__, "status": getattr(error, "code", None)}
    benchmark = root / f"{BENCHMARK}.bars.jsonl"
    if BENCHMARK in assets:
        benchmark_days = {json.loads(line)["day"] for line in benchmark.read_text().splitlines()}
        for symbol, entry in assets.items():
            dates = {json.loads(line)["day"] for line in (root / f"{symbol}.bars.jsonl").read_text().splitlines()}
            entry["missing_vs_spy"] = sorted(benchmark_days - dates)
            entry["extra_vs_spy"] = sorted(dates - benchmark_days)
    manifest = {
        "version": MARKET_VERSION, "source": "Yahoo Finance daily chart", "start": str(START), "end": str(END),
        "benchmark": BENCHMARK, "calendar": f"XNYS/exchange-calendars {version('exchange-calendars')}",
        "expected_sessions": len(sessions), "failures": failures, "assets": assets,
        "data_class": "retrospective vendor snapshot; historical values and availability are not independently authenticated",
    }
    write_json(root / ("market-attempt.json" if failures else "manifest.json"), manifest)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "market")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    manifest = snapshot_market(args.output, offline=args.offline)
    print(json.dumps({"assets": len(manifest["assets"]), "failed": manifest["failures"]}))
    raise SystemExit(bool(manifest["failures"]))
