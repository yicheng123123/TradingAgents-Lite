"""Freeze a vendor snapshot and serve only information observable by a task cutoff."""
from datetime import UTC, date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import statistics
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .schema import Bar, Outcome, Task

NY = ZoneInfo("America/New_York")
ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/"
BENCHMARK = "SPY"


class DataError(ValueError):
    """Historical data is missing, inconsistent or outside the allowed window."""


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def close_available(day: date) -> datetime:
    # 16:15 ET is conservative on regular and early-close sessions.
    return datetime.combine(day, time(16, 15), NY).astimezone(UTC)


def read_tasks(path: Path) -> list[Task]:
    tasks = [Task.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    unique_ids = {t.task_id for t in tasks}
    unique_cells = {(t.symbol, t.decision_date) for t in tasks}
    if not tasks or len(tasks) != len(unique_ids) or len(tasks) != len(unique_cells):
        raise DataError("Task list must contain unique IDs and symbol/date pairs")
    return tasks


def _parse_chart(raw: bytes, symbol: str, start: date, end: date) -> tuple[list[Bar], dict]:
    chart = json.loads(raw)["chart"]
    if chart.get("error") or len(chart.get("result") or []) != 1:
        raise DataError(f"Invalid chart response for {symbol}")
    result = chart["result"][0]
    meta = result["meta"]
    if (meta["symbol"] != symbol or meta["currency"] != "USD" or
            meta["exchangeTimezoneName"] != NY.key or meta.get("dataGranularity") != "1d"):
        raise DataError(f"Unexpected historical price metadata for {symbol}")
    stamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    keys = ("open", "high", "low", "close", "volume")
    if any(len(quote[k]) != len(stamps) for k in keys):
        raise DataError(f"Chart arrays differ in length for {symbol}")
    bars = []
    for i, ts in enumerate(stamps):
        day = datetime.fromtimestamp(ts, NY).date()
        if not start <= day <= end:
            raise DataError("Vendor returned a trading date outside the requested window")
        values = {k: quote[k][i] for k in keys}
        bars.append(Bar(day=day, available_at=close_available(day), **values))
    if not bars or sorted({b.day for b in bars}) != [b.day for b in bars]:
        raise DataError(f"Invalid trading calendar for {symbol}")
    events = result.get("events", {})
    return bars, {"split_events_reported": len(events.get("splits", {})),
                  "dividend_events_reported": len(events.get("dividends", {}))}


def snapshot(output: Path, tasks: list[Task], start: date, end: date) -> dict:
    """Save raw responses and checked bar files. Publish the manifest only on success."""
    if start >= end or end >= datetime.now(NY).date():
        raise DataError("Use a completed historical window")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source": "Yahoo Finance chart daily quote OHLCV",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "start": start.isoformat(), "end": end.isoformat(), "benchmark": BENCHMARK,
        "data_class": "retrospective vendor snapshot; revision and split history not authenticated point-in-time",
        "assets": {},
    }
    for symbol in sorted({t.symbol for t in tasks} | {BENCHMARK}):
        start_ts = int(datetime.combine(start, time.min, NY).timestamp())
        end_ts = int(datetime.combine(end + timedelta(days=1), time.min, NY).timestamp())
        query = urlencode({"period1": start_ts, "period2": end_ts, "interval": "1d", "events": "div,splits"})
        url = f"{ENDPOINT}{symbol}?{query}"
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=25) as response:
            raw = response.read(3_000_001)
        if len(raw) > 3_000_000:
            raise DataError("Chart response exceeded 3 MB")
        bars, audit = _parse_chart(raw, symbol, start, end)
        raw_path = output / f"{symbol}.raw.json"
        raw_path.write_bytes(raw)
        bar_path = output / f"{symbol}.bars.jsonl"
        bar_path.write_text("".join(b.model_dump_json() + "\n" for b in bars), encoding="utf-8")
        manifest["assets"][symbol] = {
            "url": url, "rows": len(bars), "first_day": str(bars[0].day),
            "last_day": str(bars[-1].day), "raw_sha256": sha256(raw_path),
            "bars_sha256": sha256(bar_path), **audit,
        }
    market = HistoricalData(output, manifest)
    for task in tasks:
        market.view(task)
        market.outcome(task)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class MarketView:
    """Agent-facing tool data. It contains no future bars or benchmark outcomes."""

    def __init__(self, task: Task, history: tuple[Bar, ...]):
        self.task = task
        cutoff = close_available(task.decision_date)
        self.history = tuple(b for b in history if b.day <= task.decision_date and b.available_at <= cutoff)
        if len(self.history) < 21 or self.history[-1].day != task.decision_date:
            raise DataError("Need 21 available sessions and an exact decision-date bar")

    def get_bars(self, limit: int) -> dict:
        bars = self.history[-limit:]
        return {"evidence_id": f"{self.task.symbol}:{self.task.decision_date}:bars:{limit}",
                "bars": [b.model_dump(mode="json") for b in bars]}

    def indicators(self) -> dict:
        bars = self.history[-21:]
        closes = [b.close for b in bars]
        changes = [y / x - 1 for x, y in zip(closes, closes[1:])]
        return {"evidence_id": f"{self.task.symbol}:{self.task.decision_date}:indicators",
                "date": self.task.decision_date.isoformat(), "close": closes[-1],
                "return_5": closes[-1] / closes[-6] - 1,
                "sma_20": statistics.fmean(closes[-20:]),
                "daily_volatility_20": statistics.stdev(changes)}


class HistoricalData:
    """Load a hash-checked snapshot. Only the runner receives this full object."""

    def __init__(self, directory: Path, manifest: dict | None = None):
        self.manifest = manifest if manifest is not None else json.loads((directory / "manifest.json").read_text())
        self.series = {}
        for symbol, entry in self.manifest["assets"].items():
            if not symbol.isascii() or not symbol.isalnum():
                raise DataError("Invalid snapshot symbol")
            path = directory / f"{symbol}.bars.jsonl"
            if (sha256(path) != entry["bars_sha256"]
                    or sha256(directory / f"{symbol}.raw.json") != entry["raw_sha256"]):
                raise DataError(f"Snapshot hash mismatch for {symbol}")
            bars = tuple(Bar.model_validate_json(r) for r in path.read_text().splitlines())
            if (not bars or len(bars) != entry["rows"]
                    or [b.day for b in bars] != sorted({b.day for b in bars})
                    or any(b.available_at < close_available(b.day) for b in bars)):
                raise DataError(f"Invalid snapshot calendar or availability for {symbol}")
            self.series[symbol] = bars

    def view(self, task: Task) -> MarketView:
        return MarketView(task, self.series[task.symbol])

    def outcome(self, task: Task) -> Outcome:
        future = [b for b in self.series[task.symbol] if b.day > task.decision_date]
        if len(future) < task.horizon_sessions:
            raise DataError("Incomplete outcome window")
        window = future[:task.horizon_sessions]
        bench = {b.day: b for b in self.series[self.manifest["benchmark"]]}
        expected = [b.day for b in bench.values() if b.day > task.decision_date][:task.horizon_sessions]
        if [b.day for b in window] != expected:
            raise DataError("Stock and benchmark must share the entire outcome calendar")
        entry, exit_bar = window[0], window[-1]
        a, b = bench[entry.day], bench[exit_bar.day]
        return Outcome(entry_day=entry.day, exit_day=exit_bar.day,
                       available_at=max(exit_bar.available_at, b.available_at),
                       asset_return=exit_bar.close / entry.open - 1,
                       benchmark_return=b.close / a.open - 1)
