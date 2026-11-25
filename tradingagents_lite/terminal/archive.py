"""Read archived runs and resolve their frozen data for presentation."""
from contextlib import closing
from copy import copy
from datetime import datetime
import json
import re
import sqlite3

from .. import research_ui
from ..data import HistoricalData, close_available, read_tasks, sha256
from ..schema import Task


ROOT = research_ui.ROOT
ARCHIVE_RUNS = ("smoke-a0-001", "b0-development-001")
RUN_ID = re.compile(r"[0-9a-f]{32}")


def expansion_root():
    return getattr(research_ui, "V2_DATA", ROOT / "data/data-expansion-v2")


def task_sets():
    return ["Legacy", "Development V2"] if hasattr(research_ui, "research_tasks") else ["Legacy"]


def tasks_for(dataset):
    if dataset != "Legacy":
        return research_ui.research_tasks(dataset)
    return read_tasks(ROOT / "tasks/development.jsonl")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def archived_rows(folder):
    path = folder / "tasks.jsonl"
    if not path.is_file():
        return []
    metadata = read_json(folder / "run.json") if (folder / "run.json").is_file() else {}
    return [dict(row, run_config=metadata.get("config", {}),
                 snapshot_sha256=metadata.get("snapshot_sha256"),
                 run_started_at=metadata.get("started_at")) for row in read_rows(path)]


def load(identity):
    try:
        if RUN_ID.fullmatch(identity or ""):
            rows = archived_rows(research_ui.ui_root() / identity / "run")
            return dict(rows[0], run_id=identity) if rows else None
        archive, _, task_id = (identity or "").partition(":")
        if archive in ARCHIVE_RUNS:
            rows = archived_rows(ROOT / "results" / archive)
            return next((dict(row, run_id=identity) for row in rows if row["task_id"] == task_id), None)
    except (OSError, ValueError, TypeError):
        return None
    return None


def history():
    root = research_ui.ui_root()
    rows = []
    if root.is_dir():
        for entry in root.iterdir():
            row = load(entry.name) if entry.is_dir() and RUN_ID.fullmatch(entry.name) else None
            if row:
                rows.append(row)
    for name in ARCHIVE_RUNS:
        try:
            rows.extend(dict(row, run_id=f"{name}:{row['task_id']}")
                        for row in archived_rows(ROOT / "results" / name))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(rows, key=lambda row: (row.get("run_started_at") or "", row["decision_date"]), reverse=True)


def task_of(row):
    return Task.model_validate({key: row[key] for key in Task.model_fields})


def market_directory(task):
    if task.task_id.startswith("v2-development-"):
        return expansion_root() / "market"
    return ROOT / "data/yahoo-2024-v1"


def market_series(task, archived_hash=None):
    folder = market_directory(task)
    path = folder / "manifest.json"
    if archived_hash and sha256(path) != archived_hash:
        raise ValueError("Snapshot differs from the archived run")
    manifest = read_json(path)
    assets = manifest["assets"]
    benchmark = manifest.get("benchmark", "SPY")
    market = HistoricalData(folder, {**manifest, "assets": {task.symbol: assets[task.symbol]}})
    primary = market.view(task).history
    comparison = ()
    if benchmark in assets and benchmark != task.symbol:
        try:
            market = HistoricalData(folder, {**manifest, "assets": {benchmark: assets[benchmark]}})
            comparison = market.view(task.model_copy(update={"symbol": benchmark})).history
        except (OSError, ValueError):
            pass
    return primary, comparison, manifest.get("source", "Local market manifest")


def real_tokens(row):
    usage = row.get("usage")
    if usage is not None:
        if usage.get("total_tokens") is not None:
            return usage["total_tokens"]
        counts = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if all(value is not None for value in counts):
            return sum(counts)
    if row.get("run_config", {}).get("backend") == "offline":
        return 0
    return None


def instant(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.utcoffset() is not None else None
    except (TypeError, ValueError):
        return None


def compact_episode(row, episode, ranked):
    identity = row.get("run_id") or ""
    if not RUN_ID.fullmatch(identity) or not ranked.get("version_hash"):
        return {}
    path = research_ui.ui_root() / identity / "run-memory/episodes.sqlite"
    if not path.is_file():
        return {}
    try:
        from ..memory import Episode

        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            record = db.execute("SELECT record FROM versions WHERE memory_id=? AND version_hash=?",
                                (episode["memory_id"], ranked["version_hash"])).fetchone()
        if record:
            item = Episode.model_validate_json(record[0])
            if (item.version_hash == ranked["version_hash"] and item.memory_id == episode["memory_id"]
                    and item.eligible_at == instant(episode.get("eligible_at"))):
                return {"decision_time": item.decision_time.isoformat(),
                        "historical_outcome": item.outcome.model_dump(mode="json")}
    except (OSError, ValueError, sqlite3.Error):
        pass
    return {}


def eligible_episodes(row):
    cutoff = close_available(task_of(row).decision_date)
    memory = row.get("memory_trace") or {}
    selected = {item["memory_id"]: item for item in memory.get("selected", [])}
    episodes = []
    for episode in memory.get("context", []):
        ranked = selected.get(episode["memory_id"], {})
        if "decision_time" not in episode or "historical_outcome" not in episode:
            episode = {**compact_episode(row, episode, ranked), **episode}
        dates = [instant(episode.get("eligible_at")), instant(episode.get("decision_time")),
                 instant((episode.get("historical_outcome") or {}).get("available_at"))]
        if all(value is not None and value <= cutoff for value in dates) and dates[1] < cutoff:
            episodes.append({**ranked, **episode})
    return sorted(episodes, key=lambda episode: instant(episode["eligible_at"]), reverse=True)

def xbrl_series(task, metrics, config):
    if not task.task_id.startswith("v2-development-"):
        return []
    directory = expansion_root() / "sec/fundamentals"
    manifest = directory / "manifest.json"
    expected = config.get("fundamentals_manifest_sha256")
    try:
        from ..fundamentals import FundamentalsView

        if expected and sha256(manifest) != expected:
            return []
        view = FundamentalsView.load(directory, task, "expansion-v2")
        rows = []
        for end in sorted(view.report_periods)[-8:]:
            # Preserve the original decision cutoff for all reported periods.
            period = copy(view)
            period.report_periods = {end: view.report_periods[end]}
            rows.extend(period.query(metrics, "quarter")["items"])
        return rows
    except (ImportError, OSError, KeyError, ValueError):
        return []
