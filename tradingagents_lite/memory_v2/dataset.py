"""Read the two development cohorts and derive point-in-time market-regime proxies."""
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

from ..data import HistoricalData, close_available, read_tasks, sha256 as file_hash
from ..memory import Episode, deterministic_reflection
from ..schema import Bar, Budget, Decision, Outcome
from ..workflow import MomentumBackend, build_workflow
from .reflection import bind, rule_notes

ARM = "a2-v2-frozen-replay"
DEV_MARKET_END = date(2026, 8, 7)
BUDGET = Budget()


@dataclass(frozen=True)
class Case:
    episode: Episode
    query: str
    momentum: str
    high_volatility: bool


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def regime(indicators):
    value = indicators["return_5"]
    momentum = "positive" if value > BUDGET.momentum_threshold else (
        "negative" if value < -BUDGET.momentum_threshold else "flat")
    return momentum, indicators["daily_volatility_20"] > BUDGET.volatility_limit


def make_case(task, cohort, report, decision, outcome, evidence):
    cutoff = close_available(task.decision_date)
    indicator = evidence[f"{task.symbol}:{task.decision_date}:indicators"]
    if indicator["date"] != str(task.decision_date) or any(
        key not in evidence for key in decision.evidence_ids
    ):
        raise ValueError("Unbound decision-time evidence")
    momentum, high_volatility = regime(indicator)
    episode = Episode(
        source_task_id=task.task_id, ticker=task.symbol, decision_time=cutoff,
        outcome_available_at=outcome.available_at, reflection_created_at=outcome.available_at,
        eligible_at=outcome.available_at, action=decision.action, decision_rationale=decision.rationale,
        observed_evidence_ids=decision.evidence_ids, risk_veto=decision.risk_veto,
        evidence_quality=0.75 if decision.evidence_ids else 0.0, outcome=outcome,
        reflection=deterministic_reflection(decision, outcome), dataset_partition=cohort,
        experiment_arm=ARM,
    )
    return Case(episode, f"{task.symbol} {report['view'][:600]}", momentum, high_volatility)


def reflect(case):
    volatility = "above" if case.high_volatility else "within"
    report_view = case.query.partition(" ")[2]
    situation = f"{report_view} Daily volatility {volatility} the risk limit."
    notes = rule_notes(case.momentum, case.episode.risk_veto)
    return bind(case.episode, situation, notes)


def archived_a2(root, repo):
    tasks_path = repo / "tasks/development.jsonl"
    archive = root / "memory-run-001/M3/tasks.jsonl"
    market_manifest = root / "yahoo-2024-v1/manifest.json"
    metadata = json.loads((repo / "results/a2-memory-local-001/summary.json").read_text())
    if file_hash(tasks_path) != metadata["tasks_sha256"] or file_hash(market_manifest) != metadata["market_sha256"]:
        raise ValueError("A2 historical input fingerprints differ")
    rows = {row["task_id"]: row for row in jsonl(archive)}
    cases = []
    for task in read_tasks(tasks_path):
        row = rows[task.task_id]
        if row["llm_calls"] or row["decision_date"] != str(task.decision_date) or row["symbol"] != task.symbol:
            raise ValueError("A2 archive identity mismatch")
        cases.append(make_case(task, "a2-development", row["report"],
                               Decision.model_validate(row["decision"]),
                               Outcome.model_validate(row["outcome"]), row["evidence"]))
    return cases, {"source": "A2 historical M3 archived offline MomentumBackend",
                   "tasks_sha256": file_hash(tasks_path), "archive_sha256": file_hash(archive),
                   "market_manifest_sha256": file_hash(market_manifest)}


class DevelopmentMarket(HistoricalData):
    """Read only the pre-holdout prefix of the bars, including development settlement."""

    def __init__(self, root, tasks):
        manifest = root / "manifest.json"
        self.manifest = json.loads(manifest.read_text())
        if self.manifest.get("version") != "yahoo-us-v3-001":
            raise ValueError("Expected the V3 market snapshot manifest")
        self.series = {}
        self.split_events = {}
        self.allowed = {t.task_id: t for t in tasks}
        self.fingerprints = {"manifest_sha256": file_hash(manifest), "bars_prefix_sha256": {}}
        for symbol in sorted({t.symbol for t in tasks} | {self.manifest["benchmark"]}):
            if self.manifest["assets"][symbol]["split_events"]:
                raise ValueError("Development prefix loader requires split-free series")
            lines = []
            bars = []
            with (root / f"{symbol}.bars.jsonl").open("rb", buffering=0) as file:
                # Fixed-size buffered reads could fetch sealed dates. Read one
                # line at a time using the development calendar's known end row.
                for line in file:
                    bar = Bar.model_validate_json(line)
                    if bar.day > DEV_MARKET_END or (bars and bars[-1].day >= bar.day):
                        raise ValueError("Unexpected development market calendar")
                    if bar.available_at < close_available(bar.day):
                        raise ValueError("Premature market availability")
                    lines.append(line)
                    bars.append(bar)
                    if bar.day == DEV_MARKET_END:
                        break
            if not bars or bars[-1].day != DEV_MARKET_END:
                raise ValueError("Development settlement prefix is incomplete")
            self.series[symbol] = tuple(bars)
            self.fingerprints["bars_prefix_sha256"][symbol] = sha256(b"".join(lines)).hexdigest()

    def is_sealed(self, task):
        return self.allowed.get(task.task_id) != task or super().is_sealed(task)


def v3_dev(root, repo):
    tasks_path = repo / "tasks/data-expansion-v3/development.jsonl"
    tasks = read_tasks(tasks_path)
    if len(tasks) != 80 or any(not t.task_id.startswith("v3-development-") or
                               t.decision_date >= DEV_MARKET_END for t in tasks):
        raise ValueError("Unexpected task outside V3 development")
    market = DevelopmentMarket(root / "data-expansion-v3/market", tasks)
    cases = []
    for task in tasks:
        state = build_workflow(market.view(task), MomentumBackend(BUDGET), BUDGET).invoke({})
        if state["errors"] or not state.get("decision"):
            raise ValueError("Offline development workflow failed")
        cases.append(make_case(task, "v3-development", state["report"],
                               Decision.model_validate(state["decision"]), market.outcome(task), state["evidence"]))
    return cases, {"source": "V3 Development retrospective vendor bars with offline MomentumBackend replay",
                   "tasks_sha256": file_hash(tasks_path), "last_read_market_date": str(DEV_MARKET_END),
                   **market.fingerprints}


def chronological(cases, warmup, calibration):
    days = sorted({case.episode.decision_time for case in cases})
    thresholds = days[warmup], days[warmup + calibration]
    result = {"warmup": [], "calibration": [], "validation": []}
    for case in sorted(cases, key=lambda c: (c.episode.decision_time, c.episode.source_task_id)):
        ts = case.episode.decision_time
        split = "warmup" if ts < thresholds[0] else "calibration" if ts < thresholds[1] else "validation"
        result[split].append(case)
    return result


def relevance(current, past):
    """Descriptive regime proxy, using no current decision or future settlement."""
    match = (current.momentum == past.momentum, current.high_volatility == past.high_volatility)
    return 3 if all(match) else 2 if match[0] else 1 if match[1] else 0
