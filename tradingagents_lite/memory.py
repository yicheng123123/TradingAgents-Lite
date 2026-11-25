"""Event-time episodic memory with SQLite authority and a recoverable vector index."""
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
import json
from math import exp
from pathlib import Path
import sqlite3
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from pydantic import AwareDatetime, Field, model_validator
from qdrant_client import models

from .data import close_available
from .documents import digest
from .rag import compact_json
from .schema import Action, Contract, Decision, Outcome
from .vectors import VectorStore, date_limit, match


class Reflection(Contract):
    potential_error: str = Field(min_length=1, max_length=450)
    lesson: str = Field(min_length=1, max_length=450)


class Episode(Contract):
    source_task_id: str
    ticker: str
    decision_time: AwareDatetime
    outcome_available_at: AwareDatetime
    reflection_created_at: AwareDatetime
    eligible_at: AwareDatetime
    action: Action
    observed_evidence_ids: list[str]
    decision_rationale: str = ""
    risk_veto: str | None = None
    evidence_quality: float | None = Field(default=None, ge=0, le=1)
    outcome: Outcome
    reflection: Reflection
    dataset_partition: str
    experiment_arm: str

    @model_validator(mode="after")
    def ordered(self):
        if (self.outcome_available_at != self.outcome.available_at
                or not self.decision_time < self.outcome_available_at
                <= self.reflection_created_at <= self.eligible_at):
            raise ValueError("Invalid episode event ordering")
        return self

    @property
    def memory_id(self) -> str:
        return str(uuid5(NAMESPACE_URL, compact_json([
            self.dataset_partition, self.experiment_arm, self.source_task_id, self.ticker,
            self.decision_time.timestamp()])))

    @property
    def version_hash(self) -> str:
        return digest(compact_json(self.model_dump(mode="json")).encode())

    @property
    def point_id(self) -> str:
        return str(uuid5(NAMESPACE_URL, self.memory_id + ":" + self.version_hash))


def deterministic_reflection(decision: Decision, outcome: Outcome) -> Reflection:
    alpha = decision.action.direction * (outcome.asset_return - outcome.benchmark_return)
    risk = ("A risk check changed the proposed action." if decision.risk_veto
            else "A single outcome cannot identify a process error.")
    if not decision.evidence_ids:
        risk += " The decision lacks observed evidence references."
    return Reflection(potential_error=risk,
        lesson=f"Observed relative return for the recorded action was {alpha:+.4f}. "
               "Compare the original thesis and its evidence across similar settled cases.")


def model_reflection(backend, decision: Decision, outcome: Outcome) -> Reflection:
    """Use the same budgeted provider entry as Analyst and Trader."""
    if backend.phase != "A2":
        raise ValueError("Bind LLM reflection to an A2 budget context")
    messages = [
        {"role": "system", "content": "Return JSON with strings potential_error and lesson. "
         "Discuss evidence and process. Profit or loss alone does not establish decision quality."},
        {"role": "user", "content": compact_json({
            "decision": decision.model_dump(mode="json"), "outcome": outcome.model_dump(mode="json")})},
    ]
    return Reflection.model_validate_json(backend._call(messages, [])["content"])


def _payload(episode: Episode) -> dict:
    return {"point_id": episode.point_id, "memory_id": episode.memory_id,
            "version_hash": episode.version_hash, "ticker": episode.ticker,
            "dataset_partition": episode.dataset_partition, "experiment_arm": episode.experiment_arm,
            "eligible_ts": episode.eligible_at.timestamp(),
            "outcome_ts": episode.outcome_available_at.timestamp(),
            "decision_ts": episode.decision_time.timestamp(),
            "text": f"{episode.ticker} {episode.action.value}. {episode.reflection.potential_error} "
                    f"{episode.reflection.lesson}"}


class MemoryStore:
    def __init__(self, root: Path, encoder):
        root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "episodes.sqlite"
        self.vectors = VectorStore(root / "vectors", encoder, "episodes")
        with self._db() as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS versions (
                version_hash TEXT PRIMARY KEY, memory_id TEXT NOT NULL,
                partition_name TEXT NOT NULL, arm TEXT NOT NULL, ticker TEXT NOT NULL,
                decision_ts REAL NOT NULL, outcome_ts REAL NOT NULL, eligible_ts REAL NOT NULL,
                record TEXT NOT NULL)""")
            db.execute("""CREATE INDEX IF NOT EXISTS memory_scope ON versions
                (partition_name, arm, ticker, eligible_ts)""")

    def _db(self):
        return closing(sqlite3.connect(self.db_path, timeout=30))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.vectors.client.close()

    def put(self, episode: Episode) -> bool:
        with self._db() as db, db:
            cursor = db.execute("INSERT OR IGNORE INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                episode.version_hash, episode.memory_id, episode.dataset_partition, episode.experiment_arm,
                episode.ticker, episode.decision_time.timestamp(), episode.outcome_available_at.timestamp(),
                episode.eligible_at.timestamp(), episode.model_dump_json()))
        # SQLite commits first. Repeating the same version repairs an interrupted Qdrant write.
        payload = _payload(episode)
        self.vectors.put([payload], id_key="point_id", text_key="text")
        return cursor.rowcount == 1

    def _all(self):
        with self._db() as db:
            return [Episode.model_validate_json(row[0]) for row in db.execute("SELECT record FROM versions")]

    def consistency(self) -> dict:
        expected = {ep.point_id: _payload(ep) for ep in self._all()}
        actual = {str(p.id): p.payload for p in self.vectors.scan()}
        return {"missing": sorted(expected.keys() - actual.keys()),
                "orphaned": sorted(actual.keys() - expected.keys()),
                "mismatched": sorted(key for key in expected.keys() & actual.keys()
                                     if actual[key] != expected[key])}

    def rebuild(self):
        state = self.consistency()
        repair = set(state["missing"] + state["mismatched"])
        self.vectors.put([_payload(ep) for ep in self._all() if ep.point_id in repair],
                         id_key="point_id", text_key="text")
        if state["orphaned"]:
            self.vectors.client.delete(self.vectors.name, models.PointIdsList(points=state["orphaned"]), wait=True)
        return self.consistency()

    def visible(self, *, ticker: str, cutoff: datetime, partition: str, arm: str) -> dict[str, Episode]:
        if cutoff.utcoffset() is None:
            raise ValueError("Memory cutoff must be timezone-aware")
        ts = cutoff.timestamp()
        with self._db() as db:
            rows = db.execute("""SELECT record FROM versions
                WHERE partition_name=? AND arm=? AND ticker=? AND decision_ts < ?
                    AND outcome_ts <= ? AND eligible_ts <= ?
                ORDER BY eligible_ts DESC, rowid DESC""", (partition, arm, ticker, ts, ts, ts))
            latest = {}
            for row in rows:
                ep = Episode.model_validate_json(row[0])
                latest.setdefault(ep.memory_id, ep)
        return {ep.point_id: ep for ep in latest.values()}

    def recall(self, query: str, *, ticker: str, cutoff: datetime, partition: str, arm: str,
               mode: str, limit: int = 3, context_bytes: int = 1800) -> dict:
        if mode not in {"M0", "M1", "M2", "M3"} or not 1 <= limit <= 10 or not 100 <= context_bytes <= 4800:
            raise ValueError("Invalid memory mode or limits")
        started = perf_counter()
        visible = {} if mode == "M0" else self.visible(ticker=ticker, cutoff=cutoff,
                                                        partition=partition, arm=arm)
        raw = []
        if visible and mode == "M1":
            raw = [(key, None) for key in list(visible)[:limit]]
        if visible and mode in {"M2", "M3"}:
            filters = models.Filter(must=[
                match("ticker", ticker), match("dataset_partition", partition), match("experiment_arm", arm),
                date_limit("eligible_ts", cutoff.timestamp()), date_limit("outcome_ts", cutoff.timestamp()),
                models.HasIdCondition(has_id=list(visible)),
            ])
            actual = {str(point.id): point.payload for point in self.vectors.scan(filters)}
            if any(actual.get(key) != _payload(ep) for key, ep in visible.items()):
                raise RuntimeError("Memory index is inconsistent; run rebuild")
            raw = [(str(p.id), max(-1., min(1., p.score)))
                   for p in self.vectors.search(query, filters, min(len(visible), max(24, limit)))]

        ranking = []
        for key, cosine in raw:
            ep = visible[key]
            age_days = max(0, (cutoff - ep.eligible_at).total_seconds() / 86400)
            semantic = (cosine + 1) / 2 if cosine is not None else None
            time = exp(-age_days / 30)
            evidence_score = ep.evidence_quality
            if evidence_score is None:
                evidence_score = min(len(set(ep.observed_evidence_ids)), 3) / 3
            score = time if mode == "M1" else semantic
            if mode == "M3":
                score = 0.65 * semantic + 0.25 * time + 0.10 * evidence_score
            ranking.append({"point_id": key, "memory_id": ep.memory_id, "version_hash": ep.version_hash,
                            "source_task_id": ep.source_task_id, "score": score, "semantic": semantic,
                            "recency": time, "evidence_score": evidence_score,
                            "historical_evidence_ids": ep.observed_evidence_ids})
        ranking.sort(key=lambda row: (-row["score"], row["point_id"]))
        selected = ranking[:limit]
        full = [_context(visible[row["point_id"]], compact=False) for row in selected]
        context, injected = [], []
        for row in selected:
            item = _context(visible[row["point_id"]], compact=mode == "M3")
            if len(compact_json([*context, item]).encode()) <= context_bytes:
                context.append(item)
                injected.append(row["memory_id"])
        return {"candidates": ranking, "selected": selected, "context": context, "injected_ids": injected,
                "eligible": len(visible), "before_bytes": len(compact_json(full).encode()),
                "context_bytes": len(compact_json(context).encode()),
                "latency_ms": (perf_counter() - started) * 1000}


def _context(ep: Episode, *, compact: bool) -> dict:
    item = {"memory_id": ep.memory_id, "historical_evidence_ids": ep.observed_evidence_ids,
            "action": ep.action.value, "eligible_at": ep.eligible_at.isoformat(),
            "lesson": ep.reflection.lesson, "potential_error": ep.reflection.potential_error}
    if compact:
        item["lesson"] = item["lesson"][:220]
        item["potential_error"] = item["potential_error"][:150]
    else:
        item["historical_outcome"] = ep.outcome.model_dump(mode="json")
        item["decision_time"] = ep.decision_time.isoformat()
    return item


@dataclass(frozen=True)
class MemoryConfig:
    mode: str
    partition: str
    arm: str
    limit: int = 3
    context_bytes: int = 1800


class MemoryReplay:
    """Advance settlement events before each decision. Outcomes stay in the runner-side queue."""
    def __init__(self, store: MemoryStore, config: MemoryConfig, reflector=None):
        self.store, self.config, self.reflector = store, config, reflector
        self.pending = []
        self.cutoff = None
        self.written = 0

    def advance(self, cutoff: datetime):
        if self.cutoff is not None and cutoff < self.cutoff:
            raise ValueError("Memory tasks must execute in increasing decision time")
        self.cutoff = cutoff
        pending = sorted(self.pending, key=lambda item: (item[2].available_at, item[0].task_id))
        self.pending = pending
        while self.pending and self.pending[0][2].available_at <= cutoff:
            task, decision, outcome, quality = self.pending[0]
            if self.reflector is None:
                reflection = deterministic_reflection(decision, outcome)
                # Deterministic replay schedules this calculation at the settlement event.
                created = outcome.available_at
            else:
                # A live reflector must supply the actual time its result became available.
                reflection, created = self.reflector(task, decision, outcome)
            episode = Episode(source_task_id=task.task_id, ticker=task.symbol,
                decision_time=close_available(task.decision_date), outcome_available_at=outcome.available_at,
                reflection_created_at=created, eligible_at=created,
                action=decision.action, observed_evidence_ids=decision.evidence_ids, outcome=outcome,
                reflection=reflection, dataset_partition=self.config.partition, experiment_arm=self.config.arm,
                decision_rationale=decision.rationale, risk_veto=decision.risk_veto, evidence_quality=quality)
            self.written += self.store.put(episode)
            self.pending.pop(0)

    def before(self, task, query: str) -> dict:
        cutoff = close_available(task.decision_date)
        self.advance(cutoff)
        return self.store.recall(query, ticker=task.symbol, cutoff=cutoff,
            partition=self.config.partition, arm=self.config.arm, mode=self.config.mode,
            limit=self.config.limit, context_bytes=self.config.context_bytes)

    def enqueue(self, task, decision: Decision, outcome: Outcome, evidence: dict | None = None):
        if self.config.mode != "M0":
            scores = []
            for evidence_id in set(decision.evidence_ids):
                item = (evidence or {}).get(evidence_id)
                if item is not None:
                    scores.append({"verified_known_at": 1., "historical_proxy": .5,
                                   "unavailable": 0.}.get(item.get("pit_level"), .75))
            quality = sum(scores) / len(scores) if scores else 0.
            self.pending.append((task, decision, outcome, quality))

    def finish(self):
        if self.pending:
            settlement_end = max(item[2].available_at for item in self.pending)
            self.advance(max(self.cutoff or settlement_end, settlement_end))
