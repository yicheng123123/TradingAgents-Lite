"""Structured sidecars preserve the original Episode and separate facts from hypotheses."""
from contextlib import closing
from datetime import UTC, datetime
import re
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from ..memory import Episode
from ..rag import compact_json
from ..schema import Action, Contract, Outcome


class ProcessNotes(Contract):
    potential_error: str = Field(min_length=1, max_length=240)
    lesson: str = Field(min_length=1, max_length=240)
    uncertainty: str = Field(min_length=1, max_length=180)
    applicability: str = Field(min_length=1, max_length=180)


class StructuredReflection(Contract):
    version: Literal["a2-reflection-v2"] = "a2-reflection-v2"
    provenance: Literal["rule_replay", "model_replay", "model_live"]
    episode_version: str
    memory_id: str
    ticker: str
    situation: str
    decision_thesis: str
    action: Action
    decision_time: AwareDatetime
    outcome_summary: Outcome
    source_evidence_ids: tuple[str, ...]
    generated_at: AwareDatetime
    simulated_eligible_at: AwareDatetime | None = None
    eligible_at: AwareDatetime
    notes: ProcessNotes

    @model_validator(mode="after")
    def event_times(self):
        maturity = self.outcome_summary.available_at
        if not self.decision_time < maturity <= self.generated_at:
            raise ValueError("Reflection generation requires a mature outcome")
        visibility = self.simulated_eligible_at
        if self.provenance == "model_live":
            if visibility is not None or self.eligible_at != self.generated_at:
                raise ValueError("Live reflection must use the real availability time")
        elif visibility != self.eligible_at or not maturity <= visibility <= self.generated_at:
            raise ValueError("Historical replay requires a separate simulated time")
        return self

    def validate_episode(self, episode: Episode) -> None:
        expected = (episode.version_hash, episode.memory_id, episode.ticker, episode.action,
                    episode.decision_time, episode.outcome, tuple(episode.observed_evidence_ids),
                    episode.decision_rationale)
        actual = (self.episode_version, self.memory_id, self.ticker, self.action, self.decision_time,
                  self.outcome_summary, self.source_evidence_ids, self.decision_thesis)
        if actual != expected or self.eligible_at < episode.eligible_at:
            raise ValueError("Reflection facts diverge from the original episode")


def rule_notes(style: str, risk_veto: str | None) -> ProcessNotes:
    if risk_veto:
        lesson = "Review volatility and the reason for the risk veto before restoring exposure."
        applicability = "A directional signal conflicts with a binding risk limit."
    elif style == "flat":
        lesson = "Check the signal threshold and seek corroboration before taking exposure."
        applicability = "Five-session momentum is near the decision threshold."
    else:
        lesson = "Compare the momentum signal against volatility and the current risk limit."
        applicability = f"Five-session momentum is {style} and a directional position is considered."
    return ProcessNotes(
        potential_error="Possible overreliance on a short momentum window.",
        lesson=lesson,
        uncertainty="A single settled outcome cannot establish causality or decision quality.",
        applicability=applicability,
    )


def bind(episode: Episode, situation: str, notes: ProcessNotes, *, provenance="rule_replay",
         generated_at: datetime | None = None) -> StructuredReflection:
    generated = generated_at or datetime.now(UTC)
    replay = episode.eligible_at if provenance != "model_live" else None
    reflection = StructuredReflection(
        provenance=provenance, episode_version=episode.version_hash, memory_id=episode.memory_id,
        ticker=episode.ticker, situation=situation, decision_thesis=episode.decision_rationale,
        action=episode.action, decision_time=episode.decision_time, outcome_summary=episode.outcome,
        source_evidence_ids=tuple(episode.observed_evidence_ids), generated_at=generated,
        simulated_eligible_at=replay, eligible_at=replay or generated, notes=notes,
    )
    reflection.validate_episode(episode)
    return reflection


class ReflectionAdapter:
    """Persist a once-per-version attempt before asking an explicitly enabled A2 provider."""

    def __init__(self, path: Path, *, enabled: bool = False):
        self.path = path
        self.enabled = enabled
        with self._connect() as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS reflection (key TEXT PRIMARY KEY, result TEXT)")

    def _connect(self):
        return closing(sqlite3.connect(self.path, timeout=30))

    def generate(self, backend, episode: Episode, situation: str, *, replay=False):
        if not self.enabled or backend.phase != "A2" or backend.max_retries != 0:
            raise ValueError("Reflection requires explicit enablement and an A2 provider with zero transport retries")
        if datetime.now(UTC) < episode.eligible_at:
            raise ValueError("Episode is not mature")
        key = episode.version_hash
        provenance = "model_replay" if replay else "model_live"
        with self._connect() as db, db:
            claimed = db.execute("INSERT OR IGNORE INTO reflection VALUES (?, NULL)", (key,)).rowcount
            if not claimed:
                stored = db.execute("SELECT result FROM reflection WHERE key=?", (key,)).fetchone()[0]
                if stored is None:
                    raise RuntimeError("Reflection attempt already recorded without success")
                result = StructuredReflection.model_validate_json(stored)
                result.validate_episode(episode)
                if result.provenance != provenance or result.situation != situation:
                    raise ValueError("Cached reflection mode or situation differs")
                return result
        messages = [
            {"role": "system", "content": "Return JSON string fields potential_error, lesson, uncertainty, "
             "applicability. Write hypothetical process advice. Do not state causation, numerical facts, "
             "dates, or identifiers. A single outcome cannot establish decision quality."},
            {"role": "user", "content": compact_json({"episode": episode.model_dump(mode="json"),
                                                      "situation": situation})},
        ]
        for repair in (False, True):
            try:
                reply = backend._call(messages, [], structured_stage="memory_reflection_v2", repair=repair)
                notes = ProcessNotes.model_validate_json(reply["content"])
                if any(re.search(r"\d", text) for text in notes.model_dump().values()):
                    raise ValueError("Process notes must leave numeric facts to the deterministic envelope")
                break
            except (ValidationError, ValueError, KeyError):
                if repair:
                    raise
                messages = [*messages, {"role": "user", "content": "Repair the JSON schema and omit all numeric claims."}]
        result = bind(episode, situation, notes, provenance=provenance)
        with self._connect() as db, db:
            db.execute("UPDATE reflection SET result=? WHERE key=?", (result.model_dump_json(), key))
        return result
