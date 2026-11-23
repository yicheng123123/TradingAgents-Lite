"""Versioned evidence records and a date-bound read interface for future retrieval."""
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, AwareDatetime, Field, field_validator, model_validator

from .data import NY, close_available, sha256
from .schema import Contract, Task

SourceType = Literal["sec_filing", "news"]
PITLevel = Literal["historical_proxy", "verified_known_at", "unavailable"]
LOOKBACK = {"sec_filing": 180, "news": 30}


class Evidence(Contract):
    ticker: str = Field(pattern=r"^[A-Z][A-Z0-9.-]{0,9}$")
    source_type: SourceType
    document_id: str = Field(min_length=1)
    source_url: AnyHttpUrl
    published_at: AwareDatetime | None = None
    first_seen_at: AwareDatetime | None = None
    ingested_at: AwareDatetime
    availability_basis: str = Field(min_length=1)
    eligible_at: AwareDatetime | None = None
    decision_cutoff: AwareDatetime | None = None
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_metadata: dict = Field(default_factory=dict)
    excerpt: str = Field(default="", max_length=1800)
    pit_level: PITLevel

    @field_validator("published_at", "first_seen_at", "ingested_at", "eligible_at", "decision_cutoff")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def check_availability(self):
        if self.pit_level == "unavailable":
            if self.eligible_at is not None:
                raise ValueError("Unavailable evidence cannot have an eligibility time")
            return self
        if self.eligible_at is None or not self.excerpt:
            raise ValueError("Usable evidence requires an eligibility time and excerpt")
        if any(t is not None and t > self.eligible_at for t in (self.published_at, self.first_seen_at)):
            raise ValueError("Eligibility precedes known publication or observation")
        if self.pit_level == "verified_known_at" and (
            self.first_seen_at is None or
            self.source_metadata.get("version_attestation_sha256") != self.content_hash or
            not self.source_metadata.get("version_attestation_source")
        ):
            raise ValueError("Verified status requires a provenance attestation for this exact version")
        if self.decision_cutoff is not None and self.eligible_at > self.decision_cutoff:
            raise ValueError("Future evidence cannot be bound to a historical decision")
        return self

    @property
    def evidence_id(self) -> str:
        key = json.dumps([self.ticker, self.source_type, self.document_id, self.content_hash])
        return "fin:" + hashlib.sha256(key.encode()).hexdigest()


def next_session_proxy(day: date, sessions: tuple[date, ...]) -> datetime | None:
    """Next observed US session at 16:15 New York, fail closed past calendar coverage."""
    if not sessions or day < sessions[0] - timedelta(days=7):
        return None
    following = next((item for item in sessions if item > day), None)
    return close_available(following) if following else None


class EvidenceView:
    """Only eligible versions and safe provenance fields enter the analyst graph."""

    def __init__(self, task: Task, records: tuple[Evidence, ...]):
        self.cutoff = close_available(task.decision_date)
        visible = []
        candidate_versions = {}
        for item in records:
            if item.ticker != task.symbol or item.eligible_at is None or item.pit_level == "unavailable":
                continue
            if self.cutoff - timedelta(days=LOOKBACK[item.source_type]) <= item.eligible_at <= self.cutoff:
                key = (item.source_type, item.document_id, item.eligible_at)
                candidate_versions.setdefault(key, []).append(item)
        for versions in candidate_versions.values():
            if len(versions) == 1:
                visible.extend(versions)
                continue
            attested = [v for v in versions if v.pit_level == "verified_known_at"]
            visible.extend(attested)
        self._items = tuple(sorted(visible, key=lambda i: (i.eligible_at, i.evidence_id), reverse=True))

    def search(self, source_type: SourceType, limit: int = 3) -> dict:
        if source_type not in LOOKBACK or not 1 <= limit <= 4:
            raise ValueError("Unsupported evidence type or limit")
        rows = []
        for i in (r for r in self._items if r.source_type == source_type):
            item = i.model_dump(mode="json", exclude={"ingested_at", "decision_cutoff"})
            # Ingestion/cache/dataset details stay in the snapshot manifest. These
            # fields describe the individual historical document, not the corpus.
            item["source_metadata"] = {
                key: value for key, value in item["source_metadata"].items()
                if key in {"title", "publisher", "form", "report_date", "filing_date", "accepted_at",
                           "published_date", "public_available_at"}
            }
            rows.append({**item, "evidence_id": i.evidence_id,
                         "decision_cutoff": self.cutoff.isoformat()})
            if len(rows) == limit:
                break
        return {"status": "available" if rows else "unavailable", "source_type": source_type,
                "reason": None if rows else "no_eligible_evidence_in_snapshot",
                "items": rows}

    def visible(self, source_type: SourceType) -> tuple[Evidence, ...]:
        return tuple(item for item in self._items if item.source_type == source_type)


class EvidenceStore:
    """Small JSONL repository. Stable IDs and temporal predicates also suit a vector index."""

    def __init__(self, records: list[Evidence]):
        unique = {}
        for item in records:
            existing = unique.get(item.evidence_id)
            if existing is not None:
                comparable = {"ingested_at"}
                if existing.model_dump(exclude=comparable) != item.model_dump(exclude=comparable):
                    raise ValueError("Conflicting metadata for the same content version")
                continue
            unique[item.evidence_id] = item
        self.records = tuple(unique.values())

    def view(self, task: Task) -> EvidenceView:
        return EvidenceView(task, self.records)

    @classmethod
    def load(cls, root: Path):
        manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
        path = root / "evidence.jsonl"
        if manifest["evidence_sha256"] != sha256(path):
            raise ValueError("Evidence snapshot SHA-256 mismatch")
        store = cls([Evidence.model_validate_json(line) for line in path.read_text().splitlines()])
        if len(store.records) != manifest["count"]:
            raise ValueError("Evidence snapshot count mismatch")
        return store
