"""Keep SQLite authoritative and index each representation in a separate collection."""
from contextlib import ExitStack
from math import exp
from pathlib import Path

from qdrant_client import models

from ..memory import Episode, MemoryStore, _payload
from ..vectors import VectorStore, date_limit, match
from .reflection import StructuredReflection

REPRESENTATIONS = ("E0", "E1", "E2")
RANKS = {
    "R0": (0, 1, 0), "R1": (1, 0, 0), "R2": (.75, .25, 0),
    "R3": (.9, 0, .1), "R4": (.65, .25, .1),
}


def representation(episode, reflection, mode):
    legacy = _payload(episode)["text"]
    if mode == "E0":
        return legacy
    if mode == "E1":
        return f"{legacy} Decision thesis: {episode.decision_rationale}"
    if mode != "E2":
        raise ValueError("Unknown memory representation")
    notes = reflection.notes
    return (f"{episode.ticker}. {reflection.situation} {reflection.decision_thesis} "
            f"{notes.potential_error} {notes.lesson} {notes.applicability}")


def eligible(episode, reflection, *, cutoff, partition, arm, tickers):
    if cutoff.utcoffset() is None:
        raise ValueError("Memory cutoff must be timezone-aware")
    return (episode.dataset_partition == partition and episode.experiment_arm == arm
            and episode.ticker in tickers and episode.decision_time < cutoff
            and episode.outcome_available_at <= cutoff and episode.reflection_created_at <= cutoff
            and episode.eligible_at <= cutoff and reflection.eligible_at <= cutoff)


class ExperimentStore:
    def __init__(self, root: Path, encoder):
        self.root = root
        self.encoder = encoder
        self.resources = ExitStack()

    def __enter__(self):
        try:
            self.authority = self.resources.enter_context(MemoryStore(self.root, self.encoder))
            self.indexes = {"E0": self.authority.vectors}
            for mode in REPRESENTATIONS[1:]:
                self.indexes[mode] = self.resources.enter_context(
                    VectorStore(self.root / mode, self.encoder, "episodes"))
            with self.authority._db() as db, db:
                db.execute("CREATE TABLE IF NOT EXISTS structured (point_id TEXT PRIMARY KEY, record TEXT)")
            return self
        except BaseException:
            self.resources.close()
            raise

    def __exit__(self, *exc):
        self.resources.close()

    def put_many(self, records):
        records = list(records)
        for ep, ref in records:
            ref.validate_episode(ep)
        # Commit episode + sidecar together, then update each derived index.
        with self.authority._db() as db, db:
            for ep, ref in records:
                db.execute("INSERT OR IGNORE INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    ep.version_hash, ep.memory_id, ep.dataset_partition, ep.experiment_arm,
                    ep.ticker, ep.decision_time.timestamp(), ep.outcome_available_at.timestamp(),
                    ep.eligible_at.timestamp(), ep.model_dump_json()))
                prior = db.execute("SELECT record FROM structured WHERE point_id=?", (ep.point_id,)).fetchone()
                if prior:
                    old = StructuredReflection.model_validate_json(prior[0])
                    if old.model_dump(exclude={"generated_at"}) != ref.model_dump(exclude={"generated_at"}):
                        raise ValueError("A sidecar already exists for this frozen episode version")
                else:
                    db.execute("INSERT INTO structured VALUES (?, ?)", (ep.point_id, ref.model_dump_json()))
        self._index(records)

    def _index(self, records):
        for mode, index in self.indexes.items():
            index.put([self._payload_for(ep, ref, mode) for ep, ref in records],
                      id_key="point_id", text_key="text")

    @staticmethod
    def _payload_for(episode, reflection, mode):
        payload = _payload(episode)
        return {**payload, "text": representation(episode, reflection, mode)}

    def records(self):
        with self.authority._db() as db:
            refs = {point: StructuredReflection.model_validate_json(record)
                    for point, record in db.execute("SELECT point_id, record FROM structured")}
        records = [(ep, refs[ep.point_id]) for ep in self.authority._all() if ep.point_id in refs]
        if len(records) != len(refs):
            raise RuntimeError("Structured sidecar without an authoritative Episode")
        for ep, ref in records:
            ref.validate_episode(ep)
        return records

    def consistency(self):
        records = self.records()
        result = {}
        for mode, index in self.indexes.items():
            expected = {ep.point_id: self._payload_for(ep, ref, mode) for ep, ref in records}
            actual = {str(point.id): point.payload for point in index.scan()}
            result[mode] = {"missing": sorted(expected.keys() - actual.keys()),
                            "orphaned": sorted(actual.keys() - expected.keys()),
                            "mismatched": sorted(key for key in expected.keys() & actual.keys()
                                                 if expected[key] != actual[key])}
        return result

    def rebuild(self):
        records = self.records()
        self._index(records)
        for mode, deviations in self.consistency().items():
            if deviations["orphaned"]:
                index = self.indexes[mode]
                index.client.delete(index.name, models.PointIdsList(points=deviations["orphaned"]), wait=True)
        return self.consistency()

    def scoped(self, *, ticker, cutoff, partition, arm, allowed_tickers=None):
        tickers = set(allowed_tickers) if allowed_tickers is not None else {ticker}
        visible = {}
        for ep, ref in self.records():
            if eligible(ep, ref, cutoff=cutoff, partition=partition, arm=arm, tickers=tickers):
                previous = visible.get(ep.memory_id)
                if previous is None or (ref.eligible_at, ep.version_hash) > (
                    previous[1].eligible_at, previous[0].version_hash
                ):
                    visible[ep.memory_id] = (ep, ref)
        return {ep.point_id: (ep, ref) for ep, ref in visible.values()}

    def cosine(self, query, mode, scoped, cutoff, partition, arm):
        if not scoped:
            return {}
        index = self.indexes[mode]
        filters = models.Filter(must=[
            models.HasIdCondition(has_id=list(scoped)),
            match("dataset_partition", partition), match("experiment_arm", arm),
            date_limit("eligible_ts", cutoff.timestamp()), date_limit("outcome_ts", cutoff.timestamp()),
        ])
        rows = index.search(query, filters, len(scoped))
        if {str(row.id) for row in rows} != scoped.keys() or any(
            row.payload != self._payload_for(*scoped[str(row.id)], mode) for row in rows
        ):
            raise RuntimeError("Memory index is inconsistent; run rebuild")
        return {str(row.id): max(-1.0, min(1.0, row.score)) for row in rows}


def rank(scoped, cosine, cutoff, mode, *, candidate_limit=24):
    weights = RANKS[mode]
    keys = list(scoped)
    if mode != "R0":
        keys = sorted(keys, key=lambda key: (-cosine[key], key))[:candidate_limit]

    def score(key):
        ep, ref = scoped[key]
        age_days = (cutoff - ref.eligible_at).total_seconds() / 86400
        semantic = (cosine[key] + 1) / 2 if mode != "R0" else 0
        recency = exp(-age_days / 30)
        quality = ep.evidence_quality
        if quality is None:
            quality = min(len(set(ep.observed_evidence_ids)), 3) / 3
        return sum(weight * value for weight, value in zip(weights, (semantic, recency, quality), strict=True))

    return sorted(keys, key=lambda key: (-score(key), key))
