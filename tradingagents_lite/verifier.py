"""Deterministic admissibility checks for citations and proposals."""
from hashlib import sha256
from datetime import datetime
import json
import re

from pydantic import ValidationError

from .data import close_available
from .schema import Action, Proposal, Task


def verdict(violations: list[str]) -> dict:
    unique = list(dict.fromkeys(violations))
    hard = {"future_evidence", "evidence_scope", "evidence_version", "evidence_time_missing"}
    return {"passed": not unique, "violations": unique, "error_code": unique[0] if unique else None,
            "repairable": bool(unique) and not hard.intersection(unique),
            "suggested_action": "hold" if unique else "risk_review",
            "citation_validity": "failed" if unique else "passed",
            "citation_entailment": "not_assessed"}


def verify_citations(ids: list[str], evidence: dict, context_ids: set[str], task: Task, *,
                     partition: str = "development", historical_ids: set[str] = frozenset()) -> dict:
    cutoff = close_available(task.decision_date)
    problems = []
    for key in ids:
        row = evidence.get(key)
        if row is None:
            problems.append("historical_memory_citation" if key in historical_ids else "unknown_citation")
            continue
        if key not in context_ids:
            problems.append("citation_not_in_context")
        if row.get("evidence_id") != key or row.get("ticker", task.symbol) != task.symbol:
            problems.append("evidence_scope")
        if row.get("dataset_partition", partition) != partition:
            problems.append("evidence_scope")
        if key.startswith(("fin:", "chunk:")):
            candidate = row.get("eligible_at")
            try:
                time = datetime.fromisoformat(candidate)
                if time.utcoffset() is None:
                    raise ValueError
                if time > cutoff:
                    problems.append("future_evidence")
            except (TypeError, ValueError):
                problems.append("evidence_time_missing")
            if row.get("pit_level") not in {"verified_known_at", "historical_proxy"}:
                problems.append("evidence_scope")
            if key.startswith("fin:"):
                fields = [row.get(k) for k in ("ticker", "source_type", "document_id", "content_hash")]
                digest = sha256(json.dumps(fields).encode()).hexdigest()
                if None in fields or key != "fin:" + digest or not re.fullmatch(r"[a-f0-9]{64}", fields[-1]):
                    problems.append("evidence_version")
            elif all(re.fullmatch(r"[a-f0-9]{64}", str(row.get(k))) for k in ("version_id", "raw_hash")):
                if candidate and "evidence_time_missing" not in problems:
                    fields = [row.get("dataset_partition"), row.get("ticker"), row.get("source_type"),
                              row.get("document_id"), row["raw_hash"], time.timestamp()]
                    expected = sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()
                    if None in fields or row["version_id"] != expected:
                        problems.append("evidence_version")
            else:
                problems.append("evidence_version")
        elif not key.startswith(f"{task.symbol}:{task.decision_date}:"):
            problems.append("evidence_scope")
        for bar in row.get("bars", []):
            try:
                if (bar["day"] > task.decision_date.isoformat()
                        or datetime.fromisoformat(bar["available_at"]) > cutoff):
                    problems.append("future_evidence")
            except (KeyError, TypeError, ValueError):
                problems.append("evidence_time_missing")
    return verdict(problems)


def verify_proposal(raw, evidence, context_ids, task, *, errors=(), volatility=None,
                    volatility_limit=None, **scope) -> dict:
    if errors:
        return {**verdict(["upstream_failure"]), "repairable": False}
    try:
        proposal = Proposal.model_validate(raw)
    except ValidationError:
        return verdict(["invalid_proposal_schema"])
    if proposal.action != Action.HOLD and not proposal.evidence_ids:
        return verdict(["direction_without_citations"])
    result = verify_citations(proposal.evidence_ids, evidence, context_ids, task, **scope)
    if (result["passed"] and proposal.action != Action.HOLD and volatility is not None
            and volatility_limit is not None and volatility > volatility_limit):
        result["suggested_action"] = "hold"
        result["risk_reason"] = "volatility_limit"
    return result
