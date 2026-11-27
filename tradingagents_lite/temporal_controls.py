"""Offline reidentification and private synthetic counterfactual controls."""
import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
from hashlib import sha256 as digest
import json
import math
from pathlib import Path
import secrets

from .anonymization import Projection, model_context
from .data import HistoricalData, close_available, read_tasks, sha256
from .data_v2_market import ISSUERS, write_json
from .rag import bm25
from .schema import Task
from .verifier import verify_citations


def fingerprint(values):
    return tuple(round(current / prior - 1, 7) for prior, current in zip(values, values[1:]))


def attack(market, tasks):
    """An adversary searches all eight-close shapes without the decision date or symbol."""
    development = [task for task in tasks if not market.is_sealed(task)]
    last_development = max(task.decision_date for task in development)
    index = defaultdict(set)
    for symbol in ISSUERS:
        closes = [bar.close for bar in market.series[symbol] if bar.day <= last_development]
        for i in range(len(closes) - 7):
            index[fingerprint(closes[i:i + 8])].add(symbol)
    exact_matches, unique_hits, literal_leaks, ratio_checks = 0, 0, 0, 0
    example_key = secrets.token_bytes(32)
    for task in development:
        bars = market.view(task).get_bars(8)
        closes = [bar["close"] for bar in bars["bars"]]
        projection = Projection(secret=example_key, case=task.task_id, ticker=task.symbol,
                                decision=task.decision_date, close=closes[-1])
        probe = model_context("P2", ticker=task.symbol, company=None, decision=task.decision_date,
                              market=bars, projection=projection)["market"]
        shown = [bar["close"] for bar in probe["bars"]]
        matches = index[fingerprint(shown)]
        exact_matches += task.symbol in matches
        unique_hits += matches == {task.symbol}
        visible = json.dumps(probe)
        literal_leaks += task.symbol in visible or str(task.decision_date) in visible
        ratio_checks += math.isclose(closes[-1] / closes[0], shown[-1] / shown[0], rel_tol=1e-12)
    return {"development_cases": len(development), "literal_ticker_or_date_leaks": literal_leaks,
            "fingerprint_contains_true_issuer": exact_matches, "fingerprint_unique_issuer_hits": unique_hits,
            "return_invariant": ratio_checks, "attack_window": 8,
            "reference": "all 20 issuers through the last Development date, no Holdout labels",
            "baseline_known_identity": len(development),
            "privacy_claim": "literal identity is removed; public return patterns remain linkable",
            "information_loss": ["absolute prices and dates", "original titles and prose",
                                 "unstructured Memory and agent reasoning", "URLs and source transport metadata"]}


def _synthetic_doc(kind, payload, when):
    encoded = json.dumps(payload, sort_keys=True)
    doc_id = f"synthetic:{kind}"
    content_hash = digest(encoded.encode()).hexdigest()
    eid = "fin:" + digest(json.dumps(["SYNTH", "synthetic", doc_id, content_hash]).encode()).hexdigest()
    return {"evidence_id": eid, "ticker": "SYNTH", "source_type": "synthetic", "document_id": doc_id,
            "content_hash": content_hash, "pit_level": "verified_known_at", "dataset_partition": "synthetic-v3c",
            "eligible_at": when.isoformat(), "kind": kind, "payload": payload}


def synthetic_cases():
    task = Task(task_id="synthetic-v3c", symbol="SYNTH", decision_date=date(2026, 6, 10), horizon_sessions=5)
    now = close_available(task.decision_date)
    for name, revenue, cost, state, avail in (
        ("base", 120, 90, "guidance_raised", True),
        ("changed_numbers", 81, 96, "guidance_raised", True),
        ("changed_event", 120, 90, "guidance_withdrawn", True),
        ("future_evidence", 120, 90, "guidance_raised", False),
    ):
        if avail:
            eligible = now - timedelta(days=1)
        else:
            eligible = now + timedelta(days=1)
        yield {"synthetic": True, "case": name, "task": task.model_dump(mode="json"), "documents": [
            _synthetic_doc("earnings financial amounts", {"revenue": revenue, "cost": cost}, eligible),
            _synthetic_doc("guidance business event", {"event": state}, eligible),
        ]}


def evaluate_synthetic(case):
    task = Task.model_validate(case["task"])
    docs = {item["evidence_id"]: item for item in case["documents"]}
    cutoff = close_available(task.decision_date)
    eligible = {eid: item for eid, item in docs.items()
                if datetime.fromisoformat(item["eligible_at"]) <= cutoff}
    ranked = [{"chunk_id": eid, "text": row["kind"]} for eid, row in eligible.items()]
    steps = ["retrieve earnings"]
    earnings = bm25("earnings financial amounts", ranked, 1)
    ratio = None
    used = []
    action = "hold"
    if earnings:
        eid = earnings[0][0]
        figures = eligible[eid]["payload"]
        used.append(eid)
        ratio = (figures["revenue"] - figures["cost"]) / figures["revenue"]
        if ratio > 0:
            steps.append("retrieve guidance")
            guidance = bm25("guidance business event", ranked, 1)
            if guidance:
                guid = guidance[0][0]
                used.append(guid)
                if eligible[guid]["payload"]["event"] == "guidance_raised":
                    action = "buy"
        else:
            action = "sell"
    accepted = verify_citations(used, eligible, set(eligible), task, partition="synthetic-v3c")
    rejected = [eid for eid in docs if eid not in eligible]
    future_result = verify_citations(rejected, docs, set(docs), task, partition="synthetic-v3c")
    return {"action": action, "margin": ratio, "steps": steps, "cited": len(used),
            "verifier_passed": accepted["passed"],
            "future_rejected": bool(rejected) and "future_evidence" in future_result["violations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path, default=Path("data/data-expansion-v3/market"))
    parser.add_argument("--tasks", type=Path, default=Path("tasks/data-expansion-v3/development.jsonl"))
    parser.add_argument("--private", type=Path, default=Path("data/data-expansion-v3/counterfactual.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = list(synthetic_cases())
    results = [evaluate_synthetic(case) for case in cases]
    if args.private.exists():
        raise FileExistsError("Private synthetic controls already frozen")
    write_json(args.private, [{**case, "gold": result} for case, result in zip(cases, results, strict=True)])
    report = attack(HistoricalData(args.market), read_tasks(args.tasks))
    report.update(private_synthetic_sha256=sha256(args.private), synthetic_cases=len(cases),
                  synthetic_verifier_passes=sum(row["verifier_passed"] for row in results),
                  synthetic_future_rejections=sum(row["future_rejected"] for row in results),
                  synthetic_distinct_plans=len({tuple(row["steps"]) for row in results}),
                  synthetic_changed_decisions=sum(row["action"] != results[0]["action"] for row in results[1:]))
    write_json(args.out, report)


if __name__ == "__main__":
    main()
