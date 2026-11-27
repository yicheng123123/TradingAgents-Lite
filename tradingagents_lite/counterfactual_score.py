"""Independent deterministic scorer for frozen Counterfactual predictions."""

import argparse
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
from pathlib import Path
import re

from .counterfactual_runner import _hash, _verify_sealed
from .data import sha256 as file_sha


NUMBER = re.compile(
    r"(?P<open>\()?(?P<sign>[+-]?)\s*(?P<prefix_currency>USD|\$)?\s*"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)\s*"
    r"(?:(?P<currency_first>USD|\$)\s*(?P<currency_scale>thousands?|millions?|billions?|trillions?|[kmbt])?"
    r"|(?P<scale>thousands?|millions?|billions?|trillions?|[kmbt])\s*(?P<scale_currency>USD|\$)?"
    r"|(?P<unit>percent(?:s|age)?|%))?\s*(?P<close>\))?",
    re.IGNORECASE,
)
CENT = Decimal("0.01")
SCALES = {"thousand": Decimal("0.001"), "k": Decimal("0.001"),
          "million": Decimal("1"), "m": Decimal("1"),
          "billion": Decimal("1000"), "b": Decimal("1000"),
          "trillion": Decimal("1000000"), "t": Decimal("1000000")}


def parse_answer(text, unit):
    """Return one value in protocol units; ambiguous multi-number prose is invalid."""
    if not isinstance(text, str):
        return None
    value = text.strip().replace("−", "-").removesuffix(".")
    match = NUMBER.fullmatch(value)
    if not match or bool(match["open"]) != bool(match["close"]):
        return None
    if match["open"] and match["sign"]:
        return None
    try:
        number = Decimal(match["number"].replace(",", ""))
    except InvalidOperation:
        return None
    if match["sign"] == "-" or match["open"]:
        number = -number
    if not number.is_finite():
        return None
    suffix = (match["currency_scale"] or match["scale"] or "").lower()
    if suffix.endswith("s"):
        suffix = suffix[:-1]
    requested = unit.casefold()
    explicit_percent = bool(match["unit"])
    currencies = [match[key] for key in ("prefix_currency", "currency_first", "scale_currency") if match[key]]
    if len(currencies) > 1:
        return None
    has_currency = bool(currencies)
    if requested == "percent":
        if has_currency or (suffix and suffix not in {"k"}):
            return None
        # A percent suffix and a bare numeric value are both in percentage points;
        # fractional ratios are not silently converted to percent.
        return number.quantize(CENT, rounding=ROUND_HALF_UP)
    if requested == "usd millions":
        if explicit_percent:
            return None
        if suffix:
            scale = SCALES.get(suffix)
            if scale is None:
                return None
            number *= scale
        elif abs(number) >= Decimal("1000000"):
            # Plain, explicit dollar figures are absolute amounts. Values in the
            # requested USD-millions unit (typically thousands) remain unchanged.
            number /= Decimal("1000000")
        return number.quantize(CENT, rounding=ROUND_HALF_UP)
    return None


def _correct(record, reference):
    status = record.get("answer_status")
    if record.get("status") != "ok":
        return False
    if reference["status"] == "abstained":
        return status == "abstained" and record.get("answer") == ""
    if status != "answered":
        return False
    predicted = parse_answer(record.get("answer"), reference["unit"])
    expected = Decimal(reference["value"]).quantize(CENT, rounding=ROUND_HALF_UP)
    return predicted == expected


def score_run(data_root, run_root, output):
    data_root, run_root, output = map(Path, (data_root, run_root, output))
    manifest_path = run_root / "manifest.json"
    prediction_path = run_root / "predictions.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    for relative, digest in source_manifest.get("sha256", {}).items():
        if file_sha(data_root / relative) != digest:
            raise ValueError("frozen Development dataset hash mismatch")
    if (manifest.get("backend") not in {"mock", "llm"} or manifest.get("status") != "complete"
            or not prediction_path.is_file()
            or file_sha(prediction_path) != manifest.get("predictions_sha256")):
        raise ValueError("predictions are not complete and frozen")
    records = []
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            _verify_sealed(row)
            records.append(row)
    expected_plan = {f"{cell['arm']}:{cell['id']}" for cell in manifest["plan"]}
    identities = [row.get("task_id") for row in records]
    if len(identities) != len(set(identities)) or set(identities) != expected_plan:
        raise ValueError("prediction set differs from the frozen plan")

    # This scorer is a separate command/process. Gold is read only after the complete
    # prediction file and plan have passed the immutable-hash checks above.
    references = {row["task_id"]: row for row in (
        json.loads(line) for line in (data_root / "scoring/reference.jsonl").read_text(encoding="utf-8").splitlines()
        if line)}
    if set(references) != {cell["id"] for cell in manifest["plan"]}:
        raise ValueError("scoring references do not exactly match frozen task IDs")
    input_hashes = {cell["id"]: cell["input_sha256"] for cell in manifest["plan"]}
    if any(row.get("input_sha256") != input_hashes.get(row.get("id")) for row in records):
        raise ValueError("predictions do not match the frozen task input hashes")
    future_ids = {}
    for task_id in references:
        authoring = json.loads((data_root / "authoring" / f"{task_id}.json").read_text(encoding="utf-8"))
        cutoff = authoring["cutoff"]
        cutoff_time = datetime.fromisoformat(cutoff)
        future_ids[task_id] = {"cf:" + doc["version_id"] for doc in authoring["documents"]
                               if datetime.fromisoformat(doc["eligible_at"]) > cutoff_time}

    scored = []
    for record in records:
        task_id = record["id"]
        reference = references[task_id]
        citations = set(record.get("citations", []))
        exposed = {item.get("evidence_id") for item in record.get("exposed", [])}
        valid_citations = bool(citations <= exposed)
        correct = _correct(record, reference) and valid_citations
        delivered_ids = {item.get("evidence_id") for trace in record.get("retrieval_trace", [])
                         for item in trace.get("delivered_evidence", [])}
        required_ids = {"cf:" + version for version in reference.get("required_versions", [])}
        planning = reference.get("planning", {})
        route_id = "cf:" + planning["routing_version"] if planning else None
        target_id = "cf:" + planning["target_version"] if planning else None
        rounds = record.get("retrieval_trace", [])
        adaptive = False
        if planning and len(rounds) >= 2:
            first_delivered = {item["evidence_id"] for item in rounds[0].get("delivered_evidence", [])}
            initial_query = rounds[0].get("query", "")
            followup_query = rounds[1].get("query", "")
            second_delivered = {item["evidence_id"] for item in rounds[1].get("delivered_evidence", [])}
            adaptive = (route_id in first_delivered and planning["approved_reference"].casefold() in followup_query.casefold()
                        and planning["approved_reference"].casefold() not in initial_query.casefold()
                        and target_id in second_delivered)
        future_citations = citations & future_ids.get(task_id, set())
        try:
            predicted_value = parse_answer(record.get("answer"), reference["unit"])
            future_values = {Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)
                             for value in reference.get("future_only_values", [])}
        except (InvalidOperation, TypeError):
            predicted_value, future_values = None, set()
        future_value_match = predicted_value in future_values and predicted_value != (
            Decimal(reference["value"]).quantize(CENT, rounding=ROUND_HALF_UP)
            if reference.get("value") is not None else None)
        scored.append({
            "task_id": task_id, "mother_id": reference["mother_id"], "variant": reference["variant"],
            "category": reference["category"], "arm": record["arm"], "status": record.get("status"),
            "answer_correct": correct, "citation_valid": valid_citations,
            "required_evidence_complete": required_ids <= delivered_ids,
            "required_evidence_cited": required_ids <= citations,
            "planning_success": bool(planning and target_id in delivered_ids and target_id in citations and valid_citations),
            "adaptive_planning_success": adaptive,
            "abstention_correct": (
                (record.get("answer_status") == "abstained" and record.get("answer") == "")
                if reference["status"] == "abstained" else correct
            ) if reference["category"] == "time_conflict_or_missing" else None,
            "future_evidence_cited": bool(future_citations), "future_value_match": bool(future_value_match),
            "future_acceptance_observed": bool(future_citations or future_value_match),
            "future_candidate_count": len(future_ids.get(task_id, set())),
            "tokens": int(record.get("budget_tokens") or 0),
            "tool_calls": int(record.get("tool_calls") or 0),
            "backend_calls": len(record.get("llm_calls") or []),
            "http_requests": sum(
                call.get("actual_model") not in {None, "mock-counterfactual"}
                for call in (record.get("llm_calls") or [])
            ),
            "latency_seconds": float(record.get("latency_seconds") or 0),
        })

    by_arm = defaultdict(list)
    for row in scored:
        by_arm[row["arm"]].append(row)
    summary = {}
    for arm in ("CF0", "CF1"):
        rows = by_arm[arm]
        time_rows = [row for row in rows if row["category"] == "time_conflict_or_missing"]
        planning_rows = [row for row in rows if row["category"] == "dynamic_planning"]
        future_rows = [row for row in rows if row["future_candidate_count"]]
        summary[arm] = {
            "answer_accuracy": {"correct": sum(row["answer_correct"] for row in rows), "total": len(rows)},
            "counterfactual_pair_accuracy": {
                "correct_mothers": sum(all(row["answer_correct"] for row in rows
                                           if row["mother_id"] == mother)
                                        and sum(row["mother_id"] == mother for row in rows) == 2
                                        for mother in {row["mother_id"] for row in rows}),
                "total_mothers": len({row["mother_id"] for row in rows}),
                "evidence_complete_mothers": sum(
                    sum(row["mother_id"] == mother and row["required_evidence_cited"] for row in rows) == 2
                    for mother in {row["mother_id"] for row in rows}),
            },
            "planning_success_rate": {"success": sum(row["planning_success"] for row in planning_rows),
                                      "total": len(planning_rows)},
            "adaptive_planning_success_rate": {"success": sum(row["adaptive_planning_success"] for row in planning_rows),
                                               "total": len(planning_rows)} if arm == "CF1" else None,
            "abstention_accuracy": {"correct": sum(row["abstention_correct"] is True for row in time_rows),
                                    "total": len(time_rows)},
            "future_evidence_acceptance_rate": {
                "observed_acceptances": sum(row["future_acceptance_observed"] for row in future_rows),
                "tasks_with_future_candidates": len(future_rows),
                "failed_or_unclassifiable": sum(row["status"] != "ok" for row in future_rows),
                "future_citations": sum(row["future_evidence_cited"] for row in future_rows),
                "future_value_matches": sum(row["future_value_match"] for row in future_rows),
            },
            "citation_validity": {"valid_tasks": sum(row["citation_valid"] for row in rows), "total": len(rows)},
            "tokens": {"total": sum(row["tokens"] for row in rows),
                       "mean_per_task": sum(row["tokens"] for row in rows) / len(rows) if rows else 0},
            "tool_calls": {"total": sum(row["tool_calls"] for row in rows),
                           "mean_per_task": sum(row["tool_calls"] for row in rows) / len(rows) if rows else 0},
            "backend_calls": sum(row["backend_calls"] for row in rows),
            "http_requests": sum(row["http_requests"] for row in rows),
        }
    output_payload = {"protocol": manifest["protocol"], "backend": manifest["backend"],
                      "run_manifest_sha256": file_sha(manifest_path),
                      "predictions_sha256": manifest["predictions_sha256"],
                      "summary": summary, "per_task": scored}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/counterfactual-development-v1-001"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score_run(args.data, args.run, args.out)
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
