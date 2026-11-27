import json
import platform
from pathlib import Path
import shutil

import pytest

from tradingagents_lite.counterfactual_dev import generate_development
from tradingagents_lite.counterfactual_runner import _hash, _sealed, _verify_sealed, run
from tradingagents_lite.counterfactual_score import _correct, parse_answer, score_run
from tradingagents_lite.data import sha256 as file_sha


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data/counterfactual-development-v1-001"


@pytest.mark.parametrize(("answer", "unit", "expected"), [
    ("12.34 percent", "percent", "12.34"),
    ("12.34%", "percent", "12.34"),
    ("$12.34 million", "USD millions", "12.34"),
    ("$1.25 billion", "USD millions", "1250.00"),
    ("$12,340,000", "USD millions", "12.34"),
    ("($2,500,000)", "USD millions", "-2.50"),
])
def test_counterfactual_deterministic_numeric_units(answer, unit, expected):
    assert str(parse_answer(answer, unit)) == expected


def test_counterfactual_numeric_rejects_ambiguous_or_wrong_dimension():
    assert parse_answer("12.34 or 13.00", "USD millions") is None
    assert parse_answer("$12.34 million", "percent") is None
    assert parse_answer("NaN", "USD millions") is None


def test_counterfactual_answer_and_abstention_scoring_are_deterministic():
    answer_ref = {"status": "answered", "value": "12.34", "unit": "USD millions"}
    answered = {"status": "ok", "answer_status": "answered", "answer": "$12,340,000"}
    assert _correct(answered, answer_ref)
    abstain_ref = {"status": "abstained", "value": None, "unit": "USD millions"}
    abstained = {"status": "ok", "answer_status": "abstained", "answer": ""}
    assert _correct(abstained, abstain_ref)
    assert not _correct({**answered, "status": "invalid_output"}, answer_ref)


def test_checkpoint_hash_detects_mutation():
    sealed = _sealed({"task_id": "CF0:synthetic", "status": "ok"})
    assert _verify_sealed(sealed)["status"] == "ok"
    with pytest.raises(ValueError, match="integrity"):
        _verify_sealed({**sealed, "status": "changed"})


def test_frozen_scorer_reports_pairs_planning_abstention_and_future_acceptance(tmp_path):
    data_root = tmp_path / "data"
    generate_development(data_root)
    references = [json.loads(line) for line in (data_root / "scoring/reference.jsonl").read_text().splitlines()]
    references_by_id = {reference["task_id"]: reference for reference in references}
    plan = []
    records = []
    time_future = None
    for reference in references:
        packet = json.loads((data_root / "agent-inputs" / f"{reference['task_id']}.json").read_text())
        docs = {doc["version_id"]: doc for doc in packet["documents"]}
        evidence = ["cf:" + version for version in reference["required_versions"]]
        exposed = [{"evidence_id": "cf:" + doc["version_id"], "document_id": doc["document_id"],
                    "page": index} for index, doc in enumerate(packet["documents"], 1)]
        for arm in ("CF0", "CF1"):
            plan.append({"id": reference["task_id"], "arm": arm,
                         "input_sha256": file_sha(data_root / "agent-inputs" / f"{reference['task_id']}.json")})
            trace = []
            planning = reference.get("planning")
            if planning:
                trace.append({"query": packet["question"], "delivered_evidence": [
                    {"evidence_id": "cf:" + planning["routing_version"]}]})
                trace.append({"query": "approved " + planning["approved_reference"],
                              "delivered_evidence": [{"evidence_id": "cf:" + planning["target_version"]}]})
            future = []
            if reference["category"] == "time_conflict_or_missing":
                authoring = json.loads((data_root / "authoring" / f"{reference['task_id']}.json").read_text())
                cutoff = authoring["cutoff"]
                future = [doc for doc in authoring["documents"] if doc["eligible_at"] > cutoff]
            row = {"id": reference["task_id"], "task_id": f"{arm}:{reference['task_id']}", "arm": arm,
                   "status": "ok", "answer_status": reference["status"],
                   "answer": reference["value"] or "", "citations": evidence,
                   "valid_citations": evidence, "exposed": exposed, "retrieval_trace": trace,
                   "tool_calls": 2, "llm_calls": [], "budget_tokens": 0, "latency_seconds": 0,
                   "input_sha256": file_sha(data_root / "agent-inputs" / f"{reference['task_id']}.json")}
            if future and time_future is None and reference["status"] == "answered":
                time_future = (row, future[0], reference["future_only_values"][0])
            records.append(_sealed(row))

    run_root = tmp_path / "run"
    (run_root / "cells").mkdir(parents=True)
    predictions = run_root / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    manifest = {"backend": "mock", "status": "complete", "protocol": "counterfactual-phase2-v1",
                "plan": plan, "predictions_sha256": file_sha(predictions)}
    (run_root / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "scores.json"
    scored = score_run(data_root, run_root, output)
    assert scored["summary"]["CF0"]["answer_accuracy"] == {"correct": 10, "total": 10}
    assert scored["summary"]["CF1"]["counterfactual_pair_accuracy"]["correct_mothers"] == 5
    assert scored["summary"]["CF1"]["adaptive_planning_success_rate"] == {"success": 4, "total": 4}
    assert scored["summary"]["CF0"]["abstention_accuracy"] == {"correct": 2, "total": 2}
    assert scored["summary"]["CF0"]["future_evidence_acceptance_rate"]["observed_acceptances"] == 0

    bad, future_doc, future_value = time_future
    changed = dict(bad)
    changed.pop("record_sha256", None)
    changed["answer"] = future_value
    changed["citations"] = ["cf:" + future_doc["version_id"]]
    changed = _sealed(changed)
    changed_key = changed["task_id"]
    records = [changed if row["task_id"] == changed_key else row for row in records]
    predictions.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    manifest["predictions_sha256"] = file_sha(predictions)
    (run_root / "manifest.json").write_text(json.dumps(manifest))
    detected = score_run(data_root, run_root, output)
    assert detected["summary"]["CF0"]["future_evidence_acceptance_rate"]["observed_acceptances"] == 1

    # A route reference guessed before the routing observation is not adaptive planning.
    planning_row = next(row for row in records
                        if row["arm"] == "CF1" and references_by_id[row["id"]].get("planning"))
    planning_ref = references_by_id[planning_row["id"]]["planning"]
    planning_row.pop("record_sha256", None)
    planning_row["retrieval_trace"][0]["query"] += " " + planning_ref["approved_reference"]
    planning_row.update(_sealed(planning_row))
    predictions.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    manifest["predictions_sha256"] = file_sha(predictions)
    (run_root / "manifest.json").write_text(json.dumps(manifest))
    preguessed = score_run(data_root, run_root, output)
    assert preguessed["summary"]["CF1"]["adaptive_planning_success_rate"] == {"success": 3, "total": 4}


@pytest.mark.skipif(platform.system() != "Linux" or shutil.which("unshare") is None,
                    reason="requires Linux user/mount/network namespaces")
def test_actual_isolated_mock_runner_and_resume(tmp_path):
    output = tmp_path / "cf-run"
    manifest = run(DEV, output, backend="mock", limit=1)
    assert manifest["status"] == "complete"
    assert manifest["recorded_cells"] == 2
    lines = (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert {row["arm"] for row in rows} == {"CF0", "CF1"}
    for row in rows:
        sandbox = row["sandbox"]
        assert all(value == "denied" for value in sandbox["host_paths"].values())
        assert sandbox["host_paths"]["symlink_to_scoring_reference"] == "denied"
        assert sandbox["host_paths"]["relative_traversal_to_host_data"] == "denied"
        assert sandbox["network_isolated"] is True
        assert sandbox["network_devices"] == ["lo"]
        assert sandbox["fd_escape"] is False
        assert sandbox["api_key_in_environment"] is False
        assert row["model"] == "mock-counterfactual"
        assert row["budget_tokens"] == 0
        assert all(call["charged_tokens"] == 0 for call in row["llm_calls"])
        assert not any(key.casefold() in {"gold", "seed", "reference"} for key in row)
    cf1 = next(row for row in rows if row["arm"] == "CF1")
    cf0_cell = str(output / "cells" / f"{cf1['id']}-CF0.json")
    assert cf1["sandbox"]["host_paths"][cf0_cell] == "denied"
    before = (output / "predictions.jsonl").read_bytes()
    resumed = run(DEV, output, backend="mock", resume=True, limit=1)
    assert resumed["status"] == "complete"
    assert (output / "predictions.jsonl").read_bytes() == before
