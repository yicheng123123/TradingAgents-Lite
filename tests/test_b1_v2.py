import json
from pathlib import Path

import pytest

from tradingagents_lite.b1 import (B1_V2_MODEL, EXPECTED_TASKS_SHA256, _json_hash,
                                   _b1_protocol, _ledger_path_hash, _selected_task_bytes)
from tradingagents_lite.b1_v2 import (
    _assert_flash_rows, _cell_quality_issues, _model_config, _report, _validated_smoke, run_b1_v2,
)
from tradingagents_lite.data import sha256
from tradingagents_lite.provider import DEFAULT_MODEL
from tradingagents_lite.workflow import _llm_tool_result, _tool_content, _valid_tool_calls


def tool_calls(count):
    return [{"id": f"call-{index}", "function": {"name": "market_indicators", "arguments": "{}"}}
            for index in range(count)]


def test_general_model_default_is_flash_and_legacy_b1_stays_glm_53():
    assert DEFAULT_MODEL == B1_V2_MODEL
    assert _b1_protocol(42)["model_requested"] == "glm-5.3"


@pytest.mark.parametrize("count", [3, 4])
def test_flash_v2_accepts_three_or_four_calls_in_one_tool_round(count):
    assert _valid_tool_calls(tool_calls(count), allowed=True, maximum=4)


def test_flash_v2_rejects_more_than_four_calls_and_a_closed_round():
    assert not _valid_tool_calls(tool_calls(5), allowed=True, maximum=4)
    assert not _valid_tool_calls(tool_calls(1), allowed=False, maximum=4)


def test_flash_archive_identity_is_exact_and_mixed_models_cannot_merge():
    valid = [{"task_id": "dev-01", "status": "ok", "llm_calls": [
        {"status": "ok", "actual_model": B1_V2_MODEL}]}]
    assert _assert_flash_rows(valid) == [B1_V2_MODEL]
    mixed = valid + [{"task_id": "dev-02", "status": "ok", "llm_calls": [
        {"status": "ok", "actual_model": "glm-5.3"}]}]
    with pytest.raises(ValueError, match="different model"):
        _assert_flash_rows(mixed)


def test_flash_smoke_quality_checks_tools_structured_output_and_truncation():
    valid = {"status": "ok", "structured_output_trace": [{"valid": True}, {"valid": True}],
             "tool_trace": [{"name": "market_indicators", "dispatch": "react", "status": "ok"}],
             "citation_trace": [{"invalid_ids": []}], "usage": {"retries": 0},
             "llm_calls": [{"request_bytes": 5326, "tool_names": ["market_indicators"]}]}
    assert _cell_quality_issues(valid) == []
    truncated = {**valid, "llm_calls": [{"request_bytes": 8001, "tool_names": [],
                                          "error_code": "output_truncated"}]}
    assert set(_cell_quality_issues(truncated)) == {"prompt_byte_limit_exceeded", "output_truncated"}


def test_flash_report_contains_full_baseline_metrics_and_pair_transitions():
    summary = {
        "tasks": 2, "requested_tasks": 2, "completed_tasks": 2, "failed_tasks": 0,
        "task_completion_rate": 1.0, "structured_output_valid_count": 4,
        "structured_output_count": 4, "structured_output_valid_rate": 1.0,
        "tool_successes": 2, "tool_calls": 2, "tool_success_rate": 1.0,
        "non_hold_coverage": 1.0, "directional_hits": 1, "directional_n": 2,
        "directional_hit_rate": 0.5, "mean_alpha": 0.01,
        "alpha_date_bootstrap_95": [0.0, 0.02], "citation_references": 2,
        "valid_citation_references": 2, "evidence_citation_valid_rate": 1.0,
        "llm_calls": 4, "input_tokens": 100, "output_tokens": 20,
        "reasoning_tokens": 15, "total_tokens_charged": 120,
        "total_tokens_charged_per_task": 60, "completed_tasks_per_100k_tokens": 1666.7,
        "directional_hits_per_100k_tokens": 833.3, "latency_p50_seconds": 2.0,
        "latency_p95_seconds": 3.0, "llm_call_latency_p50_seconds": 0.5,
        "llm_call_latency_p95_seconds": 0.8, "api_cost_cny_estimate": 0.001,
        "failure_status_counts": {}, "llm_retries": 0,
    }
    report = _report(
        {"status": "completed", "actual_models_observed": [B1_V2_MODEL],
         "smoke_source": {"path": "smoke", "counted_again": False},
         "quality_check": {"complete": True},
         "budget_capacity": {"global_remaining": 10, "phase_remaining": 20}},
        summary, summary, summary,
        {"paired_n": 2, "mean_alpha_delta_e_minus_m": 0.01,
         "alpha_delta_date_bootstrap_95": [0.0, 0.02],
         "directional_transition_n": 2, "wrong_to_right": 1,
         "right_to_wrong": 0, "action_changed_n": 1},
    )
    for text in ("Structured valid", "Evidence refs valid / total", "P50/P95",
                 "wrong-to-right: 1", "right-to-wrong: 0", "reasoning / charged tokens"):
        assert text in report


def test_compact_tool_output_keeps_evidence_values_but_bounds_repeated_text():
    result = {"status": "ok", "source_type": "sec_filing", "items": [
        {"evidence_id": f"sec:{index}", "source_type": "sec_filing", "published_at": "2024-01-02",
         "eligible_at": "2024-01-02T18:00:00Z", "pit_level": "historical_proxy",
         "value": 123.45, "unit": "USD", "period": "FY2023", "period_end": "2023-12-31",
         "source_metadata": {"title": "10-K", "publisher": "SEC"}, "excerpt": "x" * 700,
         "content_hash": "must-not-be-sent"}
        for index in range(5)
    ]}
    compact = _llm_tool_result(result, compact=True)
    payload = json.loads(_tool_content(result, compact=True))
    assert len(compact["items"]) == 4
    assert compact["items"][0]["evidence_id"] == "sec:0"
    assert compact["items"][0]["value"] == 123.45
    assert compact["items"][0]["unit"] == "USD"
    assert len(compact["items"][0]["excerpt"]) == 320
    assert "content_hash" not in payload["items"][0]
    assert "sec:3" in _tool_content(result, compact=True)


def test_b1_v2_refuses_an_existing_output_directory_before_any_run(tmp_path):
    output = tmp_path / "already-used"
    output.mkdir()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        run_b1_v2("smoke", tmp_path / "missing-tasks", tmp_path / "missing-market",
                  tmp_path / "missing-evidence", tmp_path / "missing-b0", output)


def test_b1_v2_refuses_a_smoke_archive_with_missing_trajectory(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    tasks = repo / "tasks/development.jsonl"
    archive = tmp_path / "smoke"
    archive.mkdir()
    task_bytes = _selected_task_bytes(tasks, ["dev-01", "dev-02"])
    (archive / "tasks.jsonl").write_bytes(task_bytes)
    protocol = {}
    ledger_path = tmp_path / "ledger.sqlite"
    manifest = {
        "scope": "smoke", "status": "completed", "protocol": protocol,
        "protocol_sha256": _json_hash(protocol), "model_config": _model_config(),
        "selected_task_ids": ["dev-01", "dev-02"], "full_tasks_sha256": EXPECTED_TASKS_SHA256,
        "selected_tasks_sha256": sha256(archive / "tasks.jsonl"),
        "token_budget_ledger_path_sha256": _ledger_path_hash(ledger_path),
        "token_budget": {"experiment_id": "smoke-test"},
    }
    (archive / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory is missing"):
        _validated_smoke(archive, protocol, tasks, ledger_path)
