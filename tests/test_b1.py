from datetime import UTC, date, datetime
from pathlib import Path
import json
import csv
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tradingagents_lite import b1 as b1_module
from test_stage1 import TASK, bars
from tradingagents_lite.b1 import (
    EXPECTED_TASKS_SHA256, _combine_token_budget_sources, _completion_check,
    _json_hash, _ledger_path_hash, _merge_rows, _open_b1_budget, _tool_schema_hashes,
    _validate_b1_archive, _verify_ledger_records, _write_summary_csv, request_cost_ceiling,
)
from tradingagents_lite.data import HistoricalData, MarketView, read_tasks, sha256
from tradingagents_lite.evidence import Evidence, EvidenceStore
from tradingagents_lite.provider import ModelBackend, ModelCallError, _trader_context, client_from_env
from tradingagents_lite.prompts import prompt_hashes
from tradingagents_lite.runner import run, summary, _percentile
from tradingagents_lite.schema import Action, Budget, MarketReport, Proposal
from tradingagents_lite.token_budget import TokenBudget
from tradingagents_lite.workflow import MomentumBackend, _llm_tool_result, build_workflow, call_message


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fake_response(content="{}", *, model="glm-5.3", prompt_tokens=12, completion_tokens=5):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        id="resp-test", model=model, choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def test_core_prompt_hash_is_stable_and_contains_no_condition_specific_prompt():
    first = prompt_hashes()
    assert first == prompt_hashes()
    assert len(first["combined_core"]) == 64


def test_tool_and_trader_prompt_context_keeps_evidence_but_drops_redundant_fields():
    full = {
        "status": "available", "source_type": "news", "reason": None,
        "items": [{"evidence_id": "fin:exact-id", "source_type": "news",
                   "published_at": "2024-02-12T12:00:00Z", "eligible_at": "2024-02-12T12:01:00Z",
                   "pit_level": "historical_proxy", "content_hash": "a" * 64,
                   "source_url": "https://example.test/article", "source_metadata": {"title": "Title"},
                   "excerpt": "x" * 1000}],
    }
    prompt_result = _llm_tool_result(full)
    prompt_evidence = _trader_context({"fin:exact-id": full["items"][0]})["fin:exact-id"]
    assert prompt_result["items"][0]["evidence_id"] == "fin:exact-id"
    assert len(prompt_result["items"][0]["excerpt"]) == 800
    assert "content_hash" not in prompt_result["items"][0]
    assert prompt_evidence["evidence_id"] == "fin:exact-id"
    assert len(prompt_evidence["excerpt"]) == 800

    bar_result = _llm_tool_result({"evidence_id": "AAPL:day:bars:1", "bars": [{
        "day": "2024-02-13", "open": 1, "high": 2, "low": 0, "close": 1.5,
        "volume": 10, "available_at": "2024-02-13T21:15:00Z",
    }]})
    assert bar_result["bars"][0]["day"] == "2024-02-13"
    assert "available_at" not in bar_result["bars"][0]


def test_client_has_sdk_automatic_retries_disabled(monkeypatch):
    monkeypatch.setenv("TALITE_API_KEY", "offline-test-client-token")
    monkeypatch.delenv("TALITE_MODEL", raising=False)
    monkeypatch.delenv("TALITE_BASE_URL", raising=False)
    client, _, _ = client_from_env()
    try:
        assert client.max_retries == 0
    finally:
        client.close()


def test_model_backend_retries_timeout_once_and_counts_every_attempt(tmp_path):
    completions = FakeCompletions([TimeoutError("private transport detail"), fake_response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = ModelBackend(client, "glm-5.3", Budget(max_tokens=128), None,
                           token_budget=TokenBudget(tmp_path / "budget.sqlite"))
    result = backend._call([{"role": "user", "content": "test"}], [])
    assert result["content"] == "{}"
    assert backend.usage == {
        "calls": 2, "logical_calls": 1, "retries": 1, "failed_calls": 0,
        "prompt_tokens": 12, "completion_tokens": 5, "reasoning_tokens": 0,
        "separate_reasoning_tokens": 0, "cached_tokens": 0,
        "total_tokens_charged": sum(row["charged_tokens"] for row in backend.token_budget.attempts()),
        "missing_usage_calls": 0, "reasoning_detail_missing_calls": 1,
        "cached_detail_missing_calls": 1,
        "budget_denials": 0,
    }
    assert [call["status"] for call in backend.call_log] == ["retrying", "ok"]
    assert backend.call_log[0]["error_code"] == "timeout"
    assert "private transport detail" not in json.dumps(backend.call_log)
    assert backend.token_budget.attempts()[0]["status"] == "failed"
    assert backend.token_budget.attempts()[0]["usage_missing"] == 1
    assert completions.requests[0]["extra_body"] == {
        "thinking": {"type": "enabled"}, "reasoning_effort": "low"}
    assert "thinking" not in completions.requests[0]
    assert "reasoning_effort" not in completions.requests[0]


def test_model_backend_stops_at_retry_budget_and_redacts_timeout(tmp_path):
    completions = FakeCompletions([TimeoutError("sensitive"), TimeoutError("sensitive again")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = ModelBackend(client, "glm-5.3", Budget(), None, max_retries=1,
                           token_budget=TokenBudget(tmp_path / "budget.sqlite"))
    with pytest.raises(ModelCallError) as error:
        backend._call([{"role": "user", "content": "test"}], [])
    assert (error.value.code, error.value.attempts) == ("timeout", 2)
    assert len(completions.requests) == 2
    assert backend.usage["calls"] == 2
    assert backend.usage["retries"] == 1
    assert backend.usage["failed_calls"] == 1
    assert all("sensitive" not in json.dumps(row) for row in backend.call_log)


def test_request_byte_cap_blocks_before_network_call(tmp_path):
    completions = FakeCompletions([fake_response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = ModelBackend(client, "glm-5.3", Budget(), None, max_prompt_bytes=10,
                           token_budget=TokenBudget(tmp_path / "budget.sqlite"))
    with pytest.raises(ModelCallError, match="prompt_byte_limit"):
        backend._call([{"role": "user", "content": "larger than ten bytes"}], [])
    assert completions.requests == []
    assert backend.usage["calls"] == 0


def test_invalid_structured_action_is_rejected(tmp_path):
    completions = FakeCompletions([fake_response(
        '{"action":"short","rationale":"invalid action","evidence_ids":[]}')])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = ModelBackend(client, "glm-5.3", Budget(), None,
                           token_budget=TokenBudget(tmp_path / "budget.sqlite"))
    report = MarketReport(view="observed market data", evidence_ids=["evidence-1"])
    with pytest.raises(ValidationError):
        backend.trader(report, {"evidence-1": {"close": 10}})


def test_unknown_citation_fails_closed_and_is_audited():
    class InvalidCitation(MomentumBackend):
        def analyst(self, messages, tools):
            return {"content": json.dumps({"view": "claim", "evidence_ids": ["not-observed"]})}

    budget = Budget()
    state = build_workflow(MarketView(TASK, bars()), InvalidCitation(budget), budget).invoke({})
    assert state["decision"]["risk_veto"] == "workflow_failure"
    assert state["errors"][0] == {"stage": "analyst", "code": "invalid_citation"}
    assert state["citation_trace"][0]["invalid_ids"] == ["not-observed"]


def test_tool_timeout_is_visible_and_a_later_tool_call_recovers():
    class TimeoutView(MarketView):
        calls = 0

        def indicators(self):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("local synthetic timeout")
            return super().indicators()

    class RecoveryBackend:
        usage = None
        call_log = []

        def analyst(self, messages, tools):
            tool_messages = [item for item in messages if item["role"] == "tool"]
            if not tool_messages:
                return call_message("market_indicators")
            latest = json.loads(tool_messages[-1]["content"])
            if latest.get("error") == "tool_timeout":
                return {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "next-call", "type": "function",
                    "function": {"name": "market_bars", "arguments": '{"limit":5}'},
                }]}
            report = MarketReport(view="bar data observed", evidence_ids=[latest["evidence_id"]])
            return {"content": report.model_dump_json()}

        def trader(self, report, evidence):
            return Proposal(action=Action.HOLD, rationale="wait", evidence_ids=report.evidence_ids)

    budget = Budget(max_tool_rounds=2)
    state = build_workflow(TimeoutView(TASK, bars()), RecoveryBackend(), budget).invoke({})
    assert [(event["status"], event.get("code")) for event in state["tool_trace"]] == [
        ("error", "tool_timeout"), ("ok", None)
    ]
    assert state["decision"]["action"] == "hold"
    assert state["errors"] == []


def test_b1_conditions_share_the_same_core_prompt_and_only_add_financial_tool():
    class Scripted:
        usage = None
        call_log = []

        def __init__(self):
            self.system_prompts = []
            self.tool_names = []

        def analyst(self, messages, tools):
            self.system_prompts.append(messages[0]["content"])
            self.tool_names = [item["function"]["name"] for item in tools]
            tool_messages = [item for item in messages if item["role"] == "tool"]
            if not tool_messages:
                return call_message("market_indicators")
            observed = json.loads(tool_messages[-1]["content"])
            report = MarketReport(view="same market evidence", evidence_ids=[observed["evidence_id"]])
            return {"content": report.model_dump_json()}

        def trader(self, report, evidence):
            return Proposal(action=Action.HOLD, rationale="wait", evidence_ids=report.evidence_ids)

    budget = Budget()
    m_backend, e_backend = Scripted(), Scripted()
    m_state = build_workflow(MarketView(TASK, bars()), m_backend, budget).invoke({})
    e_state = build_workflow(MarketView(TASK, bars()), e_backend, budget,
                             financial=EvidenceStore([]).view(TASK)).invoke({})
    assert m_backend.system_prompts == e_backend.system_prompts
    assert m_backend.tool_names == ["market_indicators", "market_bars"]
    assert e_backend.tool_names == ["market_indicators", "market_bars", "financial_evidence"]
    assert m_state["decision"]["action"] == e_state["decision"]["action"] == "hold"


def test_b1_tool_views_never_expose_future_market_or_financial_evidence():
    view = MarketView(TASK, bars(bump=500))
    assert all(item["day"] <= TASK.decision_date.isoformat() for item in view.get_bars(20)["bars"])
    future = Evidence(
        ticker=TASK.symbol, source_type="news", document_id="future-release",
        source_url="https://example.test/future",
        published_at=datetime(2025, 1, 2, tzinfo=UTC),
        ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
        eligible_at=datetime(2025, 1, 3, tzinfo=UTC),
        availability_basis="synthetic future fixture",
        content_hash="a" * 64, excerpt="future text", pit_level="historical_proxy",
    )
    result = EvidenceStore([future]).view(TASK).search("news")
    assert result["status"] == "unavailable"
    assert result["items"] == []


def test_model_timeout_becomes_failed_task_not_scored_hold(tmp_path):
    tasks_path = tmp_path / "one-task.jsonl"
    tasks_path.write_text(
        '{"task_id":"dev-01","symbol":"AAPL","decision_date":"2024-02-13","horizon_sessions":3}\n',
        encoding="utf-8",
    )

    class FailedModel:
        usage = {"calls": 2, "logical_calls": 1, "retries": 1, "failed_calls": 1,
                 "prompt_tokens": 0, "completion_tokens": 0, "missing_usage_calls": 0}
        call_log = [{"status": "error", "error_code": "timeout", "attempt": 2}]

        def analyst(self, messages, tools):
            raise ModelCallError("timeout", 2)

        def trader(self, report, evidence):
            raise AssertionError("Trader must not run after analyst timeout")

    result = run(
        tasks_path, Path("data/yahoo-2024-v1"), tmp_path / "failed-run",
        FailedModel, {"backend": "llm", "model": "glm-5.3"}, Budget(), 42, True,
    )
    task = json.loads((tmp_path / "failed-run/tasks.jsonl").read_text(encoding="utf-8"))
    assert task["status"] == "model_error"
    assert task["decision"]["action"] == "hold"
    assert task["alpha"] is None
    assert result["completed_tasks"] == 0
    assert result["failed_tasks"] == 1
    assert result["task_completion_rate"] == 0
    assert result["failure_status_counts"] == {"model_error": 1}


def test_cost_ceiling_covers_fixed_smoke_and_development_protocols():
    smoke = request_cost_ceiling(2)
    development = request_cost_ceiling(20)
    assert smoke["max_http_attempts_including_retries"] == 16
    assert development["max_http_attempts_including_retries"] == 160
    assert smoke["max_conservative_tokens"] == 98_432
    assert development["max_conservative_tokens"] == 984_320
    assert smoke["max_conservative_tokens"] < 120_000
    assert smoke["list_price_cost_ceiling_cny"] == pytest.approx(1.115136)
    assert development["list_price_cost_ceiling_cny"] == pytest.approx(11.15136)


def test_b1_top_level_completion_requires_all_tasks_failures_and_pairs():
    complete = {"tasks": 2, "completed_tasks": 2, "failed_tasks": 0}
    assert _completion_check(2, complete, complete, 2)["complete"] is True
    assert _completion_check(2, {**complete, "tasks": 1}, complete, 1)["complete"] is False
    assert _completion_check(2, {**complete, "failed_tasks": 1}, complete, 1)["complete"] is False
    assert _completion_check(2, complete, complete, 1)["complete"] is False


def test_resume_merge_rejects_duplicate_and_missing_task_instances():
    left = {"task_id": "dev-01", "status": "ok", "alpha": 0.1}
    right = {"task_id": "dev-02", "status": "model_error", "decision": {"action": "hold"}, "alpha": None}
    assert _merge_rows([left], [right], ["dev-01", "dev-02"]) == [left, right]
    with pytest.raises(ValueError, match="duplicate"):
        _merge_rows([left], [dict(left)], ["dev-01"])
    with pytest.raises(ValueError, match="missing or unexpected"):
        _merge_rows([left], [], ["dev-01", "dev-02"])


def test_resume_archive_rejects_compatibility_mismatch_before_reading_trajectories(tmp_path):
    root = tmp_path / "smoke"
    root.mkdir()
    compatibility = {
        "version": "b1-smoke-reuse-v1", "code_git_sha": "a" * 40,
        "protocol": {"temperature": 0.0, "statistics_seed": 42},
        "prompt_hashes": {"combined_core": "b" * 64},
        "tool_schema_hashes": {"B1-M": "c" * 64, "B1-E": "d" * 64},
    }
    manifest = {
        "scope": "smoke", "selected_task_ids": ["dev-01", "dev-02"],
        "full_tasks_sha256": EXPECTED_TASKS_SHA256,
        "protocol": compatibility["protocol"], "prompt_hashes": compatibility["prompt_hashes"],
        "tool_schema_hashes": compatibility["tool_schema_hashes"],
        "code_git_sha": compatibility["code_git_sha"],
        "resume_compatibility": compatibility,
        "resume_compatibility_sha256": _json_hash(compatibility),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    changed = {**compatibility, "protocol": {**compatibility["protocol"], "temperature": 0.2}}
    with pytest.raises(ValueError, match="does not match"):
        _validate_b1_archive(root, expected_scope="smoke", expected_ids=["dev-01", "dev-02"],
                             expected_compatibility=changed)


def test_resume_archive_rejects_missing_original_trajectory(tmp_path):
    root = tmp_path / "smoke"
    root.mkdir()
    task_bytes = (
        '{"task_id":"dev-01","symbol":"AAPL","decision_date":"2024-02-13","horizon_sessions":3}\n'
        '{"task_id":"dev-02","symbol":"MSFT","decision_date":"2024-02-13","horizon_sessions":3}\n'
    ).encode()
    (root / "tasks.jsonl").write_bytes(task_bytes)
    compatibility = {
        "version": "b1-smoke-reuse-v1", "code_git_sha": "a" * 40,
        "protocol": {"statistics_seed": 42}, "prompt_hashes": {"combined_core": "b" * 64},
        "tool_schema_hashes": {"B1-M": "c" * 64, "B1-E": "d" * 64},
    }
    manifest = {
        "scope": "smoke", "selected_task_ids": ["dev-01", "dev-02"],
        "full_tasks_sha256": EXPECTED_TASKS_SHA256, "selected_tasks_sha256": sha256(root / "tasks.jsonl"),
        "protocol": compatibility["protocol"], "prompt_hashes": compatibility["prompt_hashes"],
        "tool_schema_hashes": compatibility["tool_schema_hashes"], "code_git_sha": compatibility["code_git_sha"],
        "resume_compatibility": compatibility, "resume_compatibility_sha256": _json_hash(compatibility),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing run metadata or trajectory"):
        _validate_b1_archive(root, expected_scope="smoke", expected_ids=["dev-01", "dev-02"])


def test_resume_archive_accepts_complete_mock_trajectories_and_matching_ledger(tmp_path, monkeypatch):
    root = tmp_path / "smoke"
    root.mkdir()
    task_path = Path("tasks/development.jsonl")
    task_bytes = b1_module._selected_task_bytes(task_path, ["dev-01", "dev-02"])
    (root / "tasks.jsonl").write_bytes(task_bytes)
    task_hash = sha256(root / "tasks.jsonl")
    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("TALITE_TOKEN_BUDGET_DB", str(ledger_path))
    ledger = TokenBudget(ledger_path, global_limit=100_000, phase_limits={"B1": 100_000})
    experiment_id = "mock-smoke-resume"
    protocol = b1_module._b1_protocol(42)
    prompts = prompt_hashes()
    schemas = _tool_schema_hashes()
    compatibility = {
        "version": b1_module.RESUME_PROTOCOL_VERSION, "code_git_sha": "e" * 40,
        "tracked_worktree_clean": True, "protocol": protocol,
        "lock_sha256": sha256(Path("uv.lock")),
        "dependencies": {name: b1_module.importlib.metadata.version(name)
                         for name in ("langgraph", "pydantic", "openai")},
        "budget": Budget(max_tool_rounds=2, max_tokens=512).model_dump(), "react": True,
        "full_tasks_sha256": EXPECTED_TASKS_SHA256,
        "market_manifest_sha256": b1_module.EXPECTED_MARKET_MANIFEST_SHA256,
        "evidence_manifest_sha256": b1_module.EXPECTED_EVIDENCE_MANIFEST_SHA256,
        "evidence_sha256": b1_module.EXPECTED_EVIDENCE_SHA256,
        "b0_run_metadata_sha256": sha256(Path("results/b0-development-001/run.json")),
        "b0_trajectory_sha256": sha256(Path("results/b0-development-001/trajectory.jsonl")),
        "prompt_hashes": prompts, "tool_schema_hashes": schemas,
        "resume_protocol_document_sha256": sha256(b1_module.RESUME_PROTOCOL_DOCUMENT),
    }
    compatibility_hash = _json_hash(compatibility)
    config = {
        "backend": "llm", "model": "glm-5.3", "api_host": "open.bigmodel.cn",
        "temperature": 0.0, "max_tokens": 512, "thinking": "enabled", "reasoning_effort": "low",
        "provider_seed": None, "phase": "B1", "sdk_retries": 0, "application_retries": 0,
        "max_prompt_bytes": 5000, "pricing_cny_per_million": b1_module.PRICE_CNY_PER_MILLION,
        "pricing_source": b1_module.PRICE_SOURCE, "pricing_as_of": b1_module.PRICE_AS_OF,
    }
    protocol_metadata = {
        "prompt_hashes": prompts, "max_model_retries": 0, "max_prompt_bytes": 5000,
        "model_timeout_seconds": 25, "provider_generation_seed": None, "holdout_accessed": False,
        "tool_schema_hashes": schemas, "b1_compatibility_sha256": compatibility_hash,
        "resume_protocol_document_sha256": compatibility["resume_protocol_document_sha256"],
    }
    variant_rows = {}
    for variant in ("B1-M", "B1-E"):
        run_dir = root / variant
        run_dir.mkdir()
        rows = []
        for index, task_id in enumerate(("dev-01", "dev-02"), start=1):
            request_hash = f"{(1 if variant == 'B1-M' else 2) * 100 + index:064x}"
            attempt_id, reserved = ledger.reserve(
                experiment_id=experiment_id, task_id=task_id, phase="B1", model="glm-5.3",
                logical_call=1, attempt_number=1, request_sha256=request_hash, request_bytes=64,
                max_output_tokens=512, protocol_allowance=128, separate_reasoning_tokens=512,
            )
            charged = ledger.settle_usage(attempt_id, input_tokens=10, output_tokens=8,
                                          reasoning_tokens=3, cached_tokens=4)
            rows.append({
                "task_id": task_id, "symbol": "AAPL" if task_id == "dev-01" else "MSFT",
                "decision_date": "2024-02-13", "status": "ok", "alpha": 0.01 * index,
                "direction": 1, "directional_hit": True, "decision": {"action": "buy"},
                "latency_seconds": 0.2, "tool_trace": [], "structured_output_trace": [],
                "citation_trace": [], "llm_calls": [{
                    "logical_call": 1, "attempt": 1, "status": "ok", "actual_model": "glm-5.3",
                    "request_sha256": request_hash, "prompt_sha256": "a" * 64,
                    "tools_sha256": "b" * 64, "request_bytes": 64, "reserved_tokens": reserved,
                    "charged_tokens": charged, "prompt_tokens": 10, "completion_tokens": 8,
                    "reasoning_tokens": 3, "cached_tokens": 4, "latency_seconds": 0.1,
                }],
                "usage": {"calls": 1, "logical_calls": 1, "retries": 0, "failed_calls": 0,
                          "prompt_tokens": 10, "completion_tokens": 8, "reasoning_tokens": 3,
                          "separate_reasoning_tokens": 0, "cached_tokens": 4,
                          "total_tokens_charged": charged, "missing_usage_calls": 0,
                          "reasoning_detail_missing_calls": 0, "cached_detail_missing_calls": 0,
                          "budget_denials": 0},
            })
        variant_rows[variant] = rows
        trajectory = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        (run_dir / "tasks.jsonl").write_bytes(trajectory)
        (run_dir / "trajectory.jsonl").write_bytes(trajectory)
        run_meta = {
            "status": "completed", "ablation": variant, "tasks_sha256": task_hash,
            "requested_tasks": 2, "processed_tasks": 2, "config": config,
            "protocol_metadata": protocol_metadata,
            "snapshot_sha256": b1_module.EXPECTED_MARKET_MANIFEST_SHA256,
            "execution": "next shared session open to horizon-th shared session close, OHLC price return",
            "alpha": "position direction × (asset price return − SPY price return); no costs",
            "budget": compatibility["budget"], "react": True,
            "lock_sha256": compatibility["lock_sha256"],
            "dependencies": compatibility["dependencies"], "token_budget_phase": "B1",
            "actual_models_observed": ["glm-5.3"],
        }
        if variant == "B1-E":
            run_meta["evidence_manifest_sha256"] = b1_module.EXPECTED_EVIDENCE_MANIFEST_SHA256
            run_meta["evidence_sha256"] = b1_module.EXPECTED_EVIDENCE_SHA256
        (run_dir / "run.json").write_text(json.dumps(run_meta), encoding="utf-8")
    budget_snapshot = ledger.snapshot(experiment_id=experiment_id, phase="B1")
    manifest = {
        "scope": "smoke", "status": "completed", "selected_task_ids": ["dev-01", "dev-02"],
        "full_tasks_sha256": EXPECTED_TASKS_SHA256, "selected_tasks_sha256": task_hash,
        "market_manifest_sha256": b1_module.EXPECTED_MARKET_MANIFEST_SHA256,
        "evidence_manifest_sha256": b1_module.EXPECTED_EVIDENCE_MANIFEST_SHA256,
        "evidence_sha256": b1_module.EXPECTED_EVIDENCE_SHA256,
        "protocol": protocol, "prompt_hashes": prompts, "tool_schema_hashes": schemas,
        "code_git_sha": compatibility["code_git_sha"],
        "resume_compatibility": compatibility,
        "resume_compatibility_sha256": compatibility_hash,
        "token_budget_ledger_path_sha256": _ledger_path_hash(ledger_path),
        "token_budget": budget_snapshot,
        "trajectory_sha256": {v: sha256(root / v / "trajectory.jsonl") for v in ("B1-M", "B1-E")},
        "actual_models_observed": ["glm-5.3"], "completion_check": {"complete": True},
        "b1_m_summary": summary(variant_rows["B1-M"], 42, config),
        "b1_e_summary": summary(variant_rows["B1-E"], 42, config),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    validated = _validate_b1_archive(
        root, expected_scope="smoke", expected_ids=["dev-01", "dev-02"],
        expected_compatibility=compatibility, require_success=True, frozen_tasks_path=task_path,
    )
    assert validated["runs"]["token_budget"]["attempt_count"] == 4
    assert validated["runs"]["token_budget"]["conservatively_charged_tokens"] == 72


def test_resume_budget_records_are_verified_and_combined_once(tmp_path):
    ledger_path = tmp_path / "token-budget.sqlite"
    ledger = TokenBudget(ledger_path, global_limit=1000, phase_limits={"B1": 1000})
    experiment_id = "smoke-004"
    attempt_id, reservation = ledger.reserve(
        experiment_id=experiment_id, task_id="dev-01", phase="B1", model="glm-5.3",
        logical_call=1, attempt_number=1, request_sha256="a" * 64, request_bytes=10,
        max_output_tokens=10, protocol_allowance=2, separate_reasoning_tokens=10,
    )
    charged = ledger.settle_usage(attempt_id, input_tokens=10, output_tokens=8,
                                  reasoning_tokens=3, cached_tokens=4)
    call = {
        "logical_call": 1, "attempt": 1, "status": "ok", "request_sha256": "a" * 64,
        "prompt_sha256": "b" * 64, "tools_sha256": "c" * 64, "request_bytes": 10,
        "reserved_tokens": reservation, "charged_tokens": charged,
        "prompt_tokens": 10, "completion_tokens": 8, "reasoning_tokens": 3, "cached_tokens": 4,
        "actual_model": "glm-5.3",
    }
    # B1-M and B1-E can issue the same logical-call ordinal for a shared task.
    attempt_id_2, reservation_2 = ledger.reserve(
        experiment_id=experiment_id, task_id="dev-01", phase="B1", model="glm-5.3",
        logical_call=1, attempt_number=1, request_sha256="a" * 64, request_bytes=10,
        max_output_tokens=10, protocol_allowance=2, separate_reasoning_tokens=10,
    )
    charged_2 = ledger.settle_usage(attempt_id_2, input_tokens=10, output_tokens=8,
                                    reasoning_tokens=3, cached_tokens=4)
    assert reservation_2 == reservation and charged_2 == charged
    rows = [{"task_id": "dev-01", "llm_calls": [call]}, {"task_id": "dev-01", "llm_calls": [call]}]
    manifest = {
        "token_budget_ledger_path_sha256": _ledger_path_hash(ledger_path),
        "token_budget": {
            "experiment_id": experiment_id, "attempts_recorded": 2, "http_attempts": 2,
            "conservatively_charged_tokens": charged * 2, "experiment_used": charged * 2,
        },
    }
    audit = _verify_ledger_records(rows, ledger.attempts(experiment_id=experiment_id),
                                   manifest=manifest, ledger_path=ledger_path)
    assert audit["attempt_count"] == 2
    assert audit["conservatively_charged_tokens"] == charged * 2 == 36
    merged = _combine_token_budget_sources(
        {"experiment_id": "smoke", "attempt_count": 1, "http_attempt_count": 1,
         "conservatively_charged_tokens": 18, "attempt_ids": [1]},
        {"experiment_id": "continuation", "attempt_count": 2, "http_attempt_count": 2,
         "conservatively_charged_tokens": 22, "attempt_ids": [2, 3]},
    )
    assert merged["attempt_count"] == 3
    assert merged["conservatively_charged_tokens"] == 40
    assert merged["counted_once"] is True
    with pytest.raises(ValueError, match="overlap"):
        _combine_token_budget_sources(
            {"experiment_id": "smoke", "attempt_ids": [4]},
            {"experiment_id": "continuation", "attempt_ids": [4]},
        )


def test_real_b1_fails_closed_without_existing_budget_ledger(tmp_path):
    missing = tmp_path / "must-not-be-created.sqlite"
    assert _open_b1_budget(missing, "smoke", api_key_available=False) is None
    assert not missing.exists()
    with pytest.raises(ValueError, match="ledger is missing"):
        _open_b1_budget(missing, "smoke", api_key_available=True)
    assert not missing.exists()

    empty = tmp_path / "empty.sqlite"
    empty.touch()
    with pytest.raises(ValueError, match="no complete attempt schema"):
        _open_b1_budget(empty, "smoke", api_key_available=True)
    assert empty.stat().st_size == 0

    incomplete = tmp_path / "incomplete.sqlite"
    connection = sqlite3.connect(incomplete)
    connection.execute("CREATE TABLE token_attempts (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="no complete attempt schema"):
        _open_b1_budget(incomplete, "smoke", api_key_available=True)
    connection = sqlite3.connect(incomplete)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(token_attempts)")}
    connection.close()
    assert columns == {"id"}


def test_merged_summary_is_equivalent_to_one_shot_and_keeps_failure_rows():
    def row(task_id, alpha, status="ok"):
        return {
            "task_id": task_id, "symbol": "AAPL", "decision_date": "2024-02-13",
            "status": status, "alpha": alpha, "direction": 1 if status == "ok" else 0,
            "directional_hit": status == "ok", "latency_seconds": 0.2,
            "llm_calls": [], "tool_trace": [], "usage": None,
        }
    smoke_rows = [row("dev-01", 0.01)]
    continuation_rows = [row("dev-02", None, "model_error")]
    merged = _merge_rows(smoke_rows, continuation_rows, ["dev-01", "dev-02"])
    assert merged == [*smoke_rows, *continuation_rows]
    assert summary(merged, 42) == summary([*smoke_rows, *continuation_rows], 42)
    assert merged[1]["status"] == "model_error"
    assert merged[1]["alpha"] is None


def test_b1_merge_writes_full_archive_once_from_smoke_and_remaining_tasks(tmp_path, monkeypatch):
    task_path = Path("tasks/development.jsonl")
    tasks = read_tasks(task_path)
    task_ids = [task.task_id for task in tasks]
    compatibility = {
        "protocol": {"statistics_seed": 42},
        "resume_protocol_document_sha256": sha256(b1_module.RESUME_PROTOCOL_DOCUMENT),
        "b0_run_metadata_sha256": sha256(Path("results/b0-development-001/run.json")),
        "b0_trajectory_sha256": sha256(Path("results/b0-development-001/trajectory.jsonl")),
    }
    config = {
        "backend": "llm", "model": "glm-5.3", "pricing_cny_per_million": {"input": 8.0, "output": 28.0},
    }

    def make_rows(variant, selected):
        rows = []
        for index, task in enumerate(selected):
            m_hit = index % 2 == 1
            e_hit = index % 2 == 0
            hit = (m_hit if variant == "B1-M" else e_hit)
            direction = 1 if hit else -1
            rows.append({
                "task_id": task.task_id, "symbol": task.symbol,
                "decision_date": task.decision_date.isoformat(), "status": "ok",
                "alpha": 0.001 * (index + 1), "direction": direction,
                "directional_hit": hit, "decision": {"action": "buy" if direction > 0 else "sell"},
                "latency_seconds": 0.1 + index / 100, "tool_trace": [],
                "structured_output_trace": [{"valid": True}, {"valid": True}],
                "citation_trace": [], "llm_calls": [{"actual_model": "glm-5.3", "latency_seconds": 0.05}],
                "usage": {"calls": 1, "logical_calls": 1, "retries": 0, "failed_calls": 0,
                          "prompt_tokens": 10, "completion_tokens": 8, "reasoning_tokens": 3,
                          "separate_reasoning_tokens": 0, "cached_tokens": 0,
                          "total_tokens_charged": 18, "missing_usage_calls": 0,
                          "reasoning_detail_missing_calls": 0, "cached_detail_missing_calls": 0,
                          "budget_denials": 0},
            })
        return rows

    smoke_rows = {variant: make_rows(variant, tasks[:2]) for variant in ("B1-M", "B1-E")}
    remaining_rows = {variant: make_rows(variant, tasks[2:]) for variant in ("B1-M", "B1-E")}
    smoke_path, continuation_path = tmp_path / "smoke", tmp_path / "continuation"
    for base, rows_by_variant in ((smoke_path, smoke_rows), (continuation_path, remaining_rows)):
        base.mkdir()
        (base / "manifest.json").write_text("{}\n", encoding="utf-8")
        for variant in ("B1-M", "B1-E"):
            (base / variant).mkdir()
            (base / variant / "trajectory.jsonl").write_text("mock trajectory\n", encoding="utf-8")
    smoke_manifest_sha = sha256(smoke_path / "manifest.json")
    smoke_budget = {
        "experiment_id": "smoke-exp", "attempt_count": 4, "http_attempt_count": 4,
        "failed_attempt_count": 0, "conservatively_charged_tokens": 72,
        "attempt_ids": [1, 2, 3, 4],
    }
    continuation_budget = {
        "experiment_id": "continuation-exp", "attempt_count": 36, "http_attempt_count": 36,
        "failed_attempt_count": 0, "conservatively_charged_tokens": 648,
        "attempt_ids": list(range(5, 41)),
    }
    smoke_archive = {
        "manifest": {
            "resume_compatibility_sha256": "f" * 64,
            "protocol": compatibility["protocol"], "prompt_hashes": {}, "tool_schema_hashes": {},
            "scope": "smoke",
        },
        "compatibility": compatibility,
        "runs": {
            **{variant: {"run": {"config": config}, "rows": smoke_rows[variant],
                         "actual_models": ["glm-5.3"]} for variant in ("B1-M", "B1-E")},
            "token_budget": smoke_budget,
        },
    }
    continuation_archive = {
        "manifest": {"resume_source": {
            "manifest_sha256": smoke_manifest_sha, "source_experiment_id": "smoke-exp",
            "reused_task_ids": ["dev-01", "dev-02"], "reused_task_instances": 4,
            "reused_http_attempts": 4, "reused_charged_tokens": 72,
            "source_attempt_ids": [1, 2, 3, 4], "counted_again": False,
        }},
        "compatibility": compatibility,
        "runs": {
            **{variant: {"run": {"config": config}, "rows": remaining_rows[variant],
                         "actual_models": ["glm-5.3"]} for variant in ("B1-M", "B1-E")},
            "token_budget": continuation_budget,
        },
    }
    monkeypatch.setattr(
        b1_module, "_validate_b1_archive",
        lambda path, **kwargs: smoke_archive if kwargs["expected_scope"] == "smoke" else continuation_archive,
    )
    monkeypatch.setattr(b1_module, "_ledger_totals", lambda path: {
        "global_limit": 15_000_000, "global_used": 1000, "global_remaining": 14_999_000,
        "b1_limit": 1_000_000, "b1_used": 720, "b1_remaining": 999_280,
    })
    output = tmp_path / "merged"
    result = b1_module.merge_b1(
        smoke_path, continuation_path, task_path, Path("data/yahoo-2024-v1"),
        Path("data/financial-evidence-dev-003"), Path("results/b0-development-001"), output,
    )
    merged_manifest = json.loads((output / "manifest.json").read_text())
    paired = json.loads((output / "paired_summary.json").read_text())
    assert result["status"] == "completed"
    assert merged_manifest["completion_check"]["valid_pairs"] == 20
    assert merged_manifest["token_budget"]["merged_source_charged_tokens_counted_once"] == 720
    assert merged_manifest["token_budget"]["source_attempt_ids_disjoint"] is True
    assert paired["wrong_to_right"] > 0
    assert paired["right_to_wrong"] > 0
    assert len((output / "B1-M" / "trajectory.jsonl").read_text().splitlines()) == 20
    assert len((output / "B1-E" / "trajectory.jsonl").read_text().splitlines()) == 20
    assert len((output / "paired.csv").read_text().splitlines()) == 21
    assert sha256(smoke_path / "manifest.json") == smoke_manifest_sha


def test_latency_percentiles_are_deterministic_linear_interpolations():
    assert _percentile([], 0.95) is None
    assert _percentile([0.1, 0.2, 0.3], 0.50) == pytest.approx(0.2)
    assert _percentile([0.1, 0.2, 0.3], 0.95) == pytest.approx(0.29)


def test_b1_tool_schema_hashes_match_runtime_function_schemas():
    from tradingagents_lite.provider import _canonical_hash
    from tradingagents_lite.workflow import specs
    from tradingagents_lite.tool_registry import FINANCIAL_SPEC, SPECS
    hashes = _tool_schema_hashes()
    assert hashes["B1-M"] == _canonical_hash(specs(SPECS))
    assert hashes["B1-E"] == _canonical_hash(specs({**SPECS, "financial_evidence": FINANCIAL_SPEC}))


def test_runner_archives_and_stops_after_budget_exhaustion(tmp_path):
    tasks_path = tmp_path / "two-tasks.jsonl"
    tasks_path.write_text(
        '{"task_id":"dev-01","symbol":"AAPL","decision_date":"2024-02-13","horizon_sessions":3}\n'
        '{"task_id":"dev-02","symbol":"MSFT","decision_date":"2024-02-13","horizon_sessions":3}\n',
        encoding="utf-8",
    )
    token_budget = TokenBudget(tmp_path / "budget.sqlite", global_limit=1,
                               phase_limits={"B1": 1})
    completions = FakeCompletions([])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def factory():
        return ModelBackend(client, "glm-5.3", Budget(), None, max_retries=0,
                            token_budget=token_budget)

    result = run(
        tasks_path, Path("data/yahoo-2024-v1"), tmp_path / "partial-run", factory,
        {"backend": "llm", "model": "glm-5.3", "phase": "B1"}, Budget(), 42, True,
        token_budget=token_budget, experiment_id="budget-stop", phase="B1",
    )
    saved = [json.loads(line) for line in (tmp_path / "partial-run/tasks.jsonl").read_text().splitlines()]
    run_metadata = json.loads((tmp_path / "partial-run/run.json").read_text())
    assert result["run_status"] == "budget_exhausted"
    assert result["requested_tasks"] == 2
    assert result["tasks"] == 1
    assert len(saved) == 1
    assert saved[0]["llm_calls"][0]["error_code"] == "token_budget_exhausted"
    assert run_metadata["status"] == "budget_exhausted"
    assert completions.requests == []


def test_summary_csv_unions_columns_across_status_rows(tmp_path):
    path = tmp_path / "summary.csv"
    _write_summary_csv(path, [
        {"variant": "B0"},
        {"variant": "B1-M", "requested_tasks": 2},
    ])
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["requested_tasks"] == ""
    assert rows[1]["requested_tasks"] == "2"
