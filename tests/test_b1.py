from datetime import UTC, date, datetime
from pathlib import Path
import json
import csv
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from test_stage1 import TASK, bars
from tradingagents_lite.b1 import _completion_check, _write_summary_csv, request_cost_ceiling
from tradingagents_lite.data import HistoricalData, MarketView, read_tasks
from tradingagents_lite.evidence import Evidence, EvidenceStore
from tradingagents_lite.provider import ModelBackend, ModelCallError, _trader_context, client_from_env
from tradingagents_lite.prompts import prompt_hashes
from tradingagents_lite.runner import run
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
