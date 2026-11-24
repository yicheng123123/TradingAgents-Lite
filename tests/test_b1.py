from datetime import UTC, date, datetime
from pathlib import Path
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from test_stage1 import TASK, bars
from tradingagents_lite.b1 import request_cost_ceiling
from tradingagents_lite.data import HistoricalData, MarketView, read_tasks
from tradingagents_lite.evidence import Evidence, EvidenceStore
from tradingagents_lite.provider import ModelBackend, ModelCallError
from tradingagents_lite.prompts import prompt_hashes
from tradingagents_lite.runner import run
from tradingagents_lite.schema import Action, Budget, MarketReport, Proposal
from tradingagents_lite.workflow import MomentumBackend, build_workflow, call_message


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


def test_model_backend_retries_timeout_once_and_counts_every_attempt():
    completions = FakeCompletions([TimeoutError("private transport detail"), fake_response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = ModelBackend(client, "glm-5.3", Budget(max_tokens=128), None)
    result = backend._call([{"role": "user", "content": "test"}], [])
    assert result["content"] == "{}"
    assert backend.usage == {
        "calls": 2, "logical_calls": 1, "retries": 1, "failed_calls": 0,
        "prompt_tokens": 12, "completion_tokens": 5, "missing_usage_calls": 0,
    }
    assert [call["status"] for call in backend.call_log] == ["retrying", "ok"]
    assert backend.call_log[0]["error_code"] == "timeout"
    assert "private transport detail" not in json.dumps(backend.call_log)
    assert completions.requests[0]["thinking"] == {"type": "enabled"}
    assert completions.requests[0]["reasoning_effort"] == "low"


def test_model_backend_stops_at_retry_budget_and_redacts_timeout():
    completions = FakeCompletions([TimeoutError("sensitive"), TimeoutError("sensitive again")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = ModelBackend(client, "glm-5.3", Budget(), None, max_retries=1)
    with pytest.raises(ModelCallError) as error:
        backend._call([{"role": "user", "content": "test"}], [])
    assert (error.value.code, error.value.attempts) == ("timeout", 2)
    assert len(completions.requests) == 2
    assert backend.usage["calls"] == 2
    assert backend.usage["retries"] == 1
    assert backend.usage["failed_calls"] == 1
    assert all("sensitive" not in json.dumps(row) for row in backend.call_log)


def test_request_byte_cap_blocks_before_network_call():
    completions = FakeCompletions([fake_response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = ModelBackend(client, "glm-5.3", Budget(), None, max_prompt_bytes=10)
    with pytest.raises(ModelCallError, match="prompt_byte_limit"):
        backend._call([{"role": "user", "content": "larger than ten bytes"}], [])
    assert completions.requests == []
    assert backend.usage["calls"] == 0


def test_invalid_structured_action_is_rejected():
    completions = FakeCompletions([fake_response(
        '{"action":"short","rationale":"invalid action","evidence_ids":[]}')])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = ModelBackend(client, "glm-5.3", Budget(), None)
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
    assert smoke["list_price_cost_ceiling_cny"] == pytest.approx(9.59488)
    assert development["list_price_cost_ceiling_cny"] == pytest.approx(95.9488)
    assert smoke["max_http_attempts_including_retries"] == 32
