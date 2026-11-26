from datetime import date
import json
import sqlite3
from types import SimpleNamespace

import httpx
from openai import OpenAI
import pytest

from tradingagents_lite import b1, b1_v3
from tradingagents_lite.provider import DEFAULT_BASE_URL, ModelBackend
from tradingagents_lite.schema import Budget, Task
from tradingagents_lite.structured_output import StructuredOutput, validate_reply
from tradingagents_lite.token_budget import TokenBudget
from tradingagents_lite.workflow import build_workflow


REPORT = {"view": "Momentum is positive.", "evidence_ids": ["observed"]}
PROPOSAL = {"action": "buy", "rationale": "Positive momentum.", "evidence_ids": ["observed"]}
EVIDENCE = {"observed": {"evidence_id": "observed", "return_5": 0.02}}


@pytest.mark.parametrize(("body", "finish", "code", "loc"), [
    ("{", "stop", "invalid_json", []),
    ('{"evidence_ids":["observed"]}', "stop", "missing_field", ["view"]),
    (json.dumps({**REPORT, "private-model-key": "private-model-value"}), "stop", "extra_field", ["<extra>"]),
    (json.dumps({**REPORT, "view": "x" * 1001}), "stop", "string_too_long", ["view"]),
    (json.dumps({**REPORT, "evidence_ids": ["observed"] * 5}), "stop", "list_too_long", ["evidence_ids"]),
    (json.dumps({**REPORT, "evidence_ids": [123]}), "stop", "invalid_field_type", ["evidence_ids", 0]),
    (json.dumps(REPORT), "length", "output_truncated", []),
    (json.dumps({**REPORT, "evidence_ids": ["secret-fabricated-id"]}), "stop", "invalid_citation", ["evidence_ids", 0]),
])
def test_diagnostics_record_safe_fields_only(body, finish, code, loc):
    value, event = validate_reply("market_report", {"content": body, "finish_reason": finish}, EVIDENCE)
    assert value is None and event["valid"] is False
    assert event["error_count"] == 1
    assert event["errors"][0]["type"] == code and event["errors"][0]["loc"] == loc
    assert event["response_bytes"] == len(body.encode())
    assert event["finish_reason"] == finish and len(event["schema_hash"]) == 64
    assert "private-model" not in json.dumps(event) and "secret-fabricated" not in json.dumps(event)
    assert "input" not in event["errors"][0]


def test_fence_removal_requires_one_outer_fence_and_buy_requires_citation():
    value, event = validate_reply("market_report", {"content": "\x60\x60\x60json\n" + json.dumps(REPORT)
                                                   + "\n\x60\x60\x60"}, EVIDENCE)
    assert value.model_dump() == REPORT and event["removed_markdown_fence"]
    _, event = validate_reply("proposal", {"content": json.dumps({**PROPOSAL, "evidence_ids": []})}, EVIDENCE)
    assert event["errors"] == [{"loc": ["evidence_ids"], "type": "invalid_citation"}]


def _sdk(tmp_path, responses, cap=50_000):
    requests = []
    replies = iter(responses)

    def handle(request):
        requests.append(json.loads(request.content))
        content, tools, finish = next(replies)
        return httpx.Response(200, json={
            "id": "offline-test", "object": "chat.completion", "created": 1, "model": "glm-5.3-flash",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content, "tool_calls": tools}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
        })

    ledger = TokenBudget(tmp_path / "budget.sqlite", experiment_limit=cap)
    client = OpenAI(api_key="test-only", base_url=DEFAULT_BASE_URL, max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    backend = ModelBackend(client, "glm-5.3-flash", Budget(max_tool_rounds=1, max_tokens=64), None,
                           max_retries=0, token_budget=ledger, require_model_identity=True)
    return client, backend, requests, ledger


def _reply(value):
    return json.dumps(value), None, "stop"


@pytest.mark.parametrize("trader_valid", [True, False])
def test_workflow_repairs_one_output_without_replaying_tools(tmp_path, trader_valid):
    call = {"id": "call-1", "type": "function", "function": {"name": "market_indicators", "arguments": "{}"}}
    responses = [("", [call], "tool_calls"), ("{", None, "stop"), _reply(REPORT),
                 _reply(PROPOSAL if trader_valid else {**PROPOSAL, "evidence_ids": ["fabricated"]})]
    client, backend, requests, _ = _sdk(tmp_path, responses)
    task = Task(task_id="sample", symbol="TEST", decision_date=date(2024, 1, 4), horizon_sessions=5)
    view = SimpleNamespace(task=task, indicators=lambda: {**EVIDENCE["observed"], "daily_volatility_20": 0.01})
    with client:
        state = build_workflow(view, backend, backend.budget, flash_v3=True,
                               compact_tool_results=True, max_tool_calls_per_round=4).invoke({})
    assert len(requests) == 4 and len(state["tool_trace"]) == 1
    assert "tools" in requests[0] and "response_format" not in requests[0]
    assert all(req["response_format"] == {"type": "json_object"} and "tools" not in req for req in requests[1:])
    assert requests[2]["messages"][:-1] == requests[1]["messages"]
    assert "invalid_json" in requests[2]["messages"][-1]["content"]
    assert state["structured_output_trace"][1]["charged_tokens"] == 32
    assert bool(state["errors"]) is not trader_valid
    if not trader_valid:
        assert state["errors"][0]["code"] == "invalid_citation"
        assert state.get("proposal") is None


def test_failed_repair_stops_and_charges_once(tmp_path):
    client, backend, requests, ledger = _sdk(tmp_path, [("{", None, "length"), _reply({})])
    with client:
        output, events, fault = StructuredOutput(backend).generate(
            "market_report", [{"role": "user", "content": "Return JSON"}], EVIDENCE)
    assert output is None and fault is None and len(requests) == 2
    assert [item["errors"][0]["type"] for item in events] == ["output_truncated", "missing_field"]
    assert events[1]["charged_tokens"] == 32
    assert sum(row["charged_tokens"] for row in ledger.attempts()) == 64


def test_repair_is_denied_atomically_before_http(tmp_path):
    client, backend, requests, ledger = _sdk(tmp_path, [("{", None, "stop")], cap=350)
    with client:
        output, events, fault = StructuredOutput(backend).generate(
            "market_report", [{"role": "user", "content": "JSON"}], EVIDENCE)
    assert output is None and fault.code == "token_budget_exhausted" and len(requests) == 1
    assert events[1]["charged_tokens"] == 0
    assert [row["status"] for row in ledger.attempts()] == ["succeeded", "denied"]


def test_resume_skips_terminal_cells_and_continues_after_task_output_error(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(Task(task_id=task, symbol="TEST", decision_date=date(2024, 1, day),
                                  horizon_sessions=5).model_dump_json() + "\n"
                             for day, task in enumerate(b1.SMOKE_TASK_IDS, 4)))
    protocol = {"budget": Budget(max_tokens=64).model_dump(), "model_config": {}}
    ledger = TokenBudget(tmp_path / "ledger.sqlite", experiment_limit=50_000)
    monkeypatch.setenv("TALITE_API_KEY", "test-only")
    monkeypatch.setenv("TALITE_TOKEN_BUDGET_DB", str(ledger.db_path))
    monkeypatch.setattr(b1_v3, "_frozen_protocol", lambda *_: protocol)
    monkeypatch.setattr(b1, "_git", lambda *_: "offline-test")
    monkeypatch.setattr(b1_v3, "client_from_env", lambda **_: (
        SimpleNamespace(close=lambda: None, base_url=DEFAULT_BASE_URL), "glm-5.3-flash", "open.bigmodel.cn"))
    outcomes = iter([("workflow_error", "missing_field"), ("model_error", "http_503"), ("ok", None), ("ok", None)])
    executed = []

    def fake_run(task_slice, market, path, *args, run_label, protocol_metadata, **kwargs):
        task_id = json.loads(task_slice.read_text())["task_id"]
        status, error = next(outcomes)
        executed.append((task_id, run_label))
        path.mkdir(parents=True)
        (path / "run.json").write_text(json.dumps({"ablation": run_label, "protocol_metadata": protocol_metadata}))
        row = {"task_id": task_id, "status": status, "errors": [{"code": error}] if error else [],
               "structured_output_trace": [], "llm_calls": [], "usage": {"calls": 0, "total_tokens_charged": 0}}
        with sqlite3.connect(path / "state.sqlite") as conn:
            conn.execute("CREATE TABLE outcomes (task_id TEXT, record TEXT)")
            conn.execute("INSERT INTO outcomes VALUES (?,?)", (task_id, json.dumps(row)))

    monkeypatch.setattr(b1_v3, "run", fake_run)
    out = tmp_path / "smoke"
    result = b1_v3.run_b1_v3(tasks, tmp_path, tmp_path, tmp_path, out)
    assert result["status"] == "stopped_system_fault" and result["system_fault"] == "http_503"
    assert len(executed) == 2
    resumed = b1_v3.run_b1_v3(tasks, tmp_path, tmp_path, tmp_path, out, resume=True)
    assert resumed["status"] == "completed_with_task_failures" and resumed["valid_pairs"] == 1
    assert len(executed) == len(set(executed)) == 4
    assert resumed["failed_instances"] == 2 and resumed["unattempted_instances"] == 0
