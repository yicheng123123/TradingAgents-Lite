import json

import httpx
from openai import OpenAI
import pytest

from tradingagents_lite.provider import ModelBackend, ModelCallError
from tradingagents_lite.schema import Budget
from tradingagents_lite.token_budget import TokenBudget, TokenBudgetExceeded


def test_usage_settlement_deduplicates_reasoning_and_cached_tokens_and_persists(tmp_path):
    ledger = tmp_path / "project.sqlite"
    budget = TokenBudget(ledger, global_limit=200, phase_limits={"B1": 150},
                         experiment_limit=100)
    attempt_id, reservation = budget.reserve(
        experiment_id="smoke-001", task_id="dev-01", phase="B1", model="glm-5.3",
        logical_call=1, attempt_number=1, request_sha256="a" * 64,
        request_bytes=10, max_output_tokens=10, protocol_allowance=2,
        separate_reasoning_tokens=10,
    )
    assert reservation == 32
    charged = budget.settle_usage(
        attempt_id, input_tokens=10, output_tokens=8, reasoning_tokens=3,
        cached_tokens=4,
    )
    assert charged == 18  # reasoning and cached tokens are included in provider totals

    reopened = TokenBudget(ledger, global_limit=200, phase_limits={"B1": 150},
                           experiment_limit=100)
    snapshot = reopened.snapshot(experiment_id="smoke-001", phase="B1")
    assert snapshot["global_used"] == 18
    assert snapshot["phase_used"] == 18
    assert snapshot["experiment_used"] == 18
    assert snapshot["input_tokens"] == 10
    assert snapshot["output_tokens"] == 8
    assert snapshot["reasoning_tokens"] == 3
    assert snapshot["cached_tokens"] == 4
    row = reopened.attempts(experiment_id="smoke-001")[0]
    assert row["reasoning_in_output"] == 1
    assert row["charged_tokens"] == 18


def test_failed_attempt_keeps_reservation_and_denial_happens_before_http(tmp_path):
    ledger = TokenBudget(tmp_path / "project.sqlite", global_limit=100,
                         phase_limits={"B1": 45})
    attempt_id, reservation = ledger.reserve(
        experiment_id="b1", task_id="dev-01", phase="B1", model="glm-5.3",
        logical_call=1, attempt_number=1, request_sha256="b" * 64,
        request_bytes=10, max_output_tokens=10, protocol_allowance=10,
    )
    assert ledger.settle_unknown(attempt_id, error_code="timeout", status="failed") == reservation
    with pytest.raises(TokenBudgetExceeded) as denied:
        ledger.reserve(
            experiment_id="b1", task_id="dev-01", phase="B1", model="glm-5.3",
            logical_call=1, attempt_number=2, request_sha256="b" * 64,
            request_bytes=1, max_output_tokens=2, protocol_allowance=1,
            separate_reasoning_tokens=2,
        )
    assert denied.value.reason == "phase_limit"
    rows = ledger.attempts(experiment_id="b1")
    assert [row["status"] for row in rows] == ["failed", "denied"]
    assert rows[0]["usage_missing"] == 1
    assert rows[0]["charged_tokens"] == reservation
    assert rows[1]["is_retry"] == 1
    assert rows[1]["charged_tokens"] == 0

    global_limited = TokenBudget(tmp_path / "project.sqlite", global_limit=45,
                                 phase_limits={"B1": 100, "A1": 100})
    with pytest.raises(TokenBudgetExceeded) as global_denied:
        global_limited.reserve(
            experiment_id="other", task_id="dev-02", phase="A1", model="glm-5.3",
            logical_call=1, attempt_number=1, request_sha256="c" * 64,
            request_bytes=1, max_output_tokens=2, protocol_allowance=1,
            separate_reasoning_tokens=2,
        )
    assert global_denied.value.reason == "global_limit"


def test_reasoning_detail_only_adds_tokens_when_it_exceeds_output(tmp_path):
    budget = TokenBudget(tmp_path / "project.sqlite")
    attempt_id, _ = budget.reserve(
        experiment_id="reasoning", task_id="dev-01", phase="B1", model="glm-5.3",
        logical_call=1, attempt_number=1, request_sha256="d" * 64,
        request_bytes=10, max_output_tokens=10, protocol_allowance=1,
    )
    charged = budget.settle_usage(
        attempt_id, input_tokens=10, output_tokens=2, reasoning_tokens=5,
        cached_tokens=None,
    )
    row = budget.attempts(experiment_id="reasoning")[0]
    assert charged == 15
    assert row["reasoning_in_output"] == 0


def test_real_openai_sdk_serializes_glm_fields_through_extra_body_in_memory(tmp_path):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chatcmpl-memory-test",
            "object": "chat.completion",
            "created": 1,
            "model": "glm-5.3",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"},
                         "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 9,
                "total_tokens": 19,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        })

    client = OpenAI(
        api_key="offline-test-token-only",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    backend = ModelBackend(
        client, "glm-5.3", Budget(max_tokens=64), None, max_retries=0,
        token_budget=TokenBudget(tmp_path / "project.sqlite"),
    )
    try:
        response = backend._call([{"role": "user", "content": "offline regression"}], [])
    finally:
        client.close()

    assert response["content"] == "{}"
    assert len(captured) == 1
    assert captured[0]["thinking"] == {"type": "enabled"}
    assert captured[0]["reasoning_effort"] == "low"
    assert "extra_body" not in captured[0]
    assert backend.usage["total_tokens_charged"] == 19
    assert backend.usage["reasoning_tokens"] == 3
    assert backend.usage["cached_tokens"] == 4
    assert backend.usage["reasoning_detail_missing_calls"] == 0
    assert backend.usage["cached_detail_missing_calls"] == 0
    attempt = backend.token_budget.attempts()[0]
    assert attempt["input_tokens"] == 10
    assert attempt["output_tokens"] == 9
    assert attempt["reasoning_tokens"] == 3
    assert attempt["cached_tokens"] == 4
    assert attempt["charged_tokens"] == 19


def test_budget_denial_prevents_real_sdk_transport_call(tmp_path):
    called = []
    client = OpenAI(
        api_key="offline-test-token-only",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        http_client=httpx.Client(transport=httpx.MockTransport(
            lambda request: (called.append(request) or httpx.Response(200, json={}))
        )),
        max_retries=0,
    )
    backend = ModelBackend(
        client, "glm-5.3", Budget(max_tokens=64), None, max_retries=0,
        token_budget=TokenBudget(tmp_path / "project.sqlite", global_limit=1,
                                 phase_limits={"B1": 1}),
    )
    try:
        with pytest.raises(ModelCallError) as error:
            backend._call([{"role": "user", "content": "will not be sent"}], [])
    finally:
        client.close()

    assert error.value.code == "token_budget_exhausted"
    assert error.value.budget_exhausted is True
    assert called == []
    assert backend.usage["calls"] == 0
    assert backend.token_budget.attempts()[0]["status"] == "denied"


def test_success_without_usage_keeps_full_reservation(tmp_path):
    from types import SimpleNamespace

    response = SimpleNamespace(
        id="usage-missing-response", model="glm-5.3",
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}", tool_calls=None))],
        usage=None,
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: response)))
    backend = ModelBackend(
        client, "glm-5.3", Budget(max_tokens=64), None, max_retries=0,
        token_budget=TokenBudget(tmp_path / "project.sqlite"),
    )
    backend._call([{"role": "user", "content": "no usage fixture"}], [])
    attempt = backend.token_budget.attempts()[0]
    assert attempt["status"] == "usage_missing"
    assert attempt["usage_missing"] == 1
    assert attempt["charged_tokens"] == attempt["reservation_tokens"]
    assert backend.usage["missing_usage_calls"] == 1
    assert backend.usage["total_tokens_charged"] == attempt["reservation_tokens"]


def test_flash_v2_accepts_the_previous_5326_byte_prompt_under_new_cap(tmp_path):
    captured = []

    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json={
            "id": "flash-prompt-limit-regression", "object": "chat.completion", "created": 1,
            "model": "glm-5.3-flash",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
        })

    messages = [{"role": "user", "content": ""}]
    tools = []
    target = 5326
    low, high = 0, target
    while low <= high:
        middle = (low + high) // 2
        messages[0]["content"] = "x" * middle
        size = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False,
                              sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
        if size == target:
            break
        if size < target:
            low = middle + 1
        else:
            high = middle - 1
    else:
        raise AssertionError("could not construct the exact archived request size")

    client = OpenAI(api_key="offline-test-token-only", base_url="https://open.bigmodel.cn/api/paas/v4",
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)
    backend = ModelBackend(client, "glm-5.3-flash", Budget(max_tokens=512), None,
        max_retries=0, max_prompt_bytes=8000, require_model_identity=True,
        token_budget=TokenBudget(tmp_path / "flash.sqlite"))
    try:
        assert backend._call(messages, tools)["content"] == "{}"
    finally:
        client.close()
    assert len(captured) == 1 and backend.call_log[0]["request_bytes"] == 5326
    assert captured[0]["model"] == "glm-5.3-flash"
    assert captured[0]["max_tokens"] == 512
    assert captured[0]["thinking"] == {"type": "enabled"}
    assert captured[0]["reasoning_effort"] == "low"


def test_flash_v2_rejects_a_different_actual_model_and_keeps_usage_charged(tmp_path):
    from types import SimpleNamespace

    response = SimpleNamespace(
        id="wrong-model-response", model="glm-5.3",
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2)),
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    ledger = TokenBudget(tmp_path / "wrong-model.sqlite")
    backend = ModelBackend(client, "glm-5.3-flash", Budget(max_tokens=512), None, max_retries=0,
        require_model_identity=True, token_budget=ledger)
    with pytest.raises(ModelCallError, match="model_identity_mismatch"):
        backend._call([{"role": "user", "content": "offline identity check"}], [])
    attempt, = ledger.attempts()
    assert attempt["model"] == "glm-5.3-flash"
    assert attempt["status"] == "failed" and attempt["charged_tokens"] == 25
    assert backend.call_log[0]["actual_model"] == "glm-5.3"
    assert backend.call_log[0]["error_code"] == "model_identity_mismatch"


def test_flash_v2_records_and_rejects_truncated_output_without_retry(tmp_path):
    from types import SimpleNamespace

    response = SimpleNamespace(
        id="truncated-flash-response", model="glm-5.3-flash",
        choices=[SimpleNamespace(message=SimpleNamespace(content="{", tool_calls=None),
                                 finish_reason="length")],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=512,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=400)),
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    ledger = TokenBudget(tmp_path / "truncated.sqlite")
    backend = ModelBackend(client, "glm-5.3-flash", Budget(max_tokens=512), None, max_retries=0,
        require_model_identity=True, token_budget=ledger)
    with pytest.raises(ModelCallError, match="output_truncated"):
        backend._call([{"role": "user", "content": "offline truncation check"}], [])
    assert backend.usage["calls"] == 1 and backend.usage["retries"] == 0
    assert backend.call_log[0]["finish_reason"] == "length"
    assert backend.call_log[0]["error_code"] == "output_truncated"
    attempt, = ledger.attempts()
    assert attempt["status"] == "failed" and attempt["charged_tokens"] == 532
