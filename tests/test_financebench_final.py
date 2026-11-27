import json

import httpx
from openai import OpenAI
import pytest
from pydantic import ValidationError

from tradingagents_lite import b1, financebench as fb, rag_benchmark
from tradingagents_lite.financebench_data import Case, OracleCase, write_rows
from tradingagents_lite.financebench_score import assess, score_run
from tradingagents_lite.provider import ModelBackend
from tradingagents_lite.schema import Budget
from tradingagents_lite.token_budget import TokenBudget


CASE = Case(id="financebench_id_04735", company="Example Corp", question="What was revenue in USD millions?",
            split="calibration")
ORACLE = OracleCase(id=CASE.id, items=[{"evidence_id": "oracle-1", "document_id": "public-report",
                                      "page": 12, "excerpt": "ORACLE-ONLY approved excerpt"}])
APPROVED = {"public-report": {"company": CASE.company, "sha256": "original-pdf"}}


class Search:
    def __init__(self, override=None):
        self.requests = []
        self.override = override or {}

    def retrieve(self, query, **scope):
        self.requests.append((query, scope))
        return {"latency_ms": 1.0, "stages": {"rrf": []}, "items": [{
            "document_id": "public-report", "ticker": CASE.company, "chunk_id": str(len(self.requests)),
            "raw_hash": "original-pdf", "dataset_partition": rag_benchmark.PARTITION,
            "eligible_ts": rag_benchmark.CUTOFF.timestamp(), "page": 12,
            "text": "Observed R3 financial fact", **self.override}]}


class Model(fb.MockBackend):
    def __init__(self):
        self.messages = []

    def _call(self, messages, tools, **kwargs):
        self.messages.append((messages.copy(), tools))
        return super()._call(messages, tools, **kwargs)


def test_case_rejects_answer_justification_or_gold_fields():
    for field in ("answer", "justification", "evidence"):
        with pytest.raises(ValidationError):
            Case.model_validate({**CASE.model_dump(), field: "GOLD SENTINEL"})


@pytest.mark.parametrize("arm", fb.ARMS)
def test_gold_oracle_and_retrieval_scopes_are_isolated(arm):
    model, retrieval = Model(), Search()
    record = fb.answer_case(CASE, arm, model, retrieval, APPROVED, ORACLE)
    wire = json.dumps(model.messages)
    assert ("ORACLE-ONLY" in wire) == (arm == "FB3")
    assert ("Observed R3 financial fact" in wire) == (arm in fb.CORE)
    assert "GOLD SENTINEL" not in wire
    assert record["status"] == "ok"
    assert not model.messages[-1][1]
    assert record["tool_calls"] == (2 if arm in fb.CORE else 0)
    for _, scope in retrieval.requests:
        assert scope["ticker"] == CASE.company
        assert scope["partition"] == rag_benchmark.PARTITION and scope["mode"] == "R3"
        assert scope["cutoff"] == rag_benchmark.CUTOFF
    if arm == "FB2":
        assert retrieval.requests[0][0] != retrieval.requests[1][0]
        assert model.messages[1][0][-1]["tool_call_id"] == "mock-1"


@pytest.mark.parametrize("altered", [
    {"ticker": "Other Corp"}, {"raw_hash": "bad-version"}, {"dataset_partition": "private"},
    {"eligible_ts": rag_benchmark.CUTOFF.timestamp() + 1},
])
def test_wrong_company_version_partition_or_future_excerpt_is_never_sent(altered):
    model = Model()
    result = fb.answer_case(CASE, "FB1", model, Search(altered), APPROVED)
    assert result["status"] == "index_provenance_error"
    assert not model.messages and not result["exposed"]


def test_missing_oracle_and_invalid_citation_remain_explicit_rows():
    model = Model()
    result = fb.answer_case(CASE, "FB3", model, None, APPROVED)
    assert result["status"] == "oracle_unavailable" and not model.messages

    class BadModel(Model):
        def _call(self, messages, tools, **kwargs):
            return {"content": json.dumps({"status": "answered", "answer": "1", "evidence_ids": ["foreign"]}),
                    "finish_reason": "stop"}

    result = fb.answer_case(CASE, "FB0", BadModel(), None, APPROVED)
    assert result["status"] == "invalid_citation"
    assert result["citations"] == ["foreign"] and result["valid_citations"] == []


def test_numerical_units_and_uncertain_text_have_distinct_policies():
    row = {"status": "ok", "answer_status": "answered", "answer": "1.5 billion USD"}
    assert assess(row, "$1,500.0", CASE.question) == ("correct", "normalized_decimal_tolerance", True)
    assert assess({**row, "answer": "100 bps"}, "1%", "What was margin?")[0] == "correct"
    assert assess({**row, "answer": "1400"}, "$1500", CASE.question)[0] == "incorrect"
    assert assess({**row, "answer": "The business generated recurring income."},
                  "Yes, operating income was recurrent.", "Explain.")[0] == "review"
    assert assess({**row, "status": "timeout"}, "1500", CASE.question)[0] == "incorrect"


def test_real_scope_gate_applies_before_corpus_or_client_access(tmp_path):
    with pytest.raises(ValueError, match="explicit authorization"):
        fb.run(tmp_path / "nonexistent", tmp_path / "out", scope="test", backend="mock")


def test_scripted_smoke_cannot_unlock_formal_model_runs():
    predictions = [{"id": key, "arm": arm, "status": "ok", "llm_calls": [],
                    "exposed": [] if arm == "FB0" else [{"evidence_id": "mock"}],
                    "retrieval_trace": [{"query": "mock"}] if arm in fb.CORE else []}
                   for key in fb.SMOKE_IDS for arm in fb.ARMS]
    assert fb._smoke_ready(predictions, real=False)
    assert not fb._smoke_ready(predictions, real=True)


def test_checkpoint_tampering_and_unarchived_ledger_attempt_fail_closed(tmp_path):
    arm, key = "FB0", CASE.id
    plan = [{"id": key, "arm": arm}]
    folder = tmp_path / "cells"
    folder.mkdir()
    record = fb.answer_case(CASE, arm, fb.MockBackend(), None, APPROVED)
    checkpoint = folder / f"{key}-{arm}.json"
    checkpoint.write_text(json.dumps({**record, "record_sha256": b1._json_hash(record)}))
    assert fb._archived(tmp_path, plan)[0]["id"] == key
    checkpoint.write_text(checkpoint.read_text().replace("Scripted", "Injected"))
    with pytest.raises(ValueError, match="checkpoint integrity"):
        fb._archived(tmp_path, plan)
    ledger = TokenBudget(tmp_path / "ledger.sqlite")
    ledger.reserve(experiment_id="live-scope", task_id=record["task_id"], phase="A1", model=fb.MODEL, logical_call=1,
                   attempt_number=1, request_sha256="f" * 64, request_bytes=200, max_output_tokens=512)
    with pytest.raises(ValueError, match="attempt records do not match"):
        fb._archived(tmp_path, [], ledger, "live-scope")


def test_scoring_is_separate_requires_frozen_predictions_and_preserves_review(tmp_path):
    bundle, run = tmp_path / "bundle", tmp_path / "run"
    bundle.mkdir()
    run.mkdir()
    gold = tmp_path / "gold.jsonl"
    write_rows(bundle / "questions.jsonl", [CASE.model_dump(mode="json")])
    write_rows(gold, [{"id": CASE.id, "answer": "Recurring income from services.",
                       "anchors": [["public-report", 12]]}])
    from tradingagents_lite.data import sha256
    (bundle / "manifest.json").write_text(json.dumps({"questions_sha256": sha256(bundle / "questions.jsonl"),
                                                      "gold_sha256": sha256(gold)}))
    prediction = fb.answer_case(CASE, "FB0", fb.MockBackend(), None, APPROVED)
    write_rows(run / "predictions.jsonl", [prediction])
    manifest = {"status": "running", "predictions_sha256": sha256(run / "predictions.jsonl"),
                "backend": "mock", "scope": "smoke", "input_manifest_sha256": sha256(bundle / "manifest.json"),
                "config_sha256": "mock-config", "git_sha": "offline-test", "actual_models": [],
                "plan": [{"id": CASE.id, "arm": "FB0"}]}
    path = run / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="finalized"):
        score_run(run, bundle, gold, tmp_path / "blocked")
    path.write_text(json.dumps({**manifest, "status": "complete"}))
    report = score_run(run, bundle, gold, tmp_path / "score")
    assert report["arms"]["FB0"]["answer_accuracy"]["pending_review"] == 1
    assert report["arms"]["FB0"]["answer_accuracy"]["n"] == 1
    assert report["arms"]["FB0"]["total_tokens"] == 0


def test_sdk_in_memory_uses_frozen_json_and_reserves_before_http(tmp_path):
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline", "object": "chat.completion", "created": 1, "model": fb.MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps({"status": "answered", "answer": "1 USD million",
                                                           "evidence_ids": []})}}],
            "usage": {"prompt_tokens": 101, "completion_tokens": 19, "total_tokens": 120},
        })

    with OpenAI(api_key="offline-fixture", base_url=fb.DEFAULT_BASE_URL, max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        for cap, expected in ((2000, "ok"), (100, "token_budget_exhausted")):
            ledger = TokenBudget(tmp_path / f"test-{cap}.sqlite", experiment_limit=cap)
            backend = ModelBackend(client, fb.MODEL, Budget(max_tokens=512), None, max_retries=0,
                                   temperature=0, reasoning_effort="low", token_budget=ledger,
                                   phase="A1", require_model_identity=True)
            result = fb.answer_case(CASE, "FB0", backend, None, APPROVED)
            assert result["status"] == expected
            attempt, = ledger.attempts()
            assert attempt["charged_tokens"] == (120 if expected == "ok" else 0)
            assert attempt["status"] == ("succeeded" if expected == "ok" else "denied")
    assert len(payloads) == 1
    request = payloads[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["model"] == fb.MODEL and request["temperature"] == 0 and request["reasoning_effort"] == "low"
    assert "tools" not in request and len(request["messages"]) == 3
