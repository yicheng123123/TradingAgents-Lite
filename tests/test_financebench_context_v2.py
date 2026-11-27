"""Synthetic offline regression tests for FinanceBench V2 context delivery."""
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

from tradingagents_lite import financebench
from tradingagents_lite import rag_benchmark
from tradingagents_lite.financebench_data import Case, GOLD, OracleCase, OracleExcerpt


def _case(question="What was operating income in 2022?", identity="fixture"):
    return SimpleNamespace(id=identity, company="SYN", question=question, unavailable=())


def _item(evidence_id="chunk:fixture", text=None, *, document="fixture-doc", page=0):
    text = text or "Income statement USD thousands 2022 2021 Operating income 480 320"
    return {"evidence_id": evidence_id, "document_id": document, "page": page, "excerpt": text}


def test_query_aware_selection_delivers_late_financial_row_and_header():
    header = "Statements of income (USD thousands) | 2022 | 2021"
    filler = " ".join(f"Expense line {row} | 35 | 20" for row in range(40))
    target = "Operating income | 480 | 320"
    raw = " ".join((header, filler, target))
    assert raw.index(target) > 720

    context = financebench.CaseSearch(_case(), None, {})
    payload = context._deliver([_item(text=raw)], query="operating income 2022 2021",
                               requested_limit=1, preserve_followup=False)
    excerpt, = [entry["excerpt"] for entry in payload["items"]]
    assert header in excerpt
    assert target in excerpt
    assert len(excerpt.encode("utf-8")) < len(raw.encode("utf-8"))


def test_row_years_units_and_values_remain_bound_in_one_delivery():
    raw = ("Consolidated statement (in USD millions) fiscal year 2023 2022 "
           + "irrelevant expense value 1 2 " * 90
           + "Net income attributable to shareholders 2023 7.25 2022 6.10")
    context = financebench.CaseSearch(_case("What was net income in 2023 and 2022?"), None, {})
    payload = context._deliver([_item(text=raw)], query="net income 2023 2022", requested_limit=1)
    excerpt, = [entry["excerpt"] for entry in payload["items"]]
    assert "USD millions" in excerpt
    assert "fiscal year 2023 2022" in excerpt
    assert "Net income attributable to shareholders 2023 7.25 2022 6.10" in excerpt


def test_query_selection_can_include_multiple_distant_financial_rows():
    header = "Income statement in USD thousands 2022 2021"
    first = "Cash from operating activities 880 700"
    second = "Total current liabilities 2500 2700"
    filler = " ".join(f"Other expense item {i} 45 40" for i in range(45))
    raw = " ".join((header, first, filler, second))
    context = financebench.CaseSearch(
        _case("Compare cash from operating activities and current liabilities in 2022 and 2021."), None, {})
    payload = context._deliver([_item(text=raw)], query="cash from operations current liabilities",
                               requested_limit=1, preserve_followup=False)
    excerpt, = [entry["excerpt"] for entry in payload["items"]]
    assert header in excerpt
    assert first in excerpt
    assert second in excerpt


def test_semantically_retrieved_candidate_without_lexical_overlap_is_not_prefix_clipped():
    raw = "Evidence from the approved report: net income was USD 12.4 million for 2024."
    context = financebench.CaseSearch(_case("How did the business perform?"), None, {})
    payload = context._deliver([_item(text=raw)], query="qualitative performance", requested_limit=1)
    excerpt, = [entry["excerpt"] for entry in payload["items"]]
    assert excerpt == raw
    assert context.trace[0]["candidates"][0]["selection_reason"] == "retriever_candidate_fallback"


def test_second_round_can_deliver_new_evidence_with_reserved_budget():
    context = financebench.CaseSearch(
        _case("What were operating income and cash from operations in 2022?"), None, {})
    first = context._deliver([_item("chunk:first", "USD thousands 2022 Operating income 480 320")],
        query="operating income 2022", round_index=1, requested_limit=1,
        request_sha256="first-query-request", request_bytes=321, tool_call_id="call-1",
        role="tool", preserve_followup=True)
    after_first = context.remaining
    assert first["items"] and after_first >= int(financebench.RAG.context_bytes * 0.35)

    second = context._deliver([_item("chunk:second", "USD thousands 2022 Cash from operations 600 500")],
        query="cash from operations 2022", round_index=2, requested_limit=1,
        request_sha256="second-query-request", request_bytes=456, tool_call_id="call-2",
        role="tool", preserve_followup=False)
    assert [item["evidence_id"] for item in second["items"]] == ["chunk:second"]
    assert context.remaining < after_first
    assert context.used_context_bytes <= financebench.RAG.context_bytes
    assert context.trace[0]["remaining_context_budget_after"] >= int(financebench.RAG.context_bytes * 0.35)


def test_duplicate_evidence_and_content_do_not_consume_more_budget():
    context = financebench.CaseSearch(_case(), None, {})
    item = _item(text="USD thousands 2022 Operating income 480 320")
    context._deliver([item], query="operating income 2022", requested_limit=1)
    remaining = context.remaining

    duplicate_id = context._deliver([item], query="operating income 2022", round_index=2, requested_limit=1)
    assert duplicate_id["items"] == []
    assert context.remaining == remaining
    assert context.trace[-1]["candidates"][0]["skip_reason"] == "duplicate_evidence_id"

    duplicate_content = context._deliver([_item("chunk:other", item["excerpt"])],
        query="operating income 2022", round_index=2, requested_limit=1)
    assert duplicate_content["items"] == []
    assert context.remaining == remaining
    assert context.trace[-1]["candidates"][0]["skip_reason"] == "duplicate_content"


def test_full_serialized_prompt_still_obeys_original_8000_byte_limit():
    context = financebench.CaseSearch(_case(), None, {})
    payload = context._deliver([_item()], query="operating income 2022", requested_limit=1,
        base_messages=[{"role": "user", "content": "x" * 7850}], expect_final=True)
    assert payload["items"] == []
    assert payload["status"] == "budget_insufficient"
    assert payload["reason"] == "final_request_byte_limit"
    assert context.remaining == financebench.RAG.context_bytes
    assert context.trace[0]["candidates"][0]["prospective_request_bytes"] > financebench.MAX_PROMPT_BYTES


def test_trace_reconstructs_candidate_to_delivery_and_request_sizes():
    chunk_id = "fixture-chunk-001"
    raw = "Statements (USD thousands) 2022 2021 Operating income 480 320"

    class Retriever:
        def retrieve(self, query, **kwargs):
            return {"items": [{"chunk_id": chunk_id, "document_id": "fixture-doc", "page": 7,
                               "text": raw, "ticker": "SYN", "raw_hash": "fixture-hash",
                               "dataset_partition": rag_benchmark.PARTITION,
                               "eligible_ts": rag_benchmark.CUTOFF.timestamp()}],
                    "stages": {"bm25": [{"id": chunk_id, "rank": 1, "score": 2.0}],
                               "dense": [{"id": chunk_id, "rank": 2, "score": 0.8}],
                               "rrf": [{"id": chunk_id, "rank": 1, "score": 0.03}]},
                    "latency_ms": 3.5}

    case = _case()
    context = financebench.CaseSearch(case, Retriever(), {
        "fixture-doc": {"company": "SYN", "sha256": "fixture-hash"}})
    payload = context.search("operating income 2022", 1, round_index=1,
        request_sha256="prompt-request-sha", request_bytes=1234, preserve_followup=False)
    context.stamp_delivery_request("delivered-prompt-sha", 2345)
    trace = context.trace[0]
    assert payload["items"]
    assert trace["original_question"] == case.question
    assert trace["query"] == "operating income 2022"
    assert trace["requested_limit"] == 1
    assert {row["id"] for row in trace["candidate_rankings"]["bm25"]} == {chunk_id}
    assert trace["selected_chunk_ids"] == [chunk_id]
    assert trace["delivered_evidence"][0]["evidence_id"] == f"chunk:{chunk_id}"
    assert trace["candidates"][0]["raw_length_bytes"] == len(raw.encode("utf-8"))
    assert trace["candidates"][0]["delivered_length_bytes"] > 0
    assert trace["model_request_sha256"] == "prompt-request-sha"
    assert trace["model_request_bytes"] == 1234
    assert trace["delivery_request_sha256"] == "delivered-prompt-sha"
    assert trace["delivery_request_bytes"] == 2345
    assert trace["budgeted_serialized_message_bytes"] == financebench._message_size(
        payload, "user")
    assert trace["remaining_context_budget_before"] - trace["remaining_context_budget_after"] == \
        trace["budgeted_serialized_message_bytes"]
    assert context.used_context_bytes <= financebench.RAG.context_bytes


def test_oracle_projection_excludes_gold_and_inference_cannot_open_gold(monkeypatch):
    assert "answer" not in OracleCase.model_fields
    assert "gold" not in OracleCase.model_fields
    assert "gold_path" not in signature(financebench.answer_case).parameters

    oracle = OracleCase(id="financebench_id_00001", items=(OracleExcerpt(
        evidence_id="oracle-1", document_id="fixture-doc", page=0,
        excerpt="USD thousands 2022 Operating income 480 320"),))
    payload = financebench.CaseSearch(_case(), None, {})._deliver(
        (item.model_dump() for item in oracle.items), query="operating income 2022",
        requested_limit=1, preserve_followup=False)
    assert "answer" not in str(payload).casefold()
    assert "gold" not in str(payload).casefold()

    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path.resolve() == GOLD.resolve():
            raise AssertionError("Inference attempted to read the private Gold file")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    case = Case(id="financebench_id_00001", company="SYN", question="What was revenue?",
                split="calibration")
    record = financebench.answer_case(case, "FB0", financebench.MockBackend(), None, {})
    assert record["status"] == "ok"


def test_task_and_arm_contexts_are_instance_local():
    first = financebench.CaseSearch(_case(identity="task-1"), None, {})
    other_arm = financebench.CaseSearch(_case(identity="task-1"), None, {})
    other_task = financebench.CaseSearch(_case(identity="task-2"), None, {})
    first._deliver([_item()], query="operating income 2022", requested_limit=1)
    assert first.exposed
    assert not other_arm.exposed and not other_task.exposed
    assert other_arm.remaining == other_task.remaining == financebench.RAG.context_bytes
    assert not hasattr(first, "memory") and not hasattr(first, "cache")

