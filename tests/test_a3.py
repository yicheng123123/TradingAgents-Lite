from datetime import timedelta
import json
from pathlib import Path

import pytest

from tradingagents_lite.agents import A3Config
from tradingagents_lite.a3_benchmark import equivalence, scenario
from tradingagents_lite.data import HistoricalData, close_available, read_tasks
from tradingagents_lite.mcp_finance import MCPTransport
from tradingagents_lite.rag import Retriever
from tradingagents_lite.verifier import verify_citations, verify_proposal

from test_rag import SmallEncoder, document
from tradingagents_lite.documents import chunk_document, parse_document
from tradingagents_lite.vectors import VectorStore


MARKET = Path("data/yahoo-2024-v1")
TASKS = Path("tasks/development.jsonl")


@pytest.fixture(scope="module")
def frozen():
    return HistoricalData(MARKET), read_tasks(TASKS)[0]


def test_mcp_stdio_handshake_schema_and_shutdown(frozen, monkeypatch):
    from mcp.client.stdio import get_default_environment

    _, task = frozen
    monkeypatch.setenv("TALITE_API_KEY", "test-placeholder")
    assert "TALITE_API_KEY" not in get_default_environment()
    with MCPTransport({"task": task.model_dump(mode="json"), "market": str(MARKET.resolve())}) as bridge:
        tools = bridge.start()
        assert set(tools) == {"market_indicators", "market_bars"}
        assert tools["market_bars"]["additionalProperties"] is False
        assert bridge.call("market_indicators", "{}") == frozen[0].view(task).indicators()
        for argument in ('{"ticker":"OTHER"}', '{"cutoff":"2099-01-01"}', '{"dataset_partition":"holdout"}'):
            assert bridge.call("market_bars", argument)["error"] == "invalid_arguments"
        assert bridge.call("shell", "{}")["error"] == "unknown_tool"
    assert bridge.session is None and bridge.portal is None


def test_real_four_finance_tools_are_equivalent(frozen):
    index = Path("data/a1-development-001")
    if not index.is_dir():
        pytest.skip("A1 local model and index required for this integration case")
    items = equivalence([frozen[1]], MARKET, Path("data/financial-evidence-dev-003"), index)
    assert len(items) == 4 and all(row["equal"] and row["mcp_success"] for row in items)


@pytest.mark.parametrize("fault,code", [("mcp_timeout", "mcp_"), ("mcp_exit", "mcp_")])
def test_real_mcp_failure_can_reopen_scoped_child(tmp_path, frozen, fault, code):
    market, task = frozen
    row = scenario(market, task, tmp_path, "mcp", A3Config(transport="mcp", verifier=True, repair=True),
                   fault=fault, data_path=MARKET)
    assert row["completed"] and row["recovered"]
    assert any(item.startswith(code) for item in row["errors"])


@pytest.mark.parametrize("fault", ["malformed_json", "unknown_tool", "bad_arguments", "invalid_citation", "rag_empty"])
def test_bounded_repair_requires_actual_success(tmp_path, frozen, fault):
    market, task = frozen
    f0 = scenario(market, task, tmp_path, "f0", A3Config(verifier=True), fault=fault)
    f1 = scenario(market, task, tmp_path, "f1", A3Config(verifier=True, repair=True), fault=fault)
    assert not f0["completed"] and not f0["recovered"]
    assert f1["completed"] and f1["recovered"]
    assert f1["verification"]["passed"]
    assert f1["verification"]["citation_entailment"] == "not_assessed"


def test_future_and_step_limit_are_hard_failures_with_trace(tmp_path, frozen):
    market, task = frozen
    for fault, code in [("future_evidence", "future_evidence"), ("step_limit", "step_limit")]:
        row = scenario(market, task, tmp_path, fault, A3Config(
            verifier=True, repair=True, max_steps=3 if fault == "step_limit" else 24), fault=fault)
        assert row["decision"] == "hold" and not row["completed"] and not row["recovered"]
        assert code in row["errors"] and row["events"]


def test_memory_degrades_and_budget_fails_before_transport(tmp_path, frozen):
    market, task = frozen
    mem = scenario(market, task, tmp_path, "memory", A3Config(verifier=True, repair=True),
                   fault="memory_inconsistent")
    shut = scenario(market, task, tmp_path, "budget", A3Config(verifier=True, repair=True),
                    fault="budget_denied")
    assert mem["completed"] and mem["degraded"] and not mem["recovered"]
    assert "token_budget_exhausted" in shut["errors"] and not shut["completed"]
    assert shut["real_glm_tokens"] == 0


def test_optional_debate_roles_and_mock_usage_are_counted(tmp_path, frozen):
    market, task = frozen
    d0 = scenario(market, task, tmp_path, "d0", A3Config(), rich=True)
    d1 = scenario(market, task, tmp_path, "d1", A3Config(debate="D1"), rich=True)
    d2 = scenario(market, task, tmp_path, "d2", A3Config(debate="D2", supplement=True), rich=True)
    mock = scenario(market, task, tmp_path, "mock", A3Config(verifier=True), mock=True)
    assert d0["role_calls"] == 0 and d1["role_calls"] == 2
    assert d2["role_calls"] in {0, 4}
    assert mock["completed"] and mock["mock_tokens"] == 70 and mock["real_glm_tokens"] == 0


def test_verifier_scope_context_timing_and_historical_ids(frozen):
    market, task = frozen
    current = market.view(task).indicators()
    key = current["evidence_id"]
    scope = {key: current}
    assert verify_citations([key], scope, {key}, task)["passed"]
    assert verify_citations([key], scope, set(), task)["error_code"] == "citation_not_in_context"
    assert verify_citations(["old"], scope, {key}, task, historical_ids={"old"})["error_code"] == "historical_memory_citation"
    future = {"evidence_id": "chunk:1", "ticker": task.symbol, "eligible_at": (
        close_available(task.decision_date) + timedelta(days=1)).isoformat(), "pit_level": "historical_proxy",
        "version_id": "a"*64, "raw_hash": "b"*64, "dataset_partition": "development"}
    check = verify_citations(["chunk:1"], {"chunk:1": future}, {"chunk:1"}, task)
    assert not check["repairable"] and check["error_code"] == "future_evidence"
    future["dataset_partition"] = "holdout"
    assert "evidence_scope" in verify_citations(["chunk:1"], {"chunk:1": future}, {"chunk:1"}, task)["violations"]
    assert verify_proposal({"action":"short"}, scope, {key}, task)["error_code"] == "invalid_proposal_schema"
    risk = verify_proposal({"action":"buy","rationale":"test","evidence_ids":[key]}, scope,
                          {key}, task, volatility=.05, volatility_limit=.04)
    assert risk["passed"] and risk["suggested_action"] == "hold"


def test_rag_candidate_cache_is_invalidated_on_upsert(tmp_path):
    from test_stage1 import TASK

    lookup = {"ticker": TASK.symbol, "cutoff": close_available(TASK.decision_date),
              "partition": "development", "mode": "R1", "top_k": 4, "candidate_k": 4}
    with VectorStore(tmp_path / "cache", SmallEncoder(), "chunks") as store:
        retriever = Retriever(store)
        assert retriever.retrieve("revenue", **lookup)["items"] == []
        raw = b"<p>revenue</p>"
        doc = document(raw)
        store.put(chunk_document(doc, parse_document(doc, raw)), id_key="chunk_id", text_key="text")
        assert len(retriever.retrieve("revenue", **lookup)["items"]) == 1
