import hashlib
import json
from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from tradingagents_lite.research_ui import ROOT, archive_for, load_run, token_dashboard
from tradingagents_lite.token_budget import TokenBudget


def app(monkeypatch, tmp_path):
    monkeypatch.delenv("TALITE_API_KEY", raising=False)
    monkeypatch.setenv("TALITE_UI_RUNS", str(tmp_path / "runs"))
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=35).run()


def test_offline_start_and_explicit_llm_gate(monkeypatch, tmp_path):
    at = app(monkeypatch, tmp_path)
    assert not at.exception
    assert at.selectbox(key="backend").value == "Offline"
    assert not (tmp_path / "runs").exists()
    at.selectbox(key="backend").select("LLM").run()
    assert at.button(key="execute").disabled
    assert any("TALITE_API_KEY" in warning.value for warning in at.warning)
    assert not (tmp_path / "runs").exists()


def test_authorized_mock_request_runs_once_across_reruns(monkeypatch, tmp_path):
    from tradingagents_lite import research_ui

    at = app(monkeypatch, tmp_path)
    monkeypatch.setenv("TALITE_API_KEY", "test-placeholder")
    calls = []
    monkeypatch.setattr(research_ui, "start_run", lambda task, **options: calls.append(options))
    at.selectbox(key="backend").select("LLM").run()
    assert at.button(key="execute").disabled
    at.checkbox(key="approve_paid").check().run()
    at.button(key="execute").click().run()
    at.run()
    assert len(calls) == 1 and calls[0]["backend"] == "llm"


def test_real_frozen_task_trace_and_no_rerun_execution(monkeypatch, tmp_path):
    at = app(monkeypatch, tmp_path)
    at.checkbox(key="verifier").check()
    at.selectbox(key="debate").select("D1")
    at.button(key="execute").click().run()
    assert not at.exception and not at.error
    identity = at.session_state["run_id"]
    row = load_run(identity)
    assert row["status"] == "ok" and row["role_calls"] == 2
    at.radio(key="page").set_value("Agent Trace").run()
    assert not at.exception
    assert len(next(widget for widget in at.selectbox if widget.label == "Inspect event").options) == len(row["agent_trace"])
    assert any(metric.label == "Measured model tokens" and metric.value == "0" for metric in at.metric)
    at.radio(key="page").set_value("Research").run()
    at.run()
    assert list((tmp_path / "runs").iterdir()) == [tmp_path / "runs" / identity]
    refreshed = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=35)
    refreshed.query_params["run"] = identity
    refreshed.run()
    assert not refreshed.exception
    assert any(row["task_id"] in item.value for item in refreshed.markdown)
    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_rag_memory_and_real_mcp_are_visible(monkeypatch, tmp_path):
    if not (ROOT / "data/a1-development-001/chunks-encoder.json").is_file():
        pytest.skip("Local frozen RAG index required")
    at = app(monkeypatch, tmp_path)
    at.selectbox(key="task").select(at.selectbox(key="task").options[-1])
    at.selectbox(key="rag").select("R1")
    at.selectbox(key="memory").select("M2")
    at.selectbox(key="transport").select("MCP")
    at.checkbox(key="verifier").check()
    at.button(key="execute").click().run()
    assert not at.exception and not at.error
    identity = at.session_state["run_id"]
    row = load_run(identity)
    assert row["status"] == "ok"
    assert any(e["name"] == "rag_search" and e["status"] == "ok"
               and e["retrieval_trace"]["eligible_chunks"] for e in row["tool_trace"])
    assert row["memory_trace"] is not None
    metadata = json.loads((tmp_path / "runs" / identity / "run/run.json").read_text())
    assert metadata["config"]["a3"]["transport"] == "mcp"
    at.radio(key="page").set_value("Evidence").run()
    assert not at.exception and any("BM25" in item.value for item in at.markdown)
    at.radio(key="page").set_value("Memory").run()
    assert not at.exception and any("Context compaction" in item.value for item in at.markdown)


def test_archive_exact_metrics_and_pending_status(monkeypatch, tmp_path):
    at = app(monkeypatch, tmp_path)
    at.radio(key="page").set_value("Benchmark").run()
    at.selectbox(key="archive").select("A1").run()
    assert any("Pending" in item.value for item in at.info)
    rows = at.dataframe[0].value
    metric = rows.loc[rows["Mode"] == "R2", "recall@10"].iloc[0]
    assert metric == archive_for("A1")["metrics"]["test"]["R2"]["page_proxy"]["recall@10"]


def test_token_dashboard_is_read_only(monkeypatch, tmp_path):
    from tradingagents_lite import zhipu_account

    path = tmp_path / "ledger.sqlite"
    monkeypatch.setenv("TALITE_TOKEN_BUDGET_DB", str(path))
    assert token_dashboard()["remaining"] == 15_000_000 and not path.exists()
    ledger = TokenBudget(path)
    attempt, _ = ledger.reserve(experiment_id="fixture", task_id="task", phase="A3", model="glm-5.3",
        logical_call=1, attempt_number=1, request_sha256="a"*64, request_bytes=150, max_output_tokens=64)
    ledger.settle_usage(attempt, input_tokens=20, output_tokens=10, reasoning_tokens=4, cached_tokens=3)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    at = app(monkeypatch, tmp_path)
    calls = []

    def handler(request):
        calls.append(request.url.path)
        body = (
            {"success": True, "data": {"availableBalance": 0, "giveAmount": "1.25"}}
            if request.url.path == zhipu_account.BALANCE_PATH else {"code": 200, "rows": [], "total": 0}
        )
        return httpx.Response(200, json=body)

    snapshot = zhipu_account.account_snapshot

    def use_mock(previous, *, refresh):
        return snapshot(previous, refresh=refresh, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(zhipu_account, "account_snapshot", use_mock)
    at.radio(key="page").set_value("Usage").run()
    assert not at.exception and not at.error
    assert any("Settled" in item.value and "30 Tokens" in item.value for item in at.markdown)
    assert any("15.00M" in item.value and "15,000,000 Tokens" in item.value for item in at.markdown)
    assert calls == []
    at.button(key="refresh_bigmodel").click().run()
    assert any("TALITE_API_KEY is missing" in caption.value for caption in at.caption)
    assert calls == []
    monkeypatch.setenv("TALITE_API_KEY", "mock-account-key")
    at.button(key="refresh_bigmodel").click().run()
    at.run()
    assert not at.exception and not at.error
    assert calls == [zhipu_account.BALANCE_PATH, zhipu_account.PACKAGES_PATH]
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Available balance"] == "¥0.00"
    assert metrics["Total spend"] == "Unavailable"
    assert metrics["Remaining Tokens"] == "0"
    assert any("Settled" in item.value and "30 Tokens" in item.value for item in at.markdown)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_execution_error_is_visible_without_echoing_private_path(monkeypatch, tmp_path):
    at = app(monkeypatch, tmp_path)
    collision = tmp_path / "collision"
    collision.write_text("fixture")
    monkeypatch.setenv("TALITE_UI_RUNS", str(collision))
    at.button(key="execute").click().run()
    assert not at.exception
    assert any("Local input unavailable" in error.value for error in at.error)
    assert str(collision) not in " ".join(error.value for error in at.error)


def test_development_v2_only_offers_development_and_runs_real_mcp(monkeypatch, tmp_path):
    from tradingagents_lite.research_ui import V2_DATA

    at = app(monkeypatch, tmp_path)
    assert at.selectbox(key="task_set").options == ["Legacy", "Development V2"]
    at.selectbox(key="task_set").select("Development V2").run()
    options = at.selectbox(key="task").options
    assert len(options) == 80 and all("v2-development-" in option for option in options)
    if not (V2_DATA / "evidence/source_manifest.json").is_file():
        pytest.skip("Local V2 snapshot required to run the integrated research task")
    from tradingagents_lite.data import sha256

    fundamentals = V2_DATA / "sec/fundamentals"
    manifest_path = fundamentals / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Local SEC fundamentals manifest required")
    issuers = json.loads(manifest_path.read_text())["issuers"]
    valid = [ticker for ticker, record in issuers.items() if (fundamentals / f"{ticker}.json").is_file()
             and sha256(fundamentals / f"{ticker}.json") == record["sha256"]]
    if not valid:
        pytest.skip("Local SEC bundles differ from their frozen manifest")
    at.selectbox(key="task").select(next(option for option in options if option.split(" · ")[0] in valid))
    at.selectbox(key="transport").select("MCP")
    at.checkbox(key="verifier").check()
    at.button(key="execute").click().run()
    assert not at.error and not at.exception
    run = load_run(at.session_state["run_id"])
    assert run["status"] == "ok" and run["verification"]["passed"]
    assert any(event["tool_name"] == "xbrl_fundamentals" for event in run["agent_trace"])
