from datetime import UTC, date, datetime, timedelta
import json
from types import SimpleNamespace

import pytest

from tradingagents_lite.anonymization import Projection, model_context
from tradingagents_lite.data import HistoricalData, close_available
from tradingagents_lite.data_v3 import _partition, audit_pit
from tradingagents_lite.schema import Task
from tradingagents_lite.temporal_controls import evaluate_synthetic, fingerprint, synthetic_cases
from tradingagents_lite.temporal_probes import question, score, summarize


DAY = date(2026, 5, 4)


def projection():
    return Projection(secret=b"offline-test-secret-only", case="case1", ticker="ACME", decision=DAY, close=200)


def test_projection_preserves_scale_citations_and_filters_all_prose_and_future():
    projector = projection()
    stamp = (close_available(DAY) - timedelta(days=1)).isoformat()
    raw = {"evidence_id": "ACME-real-id", "eligible_at": stamp, "source_type": "sec_xbrl",
           "ticker": "ACME", "company": "ACME Corp", "excerpt": "ACME Corp", "title": "ACME Corp",
           "url": "https://private/ACME", "source_path": "/ACME", "period_end": "2026-03-31",
           "metric": "Revenue", "value": 200, "unit": "USD"}
    future = {**raw, "eligible_at": (close_available(DAY) + timedelta(minutes=1)).isoformat()}
    error = {"error": "secret ACME failure", "path": "/secret"}
    ctx = model_context("P3", ticker="ACME", company="ACME Corp", decision=DAY,
        market={"bars": [{"day": str(DAY), "close": 200, "open": 190, "volume": 2000}],
                "ticker": "ACME"}, external=[raw, future], rag=[raw], memory=[raw],
        agents=[{"role": "bull", "evidence_ids": ["ACME-real-id"], "content": "ACME Corp"}],
        projection=projector)
    text = json.dumps(ctx)
    assert "ACME" not in text and "2026-" not in text and "private" not in text
    evidence, = ctx["evidence"]
    assert projector.resolve_citations([evidence["evidence_id"]]) == ["ACME-real-id"]
    assert ctx["agents"][0]["evidence_ids"] == [evidence["evidence_id"]]
    assert ctx["rag"] == ctx["memory"] == ctx["evidence"]
    assert evidence["eligible_at_offset"] == -1 and evidence["period_end_offset"] < 0
    market = ctx["market"]["bars"][0]
    assert market["close"] == 100 and market["open"] / market["close"] == .95
    projected_cost = projector.record({**raw, "metric": "Net Income", "value": 50})["value"]
    assert projected_cost / evidence["value"] == pytest.approx(.25)
    assert projector.tool_result("market_bars", error) == {"status": "unavailable"}
    with pytest.raises(ValueError, match="Unknown evidence alias"):
        projector.resolve_citations(["invented"])


@pytest.mark.parametrize("profile", ["P0", "P1", "P2", "P3"])
def test_profiles_gate_identity_and_evidence(profile):
    raw = {"evidence_id": "known", "eligible_at": close_available(DAY).isoformat(), "source_type": "news"}
    context = model_context(profile, ticker="ACME", company="ACME", decision=DAY, market={},
                            external=[raw], rag=[raw], memory=[raw], projection=projection())
    assert ("ticker" in context) == (profile in {"P0", "P1"})
    assert ("evidence" in context) == (profile in {"P1", "P3"})


def test_tools_do_not_accept_identity_paths_or_arbitrary_rag_queries():
    p = projection()
    assert p.tool_arguments("rag_search", {"query": "risks", "limit": 2}) == {"query": "risks", "limit": 2}
    for args in ({"query": "ACME"}, {"query": "risks", "ticker": "ACME"}, {"query": "risks", "limit": 18}):
        with pytest.raises(ValueError):
            p.tool_arguments("rag_search", args)
    assert fingerprint([100, 102, 99]) == fingerprint([200, 204, 198])


def test_scores_abstentions_and_exact_numerical_and_mc_as_separate_metrics():
    item = {"probe_id": "test", "stratum": 3, "mode": "open", "feature": "return",
            "answer": -2.2, "tolerance": .5, "target": "alpha"}
    approximate = score(item, "-2.0")
    assert approximate["numerical_accuracy"] and not approximate["exact"] and approximate["directional"]
    assert approximate["alpha_pair"] == [-2.0, -2.2]
    failed = score(item, "UNKNOWN")
    assert failed["unknown"] and not failed["correct"] and not failed["directional"]
    mc = {**item, "mode": "choice", "letter": "B", "options": [1, -2.2, 3, 4],
          "question": "What was the return? Return a signed number."}
    assert score(mc, "B")["exact"] and not score(mc, "B because...")["correct"]
    assert "signed number" not in question(mc) and "B. -2.2" in question(mc)
    assert summarize([approximate, failed, score(mc, "B")])["overall"]["multiple_choice"]["hits"] == 1


def test_boundary_requires_sustained_drop_and_bootstrap_stability():
    rows = []
    for stratum in range(1, 7):
        for i in range(10):
            rows.append({"stratum": stratum, "mode": "open" if i < 5 else "choice", "correct": stratum < 3})
    assert summarize(rows)["boundary"]["identified"]
    assert not summarize(rows[:-1])["boundary"]["identified"]


def test_sampling_never_uses_future_or_incomplete_window_and_holdout_is_sealed(monkeypatch):
    monkeypatch.setattr("tradingagents_lite.data_v3.ISSUERS", {"ACME": ("sector", "ETF", 1)})
    days = [DAY + timedelta(days=i) for i in range(51)]
    bars = [SimpleNamespace(day=day, volume=1) for day in days]
    market = HistoricalData.__new__(HistoricalData)
    market.manifest = {"version": "yahoo-us-v3-001"}
    market.series = {"ACME": bars, "ETF": bars, "SPY": bars}
    used = {"ACME": [21]}
    tasks, _, _ = _partition(market, days, "holdout", days[21], days[-1], 3, used)
    steps = [days.index(task.decision_date) for task in tasks]
    assert all(26 <= index <= 45 for index in steps)
    assert all(j - i >= 5 for i, j in zip(steps, steps[1:]))
    assert all(market.is_sealed(task) for task in tasks)


def test_pit_cuts_off_amendments_and_financial_periods(tmp_path):
    task = Task(task_id="dev", symbol="ACME", decision_date=DAY, horizon_sessions=5)
    path = tmp_path / "fundamentals"
    path.mkdir()
    timestamp = (close_available(DAY) - timedelta(hours=1)).isoformat()
    filing = {"ticker": "ACME", "eligible_at": timestamp, "acceptance_datetime": timestamp,
              "filing_date": str(DAY), "report_date": "2026-03-31", "form": "10-Q",
              "pit_level": "historical_proxy"}
    fact = {"accession": "original", "metric": "Total Assets", "tag": "Assets", "unit": "USD",
            "start": None, "end": "2026-03-31", "value": 10}
    (path / "ACME.json").write_text(json.dumps({
        "facts": [fact, {**fact, "accession": "revised", "value": 60}],
        "filings": {"original": filing, "revised": {**filing, "form": "10-Q/A",
                     "eligible_at": datetime(2026, 5, 5, tzinfo=UTC).isoformat()}},
    }))
    result = audit_pit(tmp_path, [task])
    assert result["eligible_fact_instances_checked"] == 1 and result["future_facts"] == 0


def test_synthetic_controls_change_outcomes_and_evidence_plans():
    rows = {case["case"]: evaluate_synthetic(case) for case in synthetic_cases()}
    assert [rows[key]["action"] for key in ("base", "changed_numbers", "changed_event", "future_evidence")] == [
        "buy", "sell", "hold", "hold"]
    assert rows["base"]["margin"] == .25 and rows["changed_numbers"]["margin"] < 0
    assert all(item["verifier_passed"] for item in rows.values())
    assert rows["future_evidence"]["future_rejected"]
    assert rows["changed_numbers"]["steps"] != rows["base"]["steps"]
