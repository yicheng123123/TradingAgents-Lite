from datetime import date, timedelta
import json
import math

import pytest

from tradingagents_lite.data import DataError, HistoricalData, MarketView, close_available, read_tasks
from tradingagents_lite.runner import summary, wilson
from tradingagents_lite.schema import Bar, Budget, Task
from tradingagents_lite.workflow import MomentumBackend, build_workflow, call_message


def bars(first=date(2024, 1, 1), bump=0):
    return tuple(Bar(day=day, available_at=close_available(day), open=value, close=value,
                     low=value-1, high=value+1, volume=1000)
                 for i in range(25)
                 for day in [first + timedelta(days=i)]
                 for value in [100+i+(bump if i>20 else 0)])


TASK = Task(task_id="unit-1", symbol="TEST", decision_date=date(2024, 1, 21), horizon_sessions=3)


def test_future_changes_are_invisible_to_graph():
    budget = Budget()
    a = MarketView(TASK, bars())
    b = MarketView(TASK, bars(bump=900))
    states = [build_workflow(v, MomentumBackend(budget), budget).invoke({}) for v in [a,b]]
    assert states[0] == states[1]
    assert states[0]["decision"]["action"] == "buy"
    assert len(states[0]["tool_trace"]) == 1
    assert a.get_bars(20)["bars"][-1]["day"] == "2024-01-21"


def test_unavailable_same_day_price_fails_closed():
    source = list(bars())
    day = source[20]
    source[20] = day.model_copy(update={"available_at": day.available_at + timedelta(minutes=1)})
    with pytest.raises(DataError, match="exact decision-date"):
        MarketView(TASK, tuple(source))


def test_outcome_uses_strictly_future_open_and_aligned_exit():
    market = HistoricalData.__new__(HistoricalData)
    market.manifest = {"benchmark": "SPY"}
    market.series = {"TEST": bars(bump=50), "SPY": bars()}
    result = market.outcome(TASK)
    assert result.entry_day == date(2024, 1, 22)
    assert result.exit_day == date(2024, 1, 24)
    assert result.asset_return == pytest.approx(173/171-1)
    assert result.benchmark_return == pytest.approx(123/121-1)
    market.series["SPY"] = market.series["SPY"][:21] + market.series["SPY"][22:]
    with pytest.raises(DataError, match="calendar"):
        market.outcome(TASK)


def test_tool_error_recovers_and_prefetch_has_same_decision():
    class RecoveryBackend(MomentumBackend):
        def analyst(self, messages, tools):
            if not any(m["role"] == "tool" for m in messages):
                return call_message("not_registered")
            return super().analyst(messages, tools)

    budget = Budget()
    view = MarketView(TASK, bars())
    recovered = build_workflow(view, RecoveryBackend(budget), budget).invoke({})
    fixed = build_workflow(view, MomentumBackend(budget), budget, react=False).invoke({})
    assert [t["status"] for t in recovered["tool_trace"]] == ["error", "ok"]
    assert recovered["decision"] == fixed["decision"]
    assert recovered["errors"] == []


def test_loop_exhaustion_returns_hold_without_unbounded_tools():
    class AlwaysCalls(MomentumBackend):
        def analyst(self, messages, tools):
            return call_message("market_indicators")
    budget = Budget(max_tool_rounds=1)
    state = build_workflow(MarketView(TASK, bars()), AlwaysCalls(budget), budget).invoke({})
    assert len(state["tool_trace"]) == 1
    assert state["decision"]["action"] == "hold"
    assert state["errors"] == [{"stage": "analyst", "code": "tool_budget"}]


def test_unknown_evidence_rejects_model_report():
    class Fabrication(MomentumBackend):
        def analyst(self, messages, tools):
            return {"content": json.dumps({"view": "claim", "evidence_ids": ["imaginary"]})}
    budget = Budget()
    state = build_workflow(MarketView(TASK, bars()), Fabrication(budget), budget).invoke({})
    assert state["decision"]["risk_veto"] == "workflow_failure"
    assert state["errors"][0]["stage"] == "analyst"


def test_risk_gate_overrides_a_real_directional_proposal():
    budget = Budget(volatility_limit=.00001)
    state = build_workflow(MarketView(TASK, bars()), MomentumBackend(budget), budget).invoke({})
    assert state["proposal"]["action"] == "buy"
    assert state["decision"]["risk_veto"] == "volatility_limit"
    assert state["decision"]["action"] == "hold"


def test_task_cells_must_be_unique(tmp_path):
    p = tmp_path / "tasks.jsonl"
    p.write_text(TASK.model_dump_json() + "\n" + TASK.model_copy(update={"task_id": "again"}).model_dump_json())
    with pytest.raises(DataError, match="unique"):
        read_tasks(p)


def test_summary_excludes_hold_and_failures_and_reports_zero_denominators():
    def row(day, action, alpha, status, hit=None):
        return dict(decision_date=day, direction=action, alpha=alpha, status=status,
                    directional_hit=hit, tool_trace=[], latency_seconds=.1, usage=None)
    sample = [row("one",1,.02,"ok",True),row("two",-1,-.01,"ok",False),
              row("three",0,0,"ok"),row("four",0,None,"workflow_error")]
    metrics = summary(sample,42)
    assert metrics["directional_hit_rate"] == .5
    assert metrics["directional_coverage"] == .5
    assert metrics["mean_alpha"] == pytest.approx(.01/3)
    assert metrics["delta_vs_hold"] == metrics["mean_alpha"]
    assert metrics["tool_recovery_rate"] is None
    assert metrics["alpha_date_bootstrap_95"] == summary(sample,42)["alpha_date_bootstrap_95"]
    assert wilson(0,0) is None
