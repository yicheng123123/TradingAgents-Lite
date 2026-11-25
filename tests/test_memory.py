from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from test_rag import SmallEncoder
from test_stage1 import TASK, bars
from tradingagents_lite.data import MarketView, close_available
from tradingagents_lite.memory import Episode, MemoryConfig, MemoryReplay, MemoryStore, Reflection, model_reflection
from tradingagents_lite.provider import ModelBackend
from tradingagents_lite.schema import Action, Budget, Decision, Outcome, Proposal, Task
from tradingagents_lite.token_budget import TokenBudget
from tradingagents_lite.workflow import MomentumBackend, build_workflow


DAY = datetime(2024, 2, 20, 22, tzinfo=UTC)
OUTCOME = Outcome(entry_day=DAY.date(), exit_day=DAY.date(), available_at=DAY,
                  asset_return=-.01, benchmark_return=.02)


def episode(**overrides):
    return Episode.model_validate({
        "source_task_id": "case", "ticker": "AAPL", "decision_time": DAY - timedelta(days=2),
        "outcome_available_at": DAY, "reflection_created_at": DAY, "eligible_at": DAY,
        "action": Action.BUY, "observed_evidence_ids": ["evidence:1"],
        "outcome": OUTCOME, "reflection": Reflection(potential_error="risk", lesson="revenue item"),
        "dataset_partition": "development", "experiment_arm": "M2", **overrides})


def recall(store, cutoff=DAY, **kwargs):
    return store.recall("revenue", **{"ticker": "AAPL", "cutoff": cutoff,
        "partition": "development", "arm": "M2", "mode": "M2", **kwargs})


def test_future_reflection_and_scopes_are_gated(tmp_path):
    with MemoryStore(tmp_path, SmallEncoder()) as store:
        delayed = episode(reflection_created_at=DAY+timedelta(days=1), eligible_at=DAY+timedelta(days=1))
        store.put(delayed)
        assert not recall(store, DAY - timedelta(seconds=1))["selected"]
        assert not recall(store, DAY)["selected"]
        assert recall(store, DAY+timedelta(days=1))["injected_ids"] == [delayed.memory_id]
        assert not recall(store, DAY+timedelta(days=1), arm="M3")["selected"]
        assert not recall(store, DAY+timedelta(days=1), partition="holdout")["selected"]
        assert not recall(store, DAY+timedelta(days=1), ticker="MSFT")["selected"]
        assert not recall(store, DAY+timedelta(days=1), mode="M0")["selected"]


def test_sqlite_versions_reopen_and_repair(tmp_path):
    first = episode()
    late = DAY + timedelta(days=2)
    revised = episode(reflection_created_at=late, eligible_at=late,
                      reflection=Reflection(potential_error="test", lesson="new revenue lesson"))
    with MemoryStore(tmp_path, SmallEncoder()) as store:
        assert store.put(first)
        assert not store.put(first)
        assert store.put(revised)
    with MemoryStore(tmp_path, SmallEncoder()) as store:
        assert recall(store)["selected"][0]["version_hash"] == first.version_hash
        assert recall(store, late)["selected"][0]["version_hash"] == revised.version_hash
        store.vectors.client.delete(store.vectors.name, [revised.point_id])
        assert store.consistency()["missing"] == [revised.point_id]
        with pytest.raises(RuntimeError, match="inconsistent"):
            recall(store, late)
        assert not any(store.rebuild().values())
        assert recall(store, late)["injected_ids"] == [first.memory_id]


def test_same_session_outcome_cannot_enter_next_same_session_task(tmp_path):
    day = (DAY - timedelta(days=2)).date()
    tasks = [Task(task_id=f"dev-0{i}", symbol="AAPL", decision_date=day, horizon_sessions=1)
             for i in (1, 2)]
    decision = Decision(action=Action.BUY, rationale="test", evidence_ids=["evidence:1"])
    with MemoryStore(tmp_path, SmallEncoder()) as store:
        replay = MemoryReplay(store, MemoryConfig("M2", "development", "M2"))
        assert replay.before(tasks[0], "revenue")["eligible"] == 0
        replay.enqueue(tasks[0], decision, OUTCOME)
        assert replay.before(tasks[1], "revenue")["eligible"] == 0
        assert replay.written == 0
        replay.advance(DAY)
        assert replay.written == 1
        with pytest.raises(ValueError, match="increasing"):
            replay.advance(close_available(day))


def test_compaction_and_byte_limit_preserve_historical_ids(tmp_path):
    with MemoryStore(tmp_path, SmallEncoder()) as store:
        first = episode(reflection=Reflection(potential_error="x"*300, lesson="revenue "*50))
        store.put(first)
        result = recall(store, mode="M3", context_bytes=850)
        assert result["before_bytes"] > result["context_bytes"]
        assert result["context"][0]["historical_evidence_ids"] == first.observed_evidence_ids
        assert not recall(store, mode="M3", context_bytes=100)["context"]


def test_trader_receives_experience_without_admitting_historical_citation():
    class TraderBackend(MomentumBackend):
        def trader_with_memory(self, report, evidence, memory):
            assert "historical-only" not in evidence
            assert memory[0]["historical_evidence_ids"] == ["historical-only"]
            return Proposal(action=Action.BUY, rationale="invalid use", evidence_ids=["historical-only"])

    observation = {"context": [{"historical_evidence_ids": ["historical-only"]}]}
    memory = SimpleNamespace(before=lambda *args: observation)
    backend = TraderBackend(Budget())
    result = build_workflow(MarketView(TASK, bars()), backend, Budget(), memory=memory).invoke({})
    assert result["memory_trace"] == {**observation, "passed_to_backend": True}
    assert result["errors"][-1]["code"] == "invalid_citation"


@pytest.mark.parametrize("valid", [True, False])
def test_mock_llm_reflection_uses_A2_budget_and_validates_output(tmp_path, valid):
    content = '{"potential_error":"risk","lesson":"lesson"}' if valid else '{"lesson":[]}'
    reply = SimpleNamespace(id="test", model="glm-5.3",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=6))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: reply)))
    budget = TokenBudget(tmp_path / "budget.sqlite")
    backend = ModelBackend(client, "glm-5.3", Budget(), None, token_budget=budget)
    backend.set_budget_context(experiment_id="mock", task_id="dev-01", phase="A2")
    decision = Decision(action=Action.BUY, rationale="test", evidence_ids=["evidence:1"])
    if valid:
        assert model_reflection(backend, decision, OUTCOME).lesson == "lesson"
    else:
        with pytest.raises(ValueError):
            model_reflection(backend, decision, OUTCOME)
    assert budget.attempts()[0]["phase"] == "A2"
    assert budget.attempts()[0]["charged_tokens"] == 26
