"""Synthetic, isolated invariants for A2-V2. No live provider or market data."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from test_memory import DAY, OUTCOME, episode
from test_rag import SmallEncoder
from tradingagents_lite.memory import Episode, Reflection, _context
from tradingagents_lite.memory_v2.compression import deliver, fidelity, size
from tradingagents_lite.memory_v2.dataset import Case, relevance
from tradingagents_lite.memory_v2.reflection import ProcessNotes, ReflectionAdapter, StructuredReflection, bind, rule_notes
from tradingagents_lite.memory_v2.retrieval import ExperimentStore, rank, eligible
from tradingagents_lite.provider import ModelBackend
from tradingagents_lite.schema import Budget
from tradingagents_lite.token_budget import TokenBudget


def make_record(**overrides):
    ep = episode(**overrides)
    return ep, bind(ep, "Five-session momentum +0.042. Volatility within the risk limit.",
                    rule_notes("positive", ep.risk_veto), generated_at=DAY + timedelta(days=100))


def scope_args(**overrides):
    return {"ticker": "AAPL", "cutoff": DAY, "partition": "development",
            "arm": "M2", **overrides}


def test_point_in_time_maturity_future_return_and_equal_cutoff(tmp_path):
    late = DAY + timedelta(days=2)
    early, _ = make_record()
    pending = make_record(source_task_id="future",
                          outcome=OUTCOME.model_copy(update={"available_at": late}),
                          outcome_available_at=late, reflection_created_at=late, eligible_at=late)
    delayed = make_record(source_task_id="delay", reflection_created_at=late, eligible_at=late)
    with ExperimentStore(tmp_path, SmallEncoder()) as store:
        store.put_many([make_record(), pending, delayed])
        assert not store.scoped(**scope_args(cutoff=DAY - timedelta(seconds=1)))
        visible = store.scoped(**scope_args())
        assert list(visible) == [early.point_id]
        assert all(record[0].outcome.available_at <= DAY for record in visible.values())
        assert len(store.scoped(**scope_args(cutoff=late))) == 3
        assert not store.scoped(**scope_args(cutoff=early.decision_time))


def test_ticker_partition_arm_and_explicit_cross_scope(tmp_path):
    primary = make_record(source_task_id="primary")
    other = make_record(source_task_id="other", ticker="MSFT")
    wrong_arm = make_record(source_task_id="arm", ticker="MSFT", experiment_arm="different")
    wrong_partition = make_record(source_task_id="partition", ticker="MSFT", dataset_partition="synthetic")
    with ExperimentStore(tmp_path, SmallEncoder()) as store:
        store.put_many([primary, other, wrong_arm, wrong_partition])
        assert set(store.scoped(**scope_args())) == {primary[0].point_id}
        assert set(store.scoped(**scope_args(allowed_tickers=["AAPL", "MSFT"]))) == {
            primary[0].point_id, other[0].point_id}


def test_schema_source_and_live_generation_time():
    ep, ref = make_record()
    assert StructuredReflection.model_validate_json(ref.model_dump_json()) == ref
    assert ref.decision_thesis == ep.decision_rationale
    assert ref.outcome_summary == OUTCOME and ref.source_evidence_ids == tuple(ep.observed_evidence_ids)
    assert ref.generated_at > ref.eligible_at == ref.simulated_eligible_at
    live = bind(ep, ref.situation, ref.notes, provenance="model_live",
                generated_at=DAY + timedelta(days=1))
    assert live.simulated_eligible_at is None
    assert not eligible(ep, live, cutoff=DAY, partition="development", arm="M2", tickers={"AAPL"})
    with pytest.raises(ValidationError):
        StructuredReflection.model_validate({**live.model_dump(), "eligible_at": DAY})
    with pytest.raises(ValueError, match="diverge"):
        ref.model_copy(update={"source_evidence_ids": ("invented",)}).validate_episode(ep)


def test_restart_duplicates_legacy_compatibility_and_rebuild(tmp_path):
    original = make_record()
    ep, ref = original
    before = ep.model_dump_json()
    with ExperimentStore(tmp_path, SmallEncoder()) as store:
        store.authority.put(ep)  # Existing Episode schema can receive its sidecar later.
        store.put_many([original])
        store.put_many([original])
        assert store.records() == [original]
    with ExperimentStore(tmp_path, SmallEncoder()) as store:
        assert store.records() == [original]
        idx = store.indexes["E2"]
        idx.client.delete(idx.name, [ep.point_id], wait=True)
        visible = store.scoped(**scope_args())
        with pytest.raises(RuntimeError, match="inconsistent"):
            store.cosine("revenue", "E2", visible, DAY, "development", "M2")
        assert store.consistency()["E2"]["missing"] == [ep.point_id]
        assert not any(items for group in store.rebuild().values() for items in group.values())
        assert Episode.model_validate_json(before) == store.records()[0][0]
        for mode in store.indexes:
            scores = store.cosine("revenue", mode, visible, DAY, "development", "M2")
            assert rank(visible, scores, DAY, "R4") == [ep.point_id]


def test_empty_candidate_state_and_future_label_independence():
    ep, _ = make_record()
    baseline = Case(ep, "query", "positive", False)
    displaced = Case(ep.model_copy(update={"outcome": OUTCOME.model_copy(update={"asset_return": 8})}),
                     "query", "positive", False)
    assert relevance(baseline, baseline) == relevance(displaced, baseline) == 3
    assert relevance(Case(ep, "query", "negative", False), baseline) == 1
    assert rank({}, {}, DAY, "R4") == []
    assert deliver([], "query", 900, "C4")["context_bytes"] == 2


@pytest.mark.parametrize("mode", ["C2", "C3", "C4"])
def test_complete_fields_utf8_boundary_and_original_episode(mode):
    row = make_record(reflection=Reflection(lesson="保护 截止 2024-02-20：收益率 -0.0100", potential_error="test"))
    ep, ref = row
    source = ep.model_dump_json()
    output = deliver([row], "profit risk momentum", 3600, mode)
    mandatory = {k: v for k, v in output["context"][0].items() if k in {
        "memory_id", "historical_evidence_ids", "situation", "action", "lesson", "uncertainty",
        "eligible_at", "decision_time"}}
    boundary = size([mandatory])
    assert not deliver([row], "profit risk", boundary - 1, mode)["context"]
    exact = deliver([row], "profit risk", boundary, mode)
    assert exact["context"] == [mandatory]
    assert exact["context_bytes"] == boundary
    fields = fidelity([row], exact)
    for key in ("lesson", "references", "action", "eligible_date", "decision_date_exact",
                "uncertainty", "numeric_content_exact"):
        assert fields[key] == 1
    assert ep.model_dump_json() == source
    assert deliver([row], "", 3600, "C1")["context"] == [_context(ep, compact=True)]


def test_global_mandatory_first_preserves_recalled_id_order():
    rows = [make_record(source_task_id=f"synthetic-{i}", decision_rationale="x" * 600) for i in range(2)]
    context = deliver(rows, "momentum threshold", 1150, "C4")
    assert context["injected"] == context["selected"]
    assert context["context_bytes"] <= 1150


def test_mock_schema_repair_budget_cache_and_fact_fidelity(tmp_path):
    ep, ref = make_record()
    calls = []
    good = ProcessNotes(potential_error="Possible short-window bias", lesson="Review signal stability",
                        uncertainty="Evidence remains limited", applicability="Short horizon").model_dump_json()

    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id=str(len(calls)), model="glm-5.3",
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"lesson":[]}' if len(calls) == 1 else good, tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=6))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    ledger = TokenBudget(tmp_path / "ledger.sqlite")
    backend = ModelBackend(client, "glm-5.3", Budget(), None, token_budget=ledger, max_retries=0)
    backend.set_budget_context(experiment_id="synthetic-reflection", task_id="synthetic-1", phase="A2")
    storage = tmp_path / "mock-reflections.sqlite"
    cache = ReflectionAdapter(storage, enabled=True)
    actual = cache.generate(backend, ep, ref.situation, replay=True)
    assert actual.provenance == "model_replay"
    assert actual.source_evidence_ids == ref.source_evidence_ids
    assert actual.outcome_summary == ep.outcome and actual.action == ep.action
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert backend.call_log[-1]["structured_repair"]
    assert all(attempt["phase"] == "A2" for attempt in ledger.attempts())
    assert ReflectionAdapter(storage, enabled=True).generate(backend, ep, ref.situation, replay=True) == actual
    assert len(calls) == 2


def test_disabled_mock_and_no_repeat_after_transport_failure(tmp_path):
    ep, ref = make_record()
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("synthetic transport failure")

    mock = SimpleNamespace(phase="A2", max_retries=0, _call=fail)
    path = tmp_path / "failure.sqlite"
    with pytest.raises(ValueError, match="enablement"):
        ReflectionAdapter(path).generate(mock, ep, ref.situation)
    with pytest.raises(RuntimeError, match="transport"):
        ReflectionAdapter(path, enabled=True).generate(mock, ep, ref.situation)
    with pytest.raises(RuntimeError, match="already recorded"):
        ReflectionAdapter(path, enabled=True).generate(mock, ep, ref.situation)
    assert len(calls) == 1 and datetime.now(UTC) > ep.outcome_available_at
