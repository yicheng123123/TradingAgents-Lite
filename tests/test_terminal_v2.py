from datetime import date

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from tradingagents_lite.data import close_available
from tradingagents_lite.terminal import archive
from tradingagents_lite.terminal.research import PRESETS


def test_chart_contains_only_eligible_market_bars():
    task = archive.tasks_for("Legacy")[0]
    if not (archive.market_directory(task) / "manifest.json").is_file():
        pytest.skip("Frozen legacy market required")
    bars, spy, _ = archive.market_series(task)
    assert bars[-1].day == task.decision_date and spy[-1].day == task.decision_date
    assert all(bar.day <= task.decision_date and bar.available_at <= close_available(task.decision_date)
               for bar in (*bars, *spy))
    with pytest.raises(ValueError, match="Snapshot differs"):
        archive.market_series(task, "mismatched-manifest")


def test_memory_filters_future_and_unknown_experience():
    timestamp = "2024-02-08T21:15:00Z"
    episode = {"eligible_at": timestamp, "decision_time": timestamp, "memory_id": "prior",
               "historical_outcome": {"available_at": timestamp}}
    row = {"task_id": "test-01", "symbol": "MSFT", "decision_date": "2024-02-13", "horizon_sessions": 3,
           "memory_trace": {"context": [episode,
               dict(episode, memory_id="future", eligible_at="2024-03-01T21:15:00Z"),
               dict(episode, memory_id="missing", historical_outcome={}),
               dict(episode, memory_id="future-settlement",
                    historical_outcome={"available_at": "2024-02-14T21:15:00Z"})]}}
    assert [item["memory_id"] for item in archive.eligible_episodes(row)] == ["prior"]


def test_preset_and_history_reopen_preserve_archive(monkeypatch, tmp_path):
    from tradingagents_lite import research_ui

    monkeypatch.setenv("TALITE_UI_RUNS", str(tmp_path / "runs"))

    def fail_on_execution(*args, **kwargs):
        pytest.fail("Navigating without executing a model is required")

    monkeypatch.setattr(research_ui, "start_run", fail_on_execution)
    at = AppTest.from_file(str(archive.ROOT / "streamlit_app.py")).run()
    at.selectbox(key="preset").set_value("Adaptive").run()
    for key, value in zip(("rag", "memory", "debate", "transport"), PRESETS["Adaptive"][:4], strict=True):
        assert at.selectbox(key=key).value == value
    at.radio(key="page").set_value("Run History").run()
    assert not at.exception and len(at.dataframe) == 2
    date_input = next(item for item in at.date_input if item.label == "Decision date range")
    date_input.set_value((date(2024, 2, 13), date(2024, 2, 13))).run()
    assert not at.exception and set(at.dataframe[0].value["Decision date"]) == {"2024-02-13"}
    at.button(key="open_run").click().run()
    assert not at.exception and at.radio(key="page").value == "Research"
    assert at.selectbox(key="preset").value == "Adaptive"
    identity = at.query_params["run"]
    assert archive.load(identity) and not (tmp_path / "runs").exists()
    assert archive.load("../run") is None
