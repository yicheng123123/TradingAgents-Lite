"""Historical episodes made available to a point-in-time research run."""
import streamlit as st

from ..data import close_available
from .archive import eligible_episodes, task_of
from .charts import bars
from .design import empty, heading, kv, number, percent, section, timeline


def render(row):
    heading("Memory", "Historical experience eligible at the selected decision cutoff")
    if not row:
        empty("No run loaded", "Open a research run to inspect historical experience.")
        return
    memory = row.get("memory_trace") or {}
    episodes = eligible_episodes(row)
    historical, compaction = st.columns([1.75, 1])
    with historical, st.container(border=True):
        section("Historical experience", f"{len(episodes)} in context")
        st.caption(f"Decision cutoff · {close_available(task_of(row).decision_date).isoformat()}")
        if memory.get("error_type"):
            st.warning(memory.get("recovery_status") or "Memory unavailable")
        if not episodes:
            empty("No eligible experience in context", "The selected run has no verifiable settled episodes in its memory context.")
        else:
            timeline((episode.get("source_task_id", episode["memory_id"]), episode["eligible_at"]) for episode in episodes)
            for episode in episodes:
                outcome = episode.get("historical_outcome") or {}
                with st.expander(f'{episode.get("source_task_id", episode["memory_id"])} · {episode.get("action", "").upper()}',
                                 expanded=True):
                    st.write(episode.get("lesson") or "Reflection unavailable.")
                    if episode.get("potential_error"):
                        st.caption(episode["potential_error"])
                    kv([("Original decision time", episode.get("decision_time")),
                        ("Settled", outcome.get("exit_day")), ("Asset return", percent(outcome.get("asset_return"), True)),
                        ("SPY return", percent(outcome.get("benchmark_return"), True)),
                        ("Semantic score", score(episode.get("semantic"))), ("Recency score", score(episode.get("recency"))),
                        ("Evidence quality", score(episode.get("evidence_score"))), ("Final score", score(episode.get("score"))),
                        ("Historical citations", episode.get("historical_evidence_ids"))])
    with compaction, st.container(border=True):
        section("Context compaction")
        before, after = memory.get("before_bytes"), memory.get("context_bytes")
        if before is not None and after is not None:
            kv([("Before", f"{number(before)} bytes"), ("After", f"{number(after)} bytes"),
                ("Reduction", percent((before - after) / before) if before else None),
                ("Passed to backend", "Yes" if memory.get("passed_to_backend") else "No")])
            bars([{"State": "Before", "Bytes": before}, {"State": "After", "Bytes": after}],
                 "State", "Bytes", "Serialized bytes", height=75)
        else:
            empty("Compaction unavailable", "No before and after context sizes were archived.")
        st.caption("The Evidence workspace shows the current task’s documents.")


def score(value):
    return f"{value:.4f}" if value is not None else None
