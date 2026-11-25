"""Benchmark dashboard over recorded measurements."""
import altair as alt
import pandas as pd
import streamlit as st

from ..research_ui import ARCHIVES, archive_for
from . import archive
from .charts import bars, chart
from .design import ACCENT, POSITIVE, WARNING, empty, heading, kv, section


def rag(data):
    split = st.radio("Query split", list(data["metrics"]), index=list(data["metrics"]).index("test"),
                     horizontal=True)
    granularity = st.radio("Ground truth", ["page_proxy", "document"], horizontal=True)
    selected = data["metrics"][split]
    metrics = ["recall@10", "mrr@10", "ndcg@10"]
    rows = [{"Mode": mode, "Queries": row["n"], **row[granularity], "Mean latency (ms)": row["mean_latency_ms"]}
            for mode, row in selected.items()]
    section("Retrieval quality", "Real local retrieval")
    st.caption(f'{split} · {granularity} · per-mode query counts shown · revision {data["revision"][:12]}')
    spec = alt.Chart(pd.DataFrame(rows)).transform_fold(metrics, as_=["Metric", "Value"]).mark_bar(
        cornerRadiusEnd=3).encode(x=alt.X("Mode:N", title=None), xOffset="Metric:N",
        y=alt.Y("Value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Metric:N", scale=alt.Scale(domain=metrics, range=[ACCENT, POSITIVE, WARNING]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Mode:N", "Metric:N", alt.Tooltip("Value:Q", format=".4f"), "Queries:Q"]).properties(height=210)
    chart(spec)
    st.dataframe(rows, hide_index=True)


def memory(data):
    recall, compact = st.columns(2)
    development = data["development"]
    with recall, st.container(border=True):
        section("Memory recall", "Frozen Development")
        bars([{"Mode": key, "Recall": row["recall@3"]} for key, row in development.items()],
             "Mode", "Recall", "Recall@3")
        st.dataframe([{"Mode": key, "Tasks": row["tasks"], "Relevant queries": row["relevance_queries"],
                       "Recall@1": row["recall@1"], "Recall@3": row["recall@3"],
                       "Reused tasks": row["reused_tasks"]} for key, row in development.items()], hide_index=True)
    with compact, st.container(border=True):
        section("Context compaction", "Serialized bytes")
        bars([{"Mode": key, "Saved": row["before_bytes"] - row["context_bytes"]}
              for key, row in development.items()], "Mode", "Saved", "Bytes saved")
        st.dataframe([{"Mode": key, "Tasks": row["tasks"], "Before": row["before_bytes"],
                       "After": row["context_bytes"]} for key, row in development.items()], hide_index=True)
    with st.expander("Synthetic memory checks"):
        kv([(key.replace("_", " ").title(), value) for key, value in data["synthetic"].items()])


def recovery(data):
    with st.container(border=True):
        section("Tool transport", "Real local / MCP")
        transport = [{"Transport": key, "Tool calls": row["n"], "Mean": row["mean"],
                      "P50": row["p50"], "P95": row["p95"]} for key, row in data["transport"].items()]
        bars(transport, "Transport", "Mean", "Mean tool latency · ms", height=70)
        st.caption("Per-call latency. MCP startup timings are recorded separately.")
        st.dataframe(transport, hide_index=True)
        with st.expander("Call-level measurements"):
            st.dataframe(data["tools"], hide_index=True)
    left, right = st.columns(2)
    groups = data["groups"]
    with left, st.container(border=True):
        section("Recovery", "Synthetic fault injection")
        st.caption(f'{data["fault_cases"]} archived fault scenarios · each fraction uses its own eligible denominator')
        st.dataframe([{"Arm": arm, "Completion / N": " / ".join(map(str, row["completion"])),
                       "Recovery / N": " / ".join(map(str, row["recovery"])),
                       "Runs": row["latency_ms"]["n"]} for arm, row in groups.items() if arm.startswith(("F", "V"))],
                     hide_index=True)
    with right, st.container(border=True):
        section("Debate calls", "Deterministic offline")
        bars([{"Arm": arm, "Role calls": groups[arm]["role_calls"]} for arm in ("D0", "D1", "D2")],
             "Arm", "Role calls", "Calls", height=100)
        kv([(arm, f'{groups[arm]["role_calls"]} calls across {groups[arm]["latency_ms"]["n"]} tasks')
            for arm in ("D0", "D1", "D2")])
    mock = groups.get("mock_llm", {})
    section("Mock model", "Mock")
    kv([("Runs", (mock.get("latency_ms") or {}).get("n")), ("Mock tokens", mock.get("mock_tokens")),
        ("Real GLM tokens", mock.get("real_glm_tokens"))])


def coverage():
    left, right = st.columns(2)
    manifest = archive.ROOT / "tasks/data-expansion-v2/manifest.json"
    if manifest.is_file():
        development = archive.read_json(manifest)["partitions"]["development"]
        tickers, sectors = development["tickers"], development["sectors"]
        source = "tasks/data-expansion-v2/manifest.json · Development partition"
    else:
        tickers = {}
        for task in archive.tasks_for("Legacy"):
            tickers[task.symbol] = tickers.get(task.symbol, 0) + 1
        sectors = {}
        source = "tasks/development.jsonl"
    st.caption(f'{source} · {sum(tickers.values())} tasks · {len(tickers)} tickers')
    with left, st.container(border=True):
        section("Ticker coverage")
        bars([{"Ticker": key, "Tasks": value} for key, value in tickers.items()], "Ticker", "Tasks",
             "Frozen tasks", height=max(95, len(tickers) * 16))
    with right, st.container(border=True):
        section("Sector coverage")
        if sectors:
            bars([{"Sector": key, "Tasks": value} for key, value in sectors.items()], "Sector", "Tasks", "Frozen tasks")
        else:
            empty("Sector coverage unavailable", "The legacy task archive has no sector annotations.")


def render():
    heading("Benchmark", "Archived experiments, denominators and measurement sources")
    st.info("Pending real GLM benchmark results. Available archives cover local retrieval, rules, mock calls and injected faults.")
    labels = {"A1": "RAG retrieval", "A2": "Memory & compaction", "A3": "MCP, recovery & debate",
              "Dataset": "Dataset coverage", "A0": "Smoke baseline", "B0": "Development baseline"}
    stage = st.selectbox("Experiment", list(labels), format_func=labels.__getitem__, key="archive")
    try:
        if stage == "Dataset":
            coverage()
            return
        data = archive_for(stage)
        st.caption(f"Source · results/{ARCHIVES[stage]}")
        if stage == "A1":
            rag(data)
        elif stage == "A2":
            memory(data)
        elif stage == "A3":
            recovery(data)
        else:
            section("Historical rule baseline", "Deterministic offline")
            kv([("Tasks", data.get("tasks")), ("Mean position alpha", data.get("mean_alpha")),
                ("Non-Hold coverage", data.get("non_hold_coverage") or data.get("coverage"))])
        with st.expander("Archive values"):
            st.json(data, expanded=False)
    except (OSError, ValueError, KeyError):
        st.error("The selected experiment archive is missing or unreadable.")
