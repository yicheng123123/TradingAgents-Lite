"""Interactive archive index and side-by-side research comparisons."""
import streamlit as st

from . import archive
from .design import empty, heading, ms, number, percent, section, show


def backend(row):
    return row.get("run_config", {}).get("backend") or "Unavailable"


def comparison(row):
    config = row.get("run_config") or {}
    a3 = config.get("a3") or {}
    latency = row.get("latency_seconds")
    return {"Task": row["task_id"], "Decision date": row["decision_date"], "Backend": backend(row),
            "Status": row.get("status"), "Decision": (row.get("decision") or {}).get("action"),
            "RAG": config.get("rag") or ("Off" if "rag" in config else None),
            "Memory": config.get("memory"), "Debate": a3.get("debate"),
            "Transport": a3.get("transport"), "Verifier": a3.get("verifier"), "Repair": a3.get("repair"),
            "Risk veto": (row.get("decision") or {}).get("risk_veto"), "Role calls": row.get("role_calls"),
            "Tokens": number(archive.real_tokens(row)), "Elapsed": ms(latency * 1000 if latency is not None else None),
            "Evaluated alpha": percent(row.get("alpha"), True), "Snapshot SHA-256": row.get("snapshot_sha256")}


def open_run(identity):
    st.session_state["run_id"] = identity
    st.session_state["page"] = "Research"
    st.query_params["run"] = identity
    st.query_params["page"] = "Research"


def date_in_range(row, date_range):
    dates = [day.isoformat() for day in date_range]
    return not dates or dates[0] <= row["decision_date"] <= dates[-1]


def render():
    heading("Run History", "Review archived decisions and compare run configuration")
    rows = archive.history()
    if not rows:
        empty("No archived runs", "Complete a frozen research task to build the archive.")
        return
    stock, dates, engine, state = st.columns(4)
    symbol = stock.selectbox("Stock", ["All", *sorted({row["symbol"] for row in rows})])
    interval = dates.date_input("Decision date range", value=(), format="YYYY-MM-DD")
    runtime = engine.selectbox("Backend", ["All", *sorted({backend(row) for row in rows})], key="history_backend")
    status = state.selectbox("Result", ["All", "Success", "Failure"])
    filtered = [row for row in rows if (symbol == "All" or row["symbol"] == symbol)
                and date_in_range(row, interval) and (runtime == "All" or backend(row) == runtime)
                and (status == "All" or (row.get("status") == "ok") == (status == "Success"))]
    if not filtered:
        empty("No matching runs", "Adjust the archive filters.")
        return
    section("Archived runs", f"{len(filtered)} / {len(rows)}")
    table = [{"Task": row["task_id"], "Ticker": row["symbol"], "Decision date": row["decision_date"],
              "Backend": backend(row).upper(), "Status": row.get("status"),
              "Action": ((row.get("decision") or {}).get("action") or "Unavailable").upper(),
              "Tokens": number(archive.real_tokens(row)), "Run ID": row["run_id"]} for row in filtered]
    st.dataframe(table, hide_index=True, height=290,
                 column_config={"Run ID": st.column_config.TextColumn(width="large")})
    by_id = {row["run_id"]: row for row in filtered}

    def title(identity):
        row = by_id[identity]
        return f'{row["symbol"]} · {row["decision_date"]} · {identity}'

    with st.container(border=True):
        section("Open saved research")
        selected = st.selectbox("Archived run", list(by_id), format_func=title)
        st.button("Open run", key="open_run", on_click=open_run, args=(selected,))
    if len(by_id) > 1:
        with st.container(border=True):
            section("Compare two runs")
            a, b = st.columns(2)
            first = a.selectbox("Run A", list(by_id), format_func=title)
            second = b.selectbox("Run B", list(by_id), index=1, format_func=title)
            one, two = comparison(by_id[first]), comparison(by_id[second])
            st.dataframe([{"Metric": key, "Run A": show(value), "Run B": show(two[key])}
                          for key, value in one.items()], hide_index=True)
