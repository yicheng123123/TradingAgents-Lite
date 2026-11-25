"""TradingAgents-Lite financial research terminal."""
import os
import sqlite3

import streamlit as st

from tradingagents_lite.research_ui import token_dashboard
from tradingagents_lite.terminal import archive, benchmark, evidence, history, memory_view, research, trace, usage
from tradingagents_lite.terminal.design import compact, html, install, number, safe


PAGES = {
    "Research": research.render,
    "Agent Trace": trace.render,
    "Evidence": evidence.render,
    "Memory": memory_view.render,
    "Benchmark": benchmark.render,
    "Usage": usage.render,
    "Run History": history.render,
}


def topbar(row, page):
    backend = (row or {}).get("run_config", {}).get("backend")
    if page == "Research":
        backend = st.session_state.get("backend") or backend
    try:
        block = token_dashboard()
        used = f'{number(block["settled"])} / {compact(block["limit"])}'
    except (OSError, sqlite3.Error):
        used = "Unavailable"
    task = f'{row["symbol"]} · {row["decision_date"]}' if row else "No run loaded"
    status = "Archived snapshot" if row and row.get("snapshot_sha256") else "Local frozen dataset"
    api = "Key configured" if os.getenv("TALITE_API_KEY", "").strip() else "Inactive · no key"
    entries = [("Active research", task), ("Data snapshot", status), ("Backend", (backend or "offline").upper()),
               ("API status", api), ("Project tokens", used)]
    html('<div class="topline">' + "".join(f'<div><span class="eyebrow">{safe(label)}</span>'
                                          f'<strong>{safe(value)}</strong></div>' for label, value in entries)
         + "</div>")


def main():
    st.set_page_config(page_title="TradingAgents-Lite", layout="wide", initial_sidebar_state="expanded")
    install()
    if st.session_state.pop("reset_paid", False):
        st.session_state["approve_paid"] = False
    identity = st.query_params.get("run") or st.session_state.get("run_id")
    row = archive.load(identity)
    with st.sidebar:
        html('<div class="brand"><span class="brand-mark" aria-hidden="true">TA</span>'
             '<div class="brand-text"><strong>TradingAgents-Lite</strong><small>Research terminal</small></div></div>')
        requested = st.query_params.get("page", "Research")
        if requested == "Token Dashboard":
            requested = "Usage"
        elif requested == "Evidence & Memory":
            requested = "Evidence"
        page = st.radio("Workspaces", list(PAGES), index=list(PAGES).index(requested) if requested in PAGES else 0,
                        key="page", label_visibility="collapsed")
        if st.query_params.get("page") != page:
            st.query_params["page"] = page
        st.divider()
        if row:
            st.caption("ACTIVE ARCHIVE")
            st.markdown(f'**{row["task_id"]}**')
            st.caption(f'{row["symbol"]} · {row["decision_date"]}')
            with st.expander("Run identity"):
                st.code(identity, language=None, wrap_lines=True)
        else:
            st.caption("No active archive")
    topbar(row, page)
    if identity and not row:
        st.warning("The requested run archive is unavailable.")
    if page in {"Benchmark", "Usage", "Run History"}:
        PAGES[page]()
    else:
        PAGES[page](row)


main()
