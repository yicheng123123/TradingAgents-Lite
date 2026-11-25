"""Research configuration, PIT market view and structured decision report."""
import os

import streamlit as st

from .. import research_ui
from ..agents import A3Config
from ..data import DataError, close_available
from ..experiment import upper_tokens
from . import archive
from .charts import market_chart
from .design import badge, empty, flow, heading, html, kv, ms, number, percent, section


PRESETS = {
    "Basic": ("Off", "M0", "D0", "Local", False, False),
    "RAG Enhanced": ("R1", "M0", "D0", "Local", True, False),
    "Full Multi-Agent": ("R1", "M3", "D1", "Local", True, True),
    "Adaptive": ("R1", "M3", "D2", "Local", True, True),
}
CONFIG_KEYS = ("rag", "memory", "debate", "transport", "verifier", "repair")
RESEARCH_KEYS = ("task_set", "task", "backend", "preset", *CONFIG_KEYS)
METRICS = ("Revenue YoY Growth", "Net Profit Margin", "Free Cash Flow", "Debt / Assets")


def apply_preset():
    for key, value in zip(CONFIG_KEYS, PRESETS[st.session_state.preset], strict=True):
        st.session_state[key] = value


def controls():
    section("Task configuration")
    dataset = st.selectbox("Task set", archive.task_sets(), key="task_set")
    if dataset == "Development V2":
        st.caption("Frozen Development partition · Final Holdout sealed")
    try:
        tasks = {f"{task.symbol} · {task.decision_date} · {task.task_id}": task
                 for task in archive.tasks_for(dataset)}
    except (OSError, ValueError, DataError):
        tasks = {}
    choice = st.selectbox("Frozen task", tasks, key="task")
    if choice:
        st.caption(tasks[choice].task_id)
    if not tasks:
        empty("Tasks unavailable", "The local task set could not be loaded.")
    backend = st.selectbox("Backend", ["Offline", "LLM"], key="backend")
    st.selectbox("Configuration preset", PRESETS, key="preset", on_change=apply_preset)
    with st.expander("Advanced agent configuration"):
        rag = st.selectbox("Retrieval", ["Off", "R1", "R2", "R3", "R4"], key="rag")
        memory = st.selectbox("Memory", ["M0", "M1", "M2", "M3"], key="memory")
        debate = st.selectbox("Debate", ["D0", "D1", "D2"], key="debate")
        transport = st.selectbox("Tool transport", ["Local", "MCP"], key="transport")
        verifier = st.checkbox("Evidence verifier", key="verifier")
        repair = st.checkbox("Bounded repair", key="repair")
    config = A3Config(debate=debate, transport=transport.lower(), verifier=verifier, repair=repair)
    st.caption(f"RAG {rag} · {memory} · {debate} · {transport}")
    allowed = True
    if backend == "LLM":
        st.metric("Estimated token ceiling", f"{upper_tokens(config):,}")
        keyed = bool(os.getenv("TALITE_API_KEY", "").strip())
        if not keyed:
            st.warning("TALITE_API_KEY is unavailable. Offline research is ready.")
        approved = st.checkbox("Authorize one paid GLM run", key="approve_paid")
        allowed = keyed and approved
    if st.button("Run frozen task", type="primary", key="execute", disabled=not tasks or not allowed,
                 width="stretch"):
        identity = research_ui.new_run_id()
        st.session_state.pop("run_error", None)
        try:
            with st.status("Running frozen task", expanded=True):
                research_ui.start_run(tasks[choice], run_id=identity, backend=backend.lower(), rag=rag,
                                      memory=memory, debate=debate, transport=transport.lower(),
                                      verifier=verifier, repair=repair)
            if archive.load(identity):
                st.session_state["run_id"] = identity
                st.query_params["run"] = identity
            else:
                st.session_state["run_error"] = "Run finished without a readable research archive."
        except (FileNotFoundError, OSError):
            st.session_state["run_error"] = "Local input unavailable. Check the frozen snapshot, index and model files."
        except Exception:
            st.session_state["run_error"] = "Run blocked. Check the selected configuration, credentials and available budget."
        if backend == "LLM":
            st.session_state["reset_paid"] = True
        st.session_state["research_config"] = {key: st.session_state[key] for key in RESEARCH_KEYS}
        st.rerun()
    if issue := st.session_state.get("run_error"):
        st.error(issue)
    st.session_state["research_config"] = {key: st.session_state[key] for key in RESEARCH_KEYS}
    return tasks.get(choice)


@st.cache_data(show_spinner=False, max_entries=32)
def read_market(task, snapshot_hash, revision):
    return archive.market_series(task, snapshot_hash)


def market(task, row):
    section(f"{task.symbol} · Daily OHLCV", "Historical replay")
    try:
        directory = archive.market_directory(task)
        revision = (directory / "manifest.json").stat().st_mtime_ns
        bars, spy, source = read_market(task, row.get("snapshot_sha256") if row else None, revision)
    except (OSError, ValueError, KeyError, DataError):
        empty("Market snapshot unavailable", "An eligible, matching frozen snapshot is required for this chart.")
        return
    last = bars[-1]
    change = last.close / bars[-2].close - 1
    html(badge(f"${last.close:,.2f}  {change:+.2%}", "good" if change >= 0 else "bad"))
    with st.container(key="market_chart"):
        market_chart(bars, spy, task.decision_date)
    st.caption(f"{source} · decision session {task.decision_date} · {min(55, len(bars))} sessions shown")
    if not spy:
        st.caption("SPY comparison unavailable.")


def decision(row):
    section("Research report", "Archived decision" if row else "Awaiting run")
    if not row:
        empty("No research report", "Execute a frozen task or open a run from Run History.")
        return
    report = row.get("report") or {}
    decision_data = row.get("decision") or {}
    choice = decision_data.get("action", "Unavailable").upper()
    html(badge(choice, {"BUY": "good", "SELL": "bad", "HOLD": "warn"}.get(choice, "")))
    st.write(decision_data.get("rationale") or "Decision rationale unavailable.")
    if report.get("view"):
        st.caption(report["view"])
    verification = row.get("verification") or {}
    latency = row.get("latency_seconds")
    kv([("Task", row["task_id"]), ("Risk gate", decision_data.get("risk_veto") or
         ("Clear" if row.get("status") == "ok" else "Unavailable")),
        ("Verifier", ("Passed" if verification.get("passed") else "Failed") if verification else "Unavailable"),
        ("Actual model tokens", number(archive.real_tokens(row))),
        ("Elapsed", ms(1000 * latency if latency is not None else None))])
    with st.expander("Key evidence", expanded=True):
        evidence = row.get("evidence") or {}
        for ident in decision_data.get("evidence_ids", []):
            item = evidence.get(ident, {})
            st.markdown(f"`{ident}`")
            if item.get("excerpt"):
                st.caption(item["excerpt"][:200])
        if not decision_data.get("evidence_ids"):
            st.caption("No evidence cited.")
    with st.expander("Bull / Bear views"):
        debate = row.get("debate") or {}
        if not debate:
            st.caption("Debate output unavailable for this run.")
        for stance, view in debate.items():
            st.markdown(f"**{stance.title()}**")
            for claim in view.get("claims", []):
                st.write(claim.get("claim") or claim.get("text") or claim.get("statement") or claim)
    for error in row.get("errors", []):
        st.error(str(error.get("code", "Run error")))


def fundamentals(row, task):
    section("Fundamentals", "SEC XBRL")
    cutoff = close_available(task.decision_date)
    facts = {}
    recorded = (row or {}).get("evidence") or {}
    config = (row or {}).get("run_config") or {}
    for item in [*archive.xbrl_series(task, METRICS, config), *recorded.values()]:
        metric = item.get("metric")
        eligible = archive.instant(item.get("eligible_at"))
        if metric in METRICS and eligible is not None and eligible <= cutoff:
            facts.setdefault(metric, {})[item["period_end"]] = item
    latest = {key: items[max(items)] for key, items in facts.items()}
    def format_fact(item):
        return percent(item["value"]) if item["unit"] == "ratio" else f'${item["value"]:,.0f}'

    kv([(metric, f'{format_fact(latest[metric])} · {latest[metric]["period_end"]}' if metric in latest else None)
        for metric in METRICS])
    trend = [{"Metric": key, "Period end": item["period_end"], "Value": format_fact(item)}
             for key, items in facts.items() if len(items) > 1
             for item in sorted(items.values(), key=lambda item: item["period_end"])]
    if trend:
        with st.expander("Historical fundamentals · as of decision cutoff"):
            st.dataframe(trend, hide_index=True)
    else:
        st.caption("Financial trend unavailable. No comparable dated XBRL series at this cutoff.")


def settlement(row):
    with st.expander("Post-decision settlement · evaluation only"):
        outcome = row.get("outcome") or {}
        if not outcome:
            empty("Settlement unavailable", "No evaluated outcome is recorded for this run.")
            return
        kv([("Entry session", outcome.get("entry_day")), ("Exit session", outcome.get("exit_day")),
            ("Available at", outcome.get("available_at")), ("Asset price return", percent(outcome.get("asset_return"), True)),
            ("SPY price return", percent(outcome.get("benchmark_return"), True)),
            ("Position alpha", percent(row.get("alpha"), True))])


def render(row):
    heading("Research", "Frozen market replay and point-in-time research")
    if "task_set" not in st.session_state:
        for key, value in st.session_state.get("research_config", {}).items():
            if value is not None:
                st.session_state[key] = value
    left, center, right = st.columns([1.05, 2.75, 1.4])
    with left, st.container(border=True):
        selected = controls()
    with center:
        with st.container(border=True):
            task = archive.task_of(row) if row else selected
            if task:
                market(task, row)
            else:
                empty("No market selected", "Select a frozen task to inspect its market snapshot.")
        with st.container(border=True):
            section("Execution flow")
            events = (row or {}).get("agent_trace") or []
            nodes = list(dict.fromkeys(event["node"] for event in events if event.get("kind") == "node"))
            if nodes:
                failed = {event["node"] for event in events if event.get("error_code")}
                flow((node.replace("_", " ").title(), "bad" if node in failed else "good") for node in nodes)
            else:
                st.caption("No execution flow in the selected archive.")
        if task:
            with st.container(border=True):
                fundamentals(row, task)
        if row:
            settlement(row)
    with right, st.container(border=True):
        decision(row)
