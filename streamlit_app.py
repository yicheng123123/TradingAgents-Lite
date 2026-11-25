"""TradingAgents-Lite local research dashboard."""
from datetime import UTC, datetime
import os
import sqlite3

import streamlit as st

from tradingagents_lite.agents import A3Config
from tradingagents_lite.experiment import upper_tokens
from tradingagents_lite.research_ui import (
    ARCHIVES, PAGES, archive_for, load_run, new_run_id, research_tasks, start_run, token_dashboard, v2_coverage,
)
from tradingagents_lite.zhipu_account import CONSOLE_TZ, account_snapshot

st.set_page_config(page_title="TradingAgents-Lite", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {max-width: 1300px; padding-top: 2.1rem;}
.talite-hero {background:#102443;color:#f5f8fc;padding:1.55rem 2rem;border-radius:16px;
             border-left:5px solid #4dd5b0;margin:0 0 1.3rem;}
.talite-hero h1 {font-size:2.15rem;letter-spacing:-.04em;color:#f5f8fc;padding:0;margin:0 0 .25rem;}
.talite-hero p {color:#b8cbe6;font-size:1rem;margin:0;}
[data-testid="stMetric"] {border:1px solid #dbe4ef;border-radius:12px;padding:.70rem .9rem;}
[data-testid="stSidebar"] {border-right:1px solid #dbe4ef;}
</style>
""", unsafe_allow_html=True)


def hero(title, detail):
    st.markdown(f'<div class="talite-hero"><h1>{title}</h1><p>{detail}</p></div>', unsafe_allow_html=True)


def research(row):
    hero("Research", "Frozen market replay and evidence review")
    dataset = st.selectbox("Task set", ["Legacy", "Development V2"], key="task_set")
    tasks = research_tasks(dataset)
    if dataset == "Development V2":
        coverage = v2_coverage()
        if coverage:
            count = coverage["tasks"]
            xbrl = coverage["xbrl_fields"]["quarter"]["Revenue"]
            st.caption(f"Development inputs · filings {coverage['sec_candidates']}/{count} · "
                       f"news {coverage['news_candidates']}/{count} · RAG {coverage['rag_candidates']}/{count} · "
                       f"XBRL quarterly revenue {xbrl}/{count}")
        st.caption("Final Holdout is sealed. V2 research uses its own evidence and Memory.")
    lookup = {f"{t.symbol} · {t.decision_date} · {t.task_id}": t for t in tasks}
    top = st.columns([1.6, 1, 1])
    choice = top[0].selectbox("Frozen task", list(lookup), key="task")
    backend = top[1].selectbox("Backend", ["Offline", "LLM"], key="backend")
    transport = top[2].selectbox("Tool transport", ["Local", "MCP"], key="transport")
    options = st.columns(3)
    rag = options[0].selectbox("Retrieval", ["Off", "R1", "R2", "R3", "R4"], key="rag")
    memory = options[1].selectbox("Memory", ["M0", "M1", "M2", "M3"], key="memory")
    debate = options[2].selectbox("Debate", ["D0", "D1", "D2"], key="debate")
    advanced = st.columns(3)
    verifier = advanced[0].checkbox("Evidence verifier", key="verifier")
    repair = advanced[1].checkbox("Bounded repair", key="repair")
    config = A3Config(debate=debate, transport=transport.lower(), verifier=verifier, repair=repair)
    keyed = bool(os.getenv("TALITE_API_KEY", "").strip())
    approved = False
    if backend == "LLM":
        st.metric("Estimated token ceiling", f"{upper_tokens(config):,}")
        if not keyed:
            st.warning("TALITE_API_KEY is unavailable. Offline research is ready.")
        approved = st.checkbox("Authorize one paid GLM run", key="approve_paid")
    active = st.session_state.get("running", False)
    if st.button("Run frozen task", type="primary", key="execute",
                 disabled=active or (backend == "LLM" and not (approved and keyed))):
        identity = new_run_id()
        st.session_state["running"] = True
        st.session_state["run_id"] = identity
        st.session_state.pop("run_error", None)
        st.query_params["run"] = identity
        try:
            with st.status("Running frozen task", expanded=True) as panel:
                start_run(lookup[choice], run_id=identity, backend=backend.lower(), rag=rag, memory=memory,
                          debate=debate, transport=transport.lower(), verifier=verifier, repair=repair)
                row = load_run(identity)
                failed = row is None or row.get("status") != "ok"
                panel.update(label="Research failed" if failed else "Research complete",
                             state="error" if failed else "complete")
        except Exception as exc:
            row = None
            st.session_state["run_error"] = (
                "Local input unavailable. Check the frozen snapshot, index and model files."
                if isinstance(exc, (FileNotFoundError, OSError))
                else "Run blocked. Check the selected configuration, credentials and available budget.")
        finally:
            st.session_state["running"] = False
            if backend == "LLM":
                st.session_state["reset_paid"] = True
    if st.session_state.get("run_error"):
        st.error(st.session_state["run_error"])
    if row:
        st.divider()
        st.subheader("Decision")
        cards = st.columns(4)
        cards[0].metric("Task", row["task_id"])
        cards[1].metric("Action", row["decision"]["action"].upper())
        cards[2].metric("Risk gate", row["decision"].get("risk_veto") or "Clear")
        cards[3].metric("Run status", row["status"])
        observed = row.get("run_config", {})
        active_modes = observed.get("a3", {})
        if observed:
            label = [observed["backend"].upper(), active_modes.get("transport", "local"),
                     "RAG " + (observed["rag"] or "Off"), observed["memory"], active_modes.get("debate", "D0")]
            st.caption("Run configuration · " + " · ".join(label))
        st.write(row["decision"]["rationale"])
        if row["errors"]:
            st.error(" · ".join(str(e["code"]) for e in row["errors"]))
    else:
        st.caption("Choose a frozen task and run it to populate the trace and evidence views.")


def agent_trace(row):
    hero("Agent Trace", "Execution order, tool activity and recovery")
    if not row:
        st.info("Run a frozen task in Research.")
        return
    events = row.get("agent_trace", [])
    cards = st.columns(3)
    cards[0].metric("Trace entries", len(events))
    cards[1].metric("Role calls", row.get("role_calls", 0))
    cards[2].metric("Measured model tokens", sum(
        e["token_usage"]["prompt_tokens"]+e["token_usage"]["completion_tokens"] for e in events))
    st.dataframe([{"Step": i+1, "Node": e["node"], "Tool": e["tool_name"], "Call ID": e["call_id"],
                   "ms": round(e["latency_ms"], 2), "Error": e["error_code"], "Retry": e["retry_count"],
                   "Recovery": e["recovery_status"]} for i, e in enumerate(events)], hide_index=True)
    st.subheader("Tool results")
    st.dataframe([{k: e.get(k) for k in ("name", "status", "code", "call_id", "evidence_id", "evidence_ids")}
                  for e in row.get("tool_trace", [])], hide_index=True)
    st.subheader("Debate and verification")
    for stance, view in row.get("debate", {}).items():
        with st.expander(stance.title(), expanded=True):
            st.dataframe(view["claims"], hide_index=True)
    if row.get("debate_trace"):
        st.json(row["debate_trace"], expanded=False)
    if row.get("verification"):
        st.json(row["verification"], expanded=False)
    if row.get("errors"):
        st.error(" · ".join(str(e["code"]) for e in row["errors"]))


def evidence_memory(row):
    hero("Evidence & Memory", "Current evidence and eligible historical experience")
    if not row:
        st.info("Run a frozen task in Research.")
        return
    st.subheader("Current task evidence")
    st.dataframe([{"Evidence ID": key, "Source": item.get("source_url", "Frozen market snapshot"),
                   "Eligible at": item.get("eligible_at"), "PIT": item.get("pit_level"), "Page": item.get("page")}
                  for key, item in row["evidence"].items()], hide_index=True)
    if row["evidence"]:
        selected = st.selectbox("Inspect evidence", list(row["evidence"]), key="inspect")
        item = row["evidence"][selected]
        if item.get("excerpt"):
            st.text(item["excerpt"][:800])
        elif item.get("bars"):
            st.dataframe(item["bars"], hide_index=True)
        else:
            st.json(item, expanded=False)
    for trace in row.get("tool_trace", []):
        stages = (trace.get("retrieval_trace") or {}).get("stages", {})
        if stages:
            with st.expander(f"{trace['name']} rankings", expanded=True):
                st.dataframe([{"Stage": stage, **candidate} for stage, values in stages.items()
                              for candidate in values], hide_index=True)
    st.subheader("Historical experience")
    memory = row.get("memory_trace") or {}
    if memory.get("error_type"):
        st.warning(memory.get("recovery_status", "Memory unavailable"))
    candidates = memory.get("selected", [])
    if candidates:
        st.dataframe([{k: e.get(k) for k in ("source_task_id", "memory_id", "semantic", "recency",
                                           "score", "historical_evidence_ids")} for e in candidates], hide_index=True)
        for episode in memory.get("context", []):
            with st.expander(f"Episode · {episode['memory_id']}"):
                st.text(episode.get("lesson", ""))
    else:
        st.caption("No eligible history selected for this run.")


def benchmark():
    hero("Benchmark", "Archived evaluation metrics and sample sizes")
    st.info("B1 · Pending real GLM results")
    stage = st.selectbox("Archive", list(ARCHIVES), index=4, key="archive")
    try:
        data = archive_for(stage)
    except (OSError, ValueError):
        st.error("The selected archive is unavailable.")
        return
    st.caption(f"Source · results/{ARCHIVES[stage]}")
    if stage in ("A0", "B0"):
        st.caption("Deterministic offline momentum")
        cards = st.columns(3)
        cards[0].metric("Tasks", data.get("tasks", "—"))
        cards[1].metric("Mean alpha", data.get("mean_alpha", "—"))
        cards[2].metric("Non-Hold coverage", data.get("non_hold_coverage", data.get("coverage", "—")))
    elif stage == "A1":
        st.caption(f"Local retrieval · public sample {data['public_questions']} · revision {data['revision'][:12]}")
        test = data["metrics"].get("test", {})
        if test:
            st.bar_chart({"Mode": list(test),
                          "Page recall@10": [metric["page_proxy"]["recall@10"] for metric in test.values()]},
                         x="Mode", y="Page recall@10")
        st.dataframe([{"Split": split, "Mode": mode, "N": row["n"], **row["page_proxy"],
                       "Mean ms": row["mean_latency_ms"]}
                      for split, modes in data["metrics"].items() for mode, row in modes.items()],
                     hide_index=True)
    elif stage == "A2":
        st.caption("Frozen Development · deterministic replay plus local embedding")
        st.dataframe([{"Mode": mode, "N": row["tasks"], "Relevant queries": row["relevance_queries"],
                       "Recall@1": row["recall@1"], "Recall@3": row["recall@3"],
                       "Reused": row["reused_tasks"], "Mean ms": row["mean_query_ms"]}
                      for mode, row in data["development"].items()], hide_index=True)
    else:
        st.caption(f"Scripted failures, offline policies, Mock LLM and real MCP · {data['fault_cases']} fault cases")
        st.dataframe([{"Arm": arm, "Completes / N": str(g["completion"]), "Recovery / N": str(g["recovery"]),
                       "Role calls": g["role_calls"], "Mean ms": g["latency_ms"]["mean"],
                       "Real tokens": g["real_glm_tokens"], "Mock tokens": g["mock_tokens"]}
                      for arm, g in data["groups"].items()], hide_index=True)
        st.metric("STDIO tool equivalence", " / ".join(map(str, data["mcp_equivalence"])))
    with st.expander("Archive values"):
        st.json(data, expanded=False)


def account_status(section, now):
    if section.synced_at is None:
        status = "Unavailable"
    else:
        label = "Last sync" if section.is_fresh(now) else "Stale cache · last sync"
        time = section.synced_at.astimezone(CONSOLE_TZ).isoformat(sep=" ", timespec="seconds")
        status = f"{label} {time}"
    return f"{status} · {section.issue}" if section.issue else status


def bigmodel_account():
    st.subheader("BigModel Account")
    refresh = st.button("Refresh BigModel account", key="refresh_bigmodel")
    snapshot = account_snapshot(st.session_state.get("bigmodel_account"), refresh=refresh)
    st.session_state["bigmodel_account"] = snapshot
    now = datetime.now(UTC)
    if not refresh and snapshot.balance.synced_at is None and snapshot.balance.issue is None:
        st.caption("Select Refresh BigModel account to query the console.")

    st.markdown("**API account · CNY**")
    balance = snapshot.balance.value
    for column, label, amount in zip(st.columns(3), ("Available balance", "Total spend", "Amount granted"),
                                     (balance.available_cny, balance.spend_cny, balance.granted_cny)
                                     if balance is not None else (None, None, None), strict=True):
        column.metric(label, f"¥{amount:,.2f}" if amount is not None else "Unavailable")
    st.caption(account_status(snapshot.balance, now))

    st.markdown("**GLM-5.3 resource packages · Tokens**")
    packages = snapshot.packages.value
    active = packages.active(now) if packages is not None else ()
    for column, label, amounts in zip(st.columns(2), ("Remaining Tokens", "Available Tokens"),
                                      ([row.remaining_tokens for row in active],
                                       [row.available_tokens for row in active]), strict=True):
        if packages is not None and packages.complete and all(amount is not None for amount in amounts):
            column.metric(label, f"{sum(amount for amount in amounts if amount is not None):,}")
        else:
            column.metric(label, "Unavailable")
    st.caption("Total package allocation: Unavailable. The console response does not confirm original allocation.")
    if active:
        st.dataframe([
            {"Remaining Tokens": row.remaining_tokens, "Available Tokens": row.available_tokens,
             "Expires (UTC+08:00)": row.expires_at.astimezone(CONSOLE_TZ).isoformat(sep=" ", timespec="seconds")}
            for row in active
        ], hide_index=True)
    if packages is not None and not packages.complete:
        st.caption("Some package rows or eligibility could not be verified. The table shows confirmed packages.")
    st.caption(account_status(snapshot.packages, now))
    st.caption("Coding Plan quota is separate. View it in the BigModel console.")


def local_token_budget():
    st.subheader("Project Token Budget")
    try:
        block = token_dashboard()
    except (OSError, sqlite3.Error):
        st.error("The token ledger could not be read.")
        return
    if not block["exists"]:
        st.caption("No local token ledger exists. No file was created.")
    cols = st.columns(4)
    cols[0].metric("Project cap", f"{block['limit'] / 1_000_000:.2f} M")
    cols[0].caption(f"{block['limit']:,} Tokens")
    for col, title, key in zip(cols[1:], ("Settled", "Pending / unknown", "Available"),
                               ("settled", "pending", "remaining"), strict=True):
        col.metric(title, f"{block[key]:,}")
    st.dataframe(block["phases"], hide_index=True)
    st.bar_chart({"phase": [row["phase"] for row in block["phases"]],
                  "remaining": [row["remaining"] for row in block["phases"]]}, x="phase", y="remaining")
    a, b = st.columns(2)
    estimate = block["list_price_estimate_cny"]
    a.metric("GLM-5.3 list-price estimate", f"¥{estimate:.4f}" if estimate is not None else "Unavailable")
    b.metric("Provider invoice", "Unavailable")
    st.caption("Cached tokens are included in input. Reasoning is a reported detail.")


def tokens():
    hero("Token Dashboard", "Local budget and BigModel account")
    local_token_budget()
    st.divider()
    bigmodel_account()


if st.session_state.pop("reset_paid", False):
    st.session_state["approve_paid"] = False
st.sidebar.title("TradingAgents-Lite")
requested_page = st.query_params.get("page")
page = st.sidebar.radio("Research console", PAGES, key="page",
                        index=PAGES.index(requested_page) if requested_page in PAGES else 0)
identity = st.session_state.get("run_id") or st.query_params.get("run")
row = load_run(identity)
if row:
    st.sidebar.caption(f"Loaded task · {row['task_id']}")
if page == "Research":
    research(row)
elif page == "Agent Trace":
    agent_trace(row)
elif page == "Evidence & Memory":
    evidence_memory(row)
elif page == "Benchmark":
    benchmark()
else:
    tokens()
