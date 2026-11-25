"""Execution timeline and inspector over archived events."""
import altair as alt
import pandas as pd
import streamlit as st

from .archive import real_tokens
from .charts import chart
from .design import ACCENT, NEGATIVE, WARNING, empty, flow, heading, kv, ms, number, section, timeline


def waterfall(events):
    cursor = 0.0
    spans = []
    for index, event in enumerate(events):
        if event.get("kind") != "node":
            continue
        end = cursor + event.get("latency_ms", 0)
        spans.append({"Node": f'{index + 1:02}  {event["node"]}', "Start": cursor, "End": end,
                      "Duration": end - cursor, "State": status(event)})
        cursor = end
    if not spans:
        return
    chart(alt.Chart(pd.DataFrame(spans)).mark_bar(size=9, cornerRadiusEnd=3).encode(
        y=alt.Y("Node:N", title=None, sort=None, axis=alt.Axis(labelLimit=135)),
        x=alt.X("Start:Q", title="Accumulated node time · ms"), x2="End:Q",
        color=alt.Color("State:N", legend=None, scale=alt.Scale(
            domain=["Complete", "Error", "Recovered", "Retrying", "Pending", "Degraded without memory"],
            range=[ACCENT, NEGATIVE, WARNING, WARNING, WARNING, WARNING])),
        tooltip=["Node", "State", alt.Tooltip("Duration:Q", format=".3f")],
    ).properties(height=max(130, len(spans) * 17)))


def status(event):
    recovery = event.get("recovery_status")
    if recovery in {"retrying", "pending", "recovered", "degraded_without_memory"}:
        return recovery.replace("_", " ").capitalize()
    if event.get("error_code"):
        return "Error"
    return "Complete"


def event_name(event):
    return event.get("tool_name") or event["node"]


def tool_calls(row, events):
    tools = row.get("tool_trace") or []
    if not tools:
        empty("No tool calls", "The selected run has no archived tool results.")
        return
    for index, tool in enumerate(tools):
        event = next((item for item in events if item.get("call_id") and
                      item["call_id"] == tool.get("call_id") and item.get("tool_name") == tool.get("name")), {})
        with st.expander(f'{index + 1:02} · {tool["name"]} · {tool.get("status", "Unavailable")}'):
            kv([("Dispatch", tool.get("dispatch")), ("Caller", event.get("agent")),
                ("Call ID", tool.get("call_id")), ("Error", tool.get("code")),
                ("Evidence", tool.get("evidence_ids") or tool.get("evidence_id")),
                ("Arguments", [f"{key} = {value}" for key, value in (tool.get("arguments") or {}).items()])])


def inspector(row, selected):
    section("Node inspector", status(selected))
    event = selected
    usage = event.get("token_usage") or {}
    kv([("Node", event.get("node")), ("Role / Agent", event.get("agent")), ("Call ID", event.get("call_id")),
        ("Duration", ms(event.get("latency_ms"))), ("Input tokens", number(usage.get("prompt_tokens"))),
        ("Output tokens", number(usage.get("completion_tokens"))), ("Charged tokens", number(usage.get("charged_tokens"))),
        ("Provider attempts", event.get("provider_attempts")), ("Retries", event.get("retry_count")),
        ("Recovery", event.get("recovery_status")), ("Error", event.get("error_code"))])
    for title, key in (("Input summary", "input_summary"), ("Structured output", "output_summary")):
        section(title)
        kv([(label.replace("_", " ").title(), value) for label, value in (event.get(key) or {}).items()])
    ids = [ident for call in row.get("tool_trace", [])
           if event.get("call_id") and call.get("call_id") == event["call_id"]
           for ident in (call.get("evidence_ids") or [call.get("evidence_id")]) if ident]
    stages = {"analyst": "market_report", "trader": "proposal"}
    for citation in row.get("citation_trace", []):
        if (event.get("call_id") and citation.get("call_id") == event["call_id"]) or (
            citation.get("stage") == stages.get(event.get("node")) and event.get("node") in stages
        ):
            ids.extend(citation.get("cited_ids", []))
    debate = (row.get("debate") or {}).get(event.get("node"))
    if debate:
        for claim in debate.get("claims", []):
            ids.extend(claim.get("evidence_ids", []))
            st.write(claim.get("claim") or claim.get("text") or claim.get("statement") or claim)
    section("Citations")
    if ids:
        for ident in dict.fromkeys(ids):
            st.markdown(f"`{ident}`")
    else:
        st.caption("No citations mapped to this event.")


def render(row):
    heading("Agent Trace", "Execution order, node latency and tool results")
    if not row:
        empty("No run loaded", "Open a research run to inspect execution events.")
        return
    events = row.get("agent_trace") or []
    tokens = real_tokens(row)
    for column, label, value in zip(st.columns(4), ("Trace entries", "Role calls", "Measured model tokens", "Retries"),
                                    (len(events), row.get("role_calls"), tokens,
                                     sum(event.get("retry_count", 0) for event in events)), strict=True):
        column.metric(label, number(value))
    if not events:
        empty("Trace unavailable", "This archive does not include agent events.")
    else:
        with st.container(border=True):
            section("Observed agent topology")
            nodes = [event["node"] for event in events if event.get("kind") == "node"]
            route = [node for node in nodes if node in {"bull", "bear", "adjudicate"}]
            flow((event["node"].replace("_", " ").title(),
                  {"Error": "bad", "Complete": "good"}.get(status(event), "warn"))
                 for event in events if event.get("kind") == "node")
            if route:
                trigger = next((item.get("reason") for item in (row.get("debate_trace") or [])
                                if item.get("triggered")), None)
                kv([("Role routing", " → ".join(["route", *route])), ("Debate trigger", trigger)])
        left, main, right = st.columns([1.15, 1.85, 1.5])
        with left, st.container(border=True):
            section("Event timeline")
            names = [f"{index + 1:02} · {event_name(event)}" for index, event in enumerate(events)]
            selection = st.selectbox("Inspect event", range(len(events)), format_func=names.__getitem__,
                                     key=f"trace_{row['run_id']}")
            timeline((name, f'{status(event)} · {ms(event.get("latency_ms"))}')
                     for name, event in zip(names, events, strict=True))
        with main:
            with st.container(border=True):
                section("Duration waterfall")
                waterfall(events)
                st.caption("Node durations stack in recorded order. Nested tools are listed separately.")
            with st.container(border=True):
                section("Tool call tree", number(len(row.get("tool_trace") or [])))
                tool_calls(row, events)
        with right, st.container(border=True):
            inspector(row, events[selection])
            verification = row.get("verification") or {}
            section("Verifier violations", number(len(verification["violations"])) if verification else "Unavailable")
            for item in verification.get("violations", []):
                st.write(item)
    with st.expander("Developer details"):
        st.json({key: row.get(key) for key in ("agent_trace", "tool_trace", "debate_trace", "verification", "errors")},
                expanded=False)
