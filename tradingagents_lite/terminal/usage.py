"""Read-only provider account and local project ledger."""
from datetime import UTC, datetime
import sqlite3

import altair as alt
import pandas as pd
import streamlit as st

from ..research_ui import token_dashboard
from .. import zhipu_account
from .charts import chart
from .design import ACCENT, BORDER, WARNING, compact, heading, number, percent, section, stats


def account_status(section, now):
    if section.synced_at is None:
        status = "Unavailable"
    else:
        label = "Last sync" if section.is_fresh(now) else "Stale cache · last sync"
        time = section.synced_at.astimezone(zhipu_account.CONSOLE_TZ).isoformat(sep=" ", timespec="seconds")
        status = f"{label} {time}"
    return f"{status} · {section.issue}" if section.issue else status


def bigmodel_account():
    section("BigModel account", "Read-only")
    refresh = st.button("Refresh BigModel account", key="refresh_bigmodel")
    snapshot = zhipu_account.account_snapshot(st.session_state.get("bigmodel_account"), refresh=refresh)
    st.session_state["bigmodel_account"] = snapshot
    now = datetime.now(UTC)
    api, resources = st.columns([1.35, 1])
    with api:
        section("API account · CNY")
        balance = snapshot.balance.value
        amounts = ((balance.available_cny, balance.spend_cny, balance.granted_cny)
                   if balance else (None, None, None))
        for column, label, amount in zip(st.columns(3), ("Available balance", "Total spend", "Amount granted"),
                                         amounts, strict=True):
            column.metric(label, f"¥{amount:,.2f}" if amount is not None else "Unavailable")
        st.caption(account_status(snapshot.balance, now))
    with resources:
        section("GLM-5.3 resource packages")
        packages = snapshot.packages.value
        active = packages.active(now) if packages is not None else ()
        for column, label, field in zip(st.columns(2), ("Remaining Tokens", "Available Tokens"),
                                         ("remaining_tokens", "available_tokens"), strict=True):
            amounts = [getattr(row, field) for row in active]
            available = packages is not None and packages.complete and all(amount is not None for amount in amounts)
            column.metric(label, number(sum(amounts)) if available else "Unavailable")
        st.caption(account_status(snapshot.packages, now))
    with st.expander("Package details"):
        if active:
            st.dataframe([{
                "Remaining Tokens": row.remaining_tokens, "Available Tokens": row.available_tokens,
                "Expires (UTC+08:00)": row.expires_at.astimezone(zhipu_account.CONSOLE_TZ).isoformat(
                    sep=" ", timespec="seconds"),
            } for row in active], hide_index=True)
        st.caption("Original allocation · Unavailable")
        if packages is not None and not packages.complete:
            st.caption("Some package rows or eligibility could not be verified. The table shows confirmed packages.")
        st.caption("Coding Plan quota is reported in the BigModel console.")
    if not refresh and snapshot.balance.synced_at is None and snapshot.balance.issue is None:
        st.caption("Select Refresh BigModel account to query the console.")


def budget_chart(phases):
    intervals = ["Settled", "Pending / unknown", "Remaining"]
    frame = pd.DataFrame([{"Phase": row["phase"], "Settled": row["settled"], "Pending / unknown": row["pending"],
                           "Remaining": row["remaining"]} for row in phases])
    spec = alt.Chart(frame).transform_fold(intervals, as_=["State", "Tokens"]).transform_calculate(
        ordering="datum.State === 'Settled' ? 0 : datum.State === 'Pending / unknown' ? 1 : 2"
    ).mark_bar(size=8).encode(
        y=alt.Y("Phase:N", sort=list(frame["Phase"]), title=None),
        x=alt.X("Tokens:Q", title="Project phase budget · tokens", axis=alt.Axis(format="~s", tickCount=7)),
        order="ordering:Q",
        color=alt.Color("State:N", scale=alt.Scale(domain=intervals, range=[ACCENT, WARNING, BORDER]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["Phase:N", "State:N", alt.Tooltip("Tokens:Q", format=",")],
    ).properties(height=len(phases) * 29)
    chart(spec)


def local_budget(block):
    section("Local project budget", "Ledger" if block["exists"] else "No ledger")
    fraction = block["settled"] / block["limit"]
    stats(("Project cap", compact(block["limit"]), f'{number(block["limit"])} Tokens'),
          ("Settled", compact(block["settled"]), f'{number(block["settled"])} Tokens'),
          ("Pending / unknown", compact(block["pending"]), f'{number(block["pending"])} Tokens'),
          ("Remaining", compact(block["remaining"]), f'{number(block["remaining"])} Tokens'),
          ("Settled utilization", percent(fraction), "Settled ÷ project cap"))
    if not block["exists"]:
        st.caption("No project ledger is present. Budget values reflect the configured cap and zero recorded attempts.")
    budget_chart(block["phases"])
    with st.expander("Full phase budget figures"):
        st.dataframe([{key.title(): row[key] for key in ("phase", "budget", "settled", "pending", "remaining")}
                      for row in block["phases"]], hide_index=True)


def details(block):
    section("Token and cost details")
    estimate = block["list_price_estimate_cny"]
    stats(*[(label, compact(block[key]), f'{number(block[key])} Tokens') for label, key in
            (("Input", "input"), ("Output", "output"), ("Cached input", "cached"), ("Reasoning", "reasoning"))],
          ("GLM-5.3 list-price estimate", f"¥{estimate:,.4f}" if estimate is not None else "Unavailable", None),
          ("Provider invoice", "Unavailable", None))
    st.caption("Cached tokens are a subset of input. Reasoning is a provider-reported detail.")
    st.dataframe([{key.title(): row[key] for key in ("phase", "input", "output", "cached", "reasoning")}
                  for row in block["phases"]], hide_index=True)


def render():
    heading("Token Dashboard", "BigModel account, project budget and recorded usage")
    with st.container(border=True):
        bigmodel_account()
    try:
        block = token_dashboard()
    except (OSError, sqlite3.Error):
        st.error("The token ledger could not be read.")
        return
    with st.container(border=True):
        local_budget(block)
    with st.container(border=True):
        details(block)
