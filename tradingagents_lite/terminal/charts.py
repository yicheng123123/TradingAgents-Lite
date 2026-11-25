"""Charts over recorded observations using Streamlit's Vega-Lite renderer."""
import altair as alt
import pandas as pd
import streamlit as st

from .design import ACCENT, BORDER, MUTED, NEGATIVE, POSITIVE, WARNING


def chart(spec):
    spec = spec.configure(background="transparent").configure_view(stroke=None).configure_axis(
        labelColor=MUTED, titleColor=MUTED, gridColor=BORDER,
        domainColor=BORDER, tickColor=BORDER, labelFontSize=10,
    ).configure_legend(labelColor=MUTED, titleColor=MUTED)
    st.altair_chart(spec, theme=None, width="stretch")


def market_chart(bars, spy, decision_date):
    frame = pd.DataFrame([bar.model_dump(mode="json") for bar in bars])
    frame["MA5"] = frame["close"].rolling(5).mean()
    frame["MA20"] = frame["close"].rolling(20).mean()
    frame["Direction"] = frame.apply(lambda row: "Up" if row.close >= row.open else "Down", axis=1)
    frame = frame.tail(55).copy()
    color = alt.Color("Direction:N", scale=alt.Scale(domain=["Up", "Down"], range=[POSITIVE, NEGATIVE]),
                      legend=None)
    x = alt.X("day:O", axis=alt.Axis(title=None, labels=False, ticks=False))
    base = alt.Chart(frame).encode(x=x, color=color,
        tooltip=[alt.Tooltip("day:N", title="Session"), *[
            alt.Tooltip(f"{key}:Q", title=key.title(), format=",.2f" if key != "volume" else ",")
            for key in ("open", "high", "low", "close", "volume")]])
    scale = alt.Scale(zero=False)
    wick = base.mark_rule().encode(y=alt.Y("low:Q", scale=scale, title="USD"), y2="high:Q")
    candle = base.mark_bar(size=5).encode(y=alt.Y("open:Q", scale=scale, title="USD"), y2="close:Q")
    averages = alt.Chart(frame).transform_fold(["MA5", "MA20"], as_=["Window", "Price"]).mark_line(
        strokeWidth=1.35).encode(x=x, y=alt.Y("Price:Q", scale=scale, title="USD"),
        color=alt.Color("Window:N", scale=alt.Scale(domain=["MA5", "MA20"], range=[ACCENT, WARNING]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["day:O", "Window:N", alt.Tooltip("Price:Q", format=".2f")])
    marker = alt.Chart(pd.DataFrame({"day": [str(decision_date)]})).mark_rule(
        color=MUTED, strokeDash=[4, 3]).encode(x=x, tooltip=[alt.Tooltip("day:N", title="Decision session")])
    prices = alt.layer(wick, candle, averages, marker).resolve_scale(color="independent").properties(height=204)
    labels = frame["day"].iloc[::max(1, len(frame) // 6)].tolist()
    date_axis = alt.X("day:O", axis=alt.Axis(title=None, values=labels, labelAngle=0,
                                         labelExpr="substring(datum.label, 5)"))
    volume = base.mark_bar(size=5).encode(x=date_axis, y=alt.Y("volume:Q", title="Volume",
        axis=alt.Axis(format="~s", tickCount=2))).properties(height=53)
    panels = [prices, volume]
    bench = pd.DataFrame([{"day": str(bar.day), "SPY": bar.close} for bar in spy])
    if not bench.empty:
        paired = frame[["day", "close"]].merge(bench, on="day", how="left")
        common = paired.dropna()
        if not common.empty:
            for symbol, col in (("Asset", "close"), ("SPY", "SPY")):
                paired[symbol] = paired[col] / common.iloc[0][col] * 100
            comparison = alt.Chart(paired).transform_fold(["Asset", "SPY"], as_=["Series", "Index"]).mark_line(
                strokeWidth=1.25).encode(x=date_axis, y=alt.Y("Index:Q", title="Base 100", scale=scale,
                axis=alt.Axis(tickCount=2)), color=alt.Color("Series:N", legend=alt.Legend(title=None, orient="top"),
                scale=alt.Scale(domain=["Asset", "SPY"], range=[ACCENT, MUTED])),
                tooltip=["day:O", "Series:N", alt.Tooltip("Index:Q", format=".2f")]).properties(height=57)
            panels.append(comparison)
    chart(alt.vconcat(*panels, spacing=8).resolve_scale(x="shared", color="independent"))


def bars(rows, label, value, title, color=ACCENT, height=175):
    frame = pd.DataFrame(rows)
    spec = alt.Chart(frame).mark_bar(color=color, cornerRadiusEnd=3).encode(
        x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(tickCount=5)), y=alt.Y(f"{label}:N", title=None, sort=None),
        tooltip=[f"{label}:N", alt.Tooltip(f"{value}:Q", format=",.4~f")]).properties(height=height)
    chart(spec)
