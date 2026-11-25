"""Shared presentation primitives."""
from html import escape
from pathlib import Path

import streamlit as st


ACCENT = "#38BDF8"
POSITIVE = "#34D399"
NEGATIVE = "#F87171"
WARNING = "#FBBF24"
MUTED = "#94A3B8"
BORDER = "#273244"
UNAVAILABLE = "Unavailable"


def html(value):
    st.markdown(value, unsafe_allow_html=True)


def install():
    html("<style>" + Path(__file__).with_name("theme.css").read_text(encoding="utf-8") + "</style>")


def show(value):
    if value is None or value == "":
        return UNAVAILABLE
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value)) or "None"
    return str(value)


def safe(value):
    return escape(show(value))


def number(value):
    return f"{value:,}" if value is not None else UNAVAILABLE


def compact(value):
    if value is not None:
        for scale, unit in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
            if abs(value) >= scale:
                return f"{value / scale:.2f}{unit}"
    return number(value)


def percent(value, signed=False):
    return format(value, "+.2%" if signed else ".2%") if value is not None else UNAVAILABLE


def ms(value):
    return f"{value:,.2f} ms" if value is not None else UNAVAILABLE


def badge(value, tone=""):
    return f'<span class="badge {tone}">{safe(value)}</span>'


def heading(title, detail):
    html(f'<header class="pagehead"><h1>{safe(title)}</h1><p>{safe(detail)}</p></header>')


def section(title, label=None, tone=""):
    flag = badge(label, tone) if label else ""
    html(f'<div class="section"><h2>{safe(title)}</h2>{flag}</div>')


def empty(title, detail):
    html(f'<div class="empty"><strong>{safe(title)}</strong>{safe(detail)}</div>')


def stats(*entries):
    parts = []
    for label, value, detail in entries:
        extra = f"<small>{safe(detail)}</small>" if detail else ""
        parts.append(f'<div class="stat"><span class="eyebrow">{safe(label)}</span>'
                     f'<strong>{safe(value)}</strong>{extra}</div>')
    html('<div class="statgrid">' + "".join(parts) + "</div>")


def kv(entries):
    html('<dl class="kv">' + "".join(f"<div><dt>{safe(key)}</dt><dd>{safe(value)}</dd></div>"
                                    for key, value in entries) + "</dl>")


def flow(entries):
    html('<div class="flow">' + '<i aria-hidden="true">→</i>'.join(
        badge(label, tone) for label, tone in entries) + "</div>")


def timeline(entries):
    html('<div class="timeline">' + "".join(f"<div><strong>{safe(label)}</strong>"
                                           f"<small>{safe(detail)}</small></div>"
                                           for label, detail in entries) + "</div>")
