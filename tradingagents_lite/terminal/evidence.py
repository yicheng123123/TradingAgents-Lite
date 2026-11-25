"""Document and retrieval provenance for the current research run."""
from urllib.parse import urlsplit

import streamlit as st

from ..data import close_available
from .archive import instant, task_of
from .design import empty, heading, html, kv, number, safe, section


def source(item):
    return item.get("source_type") or "Market snapshot"


def ranks(row, evidence_id, item):
    chunk = item.get("chunk_id") or evidence_id.removeprefix("chunk:")
    found = []
    for tool in row.get("tool_trace") or []:
        stages = (tool.get("retrieval_trace") or {}).get("stages") or {}
        for stage, candidates in stages.items():
            for candidate in candidates:
                if candidate.get("id") == chunk:
                    found.append({"Stage": stage.upper(), "Rank": candidate.get("rank"),
                                  "Score": candidate.get("score"), "Tool call": tool.get("call_id")})
    return found


def allowed(item, cutoff):
    timestamp = item.get("eligible_at")
    if timestamp:
        parsed = instant(timestamp)
        return parsed is not None and parsed <= cutoff and item.get("pit_level") != "unknown"
    return source(item) == "Market snapshot"


def render(row):
    heading("Evidence", "Current-run documents and retrieval provenance")
    if not row:
        empty("No run loaded", "Open a research run to inspect its current evidence.")
        return
    cutoff = close_available(task_of(row).decision_date)
    evidence = {key: item for key, item in (row.get("evidence") or {}).items() if allowed(item, cutoff)}
    if not evidence:
        empty("Evidence unavailable", "The selected archive has no eligible evidence records.")
        return
    left, middle, right = st.columns([1.15, 2.35, 1.35])
    with left, st.container(border=True):
        section("Evidence library", number(len(evidence)))
        sources = st.multiselect("Source type", sorted({source(item) for item in evidence.values()}))
        levels = st.multiselect("PIT level", sorted({item.get("pit_level") or "Unavailable" for item in evidence.values()}))
        filtered = {key: item for key, item in evidence.items()
                    if (not sources or source(item) in sources)
                    and (not levels or (item.get("pit_level") or "Unavailable") in levels)}
        if not filtered:
            empty("No matching evidence", "Adjust the source or PIT filters.")
            return

        def title(key):
            item = filtered[key]
            label = item.get("metric") or item.get("title") or source(item)
            page = f' · p. {item["page"]}' if item.get("page") is not None else ""
            return f'{label}{page} · {key.split(":")[-1][:8]}' if source(item) != "Market snapshot" else key

        selected = st.radio("Inspect evidence", list(filtered), format_func=title,
                            key=f"evidence_{row['run_id']}")
        st.caption(f"{len(filtered)} of {len(evidence)} eligible records")
        with st.expander("Retrieval scores", expanded=True):
            for key, item in filtered.items():
                ranking = ranks(row, key, item)
                if ranking:
                    kv([(title(key), " · ".join(f'{entry["Stage"]} {entry["Score"]:,.3f}'
                        for entry in ranking if entry["Score"] is not None))])
            if not any(ranks(row, key, item) for key, item in filtered.items()):
                st.caption("No scores available for these records.")
    record = evidence[selected]
    with middle, st.container(border=True):
        section("Document reader", "Eligible", "good")
        st.markdown(f'**{title(selected)}**')
        url = record.get("source_url")
        if url and urlsplit(url).scheme in {"http", "https"} and urlsplit(url).hostname:
            st.link_button("Open original source ↗", url)
        if record.get("excerpt"):
            html(f'<div class="document">{safe(record["excerpt"])}</div>')
        elif record.get("bars"):
            st.dataframe(record["bars"], hide_index=True)
        elif record.get("metric"):
            kv([("Period end", record.get("period_end")), ("Value", record.get("value")),
                ("Unit", record.get("unit"))])
            if record.get("inputs"):
                st.dataframe(record["inputs"], hide_index=True)
        elif source(record) == "Market snapshot":
            kv([(key.replace("_", " ").title(), value) for key, value in record.items() if key != "evidence_id"])
        else:
            empty("Document content unavailable", "The archive contains provenance for this record.")
    with right, st.container(border=True):
        section("Provenance")
        kv([("Evidence ID", selected), ("Document ID", record.get("document_id")),
            ("Chunk ID", record.get("chunk_id") or (selected.removeprefix("chunk:") if selected.startswith("chunk:") else None)),
            ("Page", record.get("page")), ("Eligible at", record.get("eligible_at")),
            ("PIT level", record.get("pit_level"))])
        section("Retrieval ranks")
        ranking = ranks(row, selected, record)
        kv([(stage, ", ".join(f'#{entry["Rank"]}' for entry in ranking if entry["Stage"] == stage) or None)
            for stage in ("BM25", "DENSE", "RRF", "RERANK")])
        if ranking:
            with st.expander("Rank and score history"):
                st.dataframe(ranking, hide_index=True)
