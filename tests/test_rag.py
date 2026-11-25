from datetime import UTC, datetime, timedelta
import json
from uuid import uuid4

import pytest

from test_stage1 import TASK, bars
from tradingagents_lite.data import MarketView, close_available
from tradingagents_lite.documents import Document, chunk_document, digest, parse_document
from tradingagents_lite.rag import RagConfig, RagView, Retriever, rrf
from tradingagents_lite.schema import Budget, MarketReport
from tradingagents_lite.vectors import VectorStore
from tradingagents_lite.workflow import MomentumBackend, build_workflow


class SmallEncoder:
    metadata = {"name": "test", "revision": "test", "dimension": 2, "normalization": "none"}

    def encode(self, texts, *, query=False):
        return [[1., 0.] if "revenue" in text else [0., 1.] for text in texts]


def document(raw, **kwargs):
    return Document.model_validate({
        "document_id": "0001", "ticker": TASK.symbol, "source_type": "sec_filing",
        "source_url": "https://example.test/filing", "eligible_at": close_available(TASK.decision_date),
        "pit_level": "historical_proxy", "dataset_partition": "development", "raw_hash": digest(raw),
        "format": "html", "use_basis": "test fixture", **kwargs})


def test_html_structure_and_fixed_chunk_ids():
    raw = b"<h1>Risk</h1><p>one two three</p><h1>Revenue</h1><table><tr><td>Sales</td><td>100</td></tr></table>"
    doc = document(raw)
    blocks = parse_document(doc, raw)
    assert [b.section for b in blocks] == ["Risk", "Risk", "Revenue", "Revenue"]
    assert blocks[-1].text == "Sales 100"
    structured = chunk_document(doc, blocks, size=4, overlap=1)
    assert structured == chunk_document(doc, blocks, size=4, overlap=1)
    assert structured[0]["start"] == ["html:0", 0]
    fixed = chunk_document(doc, blocks, size=4, overlap=1, strategy="fixed")
    assert set(r["chunk_id"] for r in fixed).isdisjoint(r["chunk_id"] for r in structured)
    with pytest.raises(ValueError, match="frozen hash"):
        parse_document(doc, raw + b"x")


def test_pdf_with_empty_open_password_preserves_page_indices():
    from io import BytesIO

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("", algorithm="AES-256")
    stream = BytesIO()
    writer.write(stream)
    raw = stream.getvalue()
    blocks = parse_document(document(raw, format="pdf"), raw)
    assert [block.page for block in blocks] == [0, 1]
    assert chunk_document(document(raw, format="pdf"), blocks) == []


def test_rrf_uses_rank_and_deterministic_ties():
    got = rrf([[("a", 1), ("b", 0)], [("b", 100), ("c", 90)]])
    assert [key for key, _ in got] == ["b", "a", "c"]
    assert got[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_real_qdrant_restart_and_candidate_filters(tmp_path):
    cutoff = close_available(TASK.decision_date)
    candidates = [
        ("valid", "revenue", {}),
        ("future", "revenue", {"eligible_at": cutoff + timedelta(days=1)}),
        ("holdout", "revenue", {"dataset_partition": "holdout"}),
        ("other", "revenue", {"ticker": "ZZZZ"}),
        ("conflict", "revenue", {}),
        ("conflict", "revenue and sales", {}),
        ("unavailable", "revenue", {"pit_level": "unavailable"}),
    ]
    rows = []
    for key, text, overrides in candidates:
        raw = f"<p>{text}</p>".encode()
        doc = document(raw, document_id=key, **overrides)
        rows.extend(chunk_document(doc, parse_document(doc, raw)))
    with VectorStore(tmp_path / "q", SmallEncoder(), "chunks") as store:
        store.put(rows, id_key="chunk_id", text_key="text")
    with VectorStore(tmp_path / "q", SmallEncoder(), "chunks") as store:
        r = Retriever(store)
        for mode in ("R1", "R2", "R3"):
            result = r.retrieve("revenue", ticker=TASK.symbol, cutoff=cutoff,
                partition="development", mode=mode, top_k=1, candidate_k=1)
            assert [item["document_id"] for item in result["items"]] == ["valid"]
            assert result["eligible_chunks"] == 1


def test_narrowed_pit_scope_cannot_revive_conflicting_proxy(tmp_path):
    cutoff = close_available(TASK.decision_date)
    verified_raw = b"<p>revenue attested</p>"
    proxy_raw = b"<p>revenue unverified</p>"
    verified = document(verified_raw, pit_level="verified_known_at", first_seen_at=cutoff,
                        version_attestation_sha256=digest(verified_raw),
                        version_attestation_source="fixture")
    proxy = document(proxy_raw)
    with VectorStore(tmp_path / "q", SmallEncoder(), "chunks") as store:
        for doc, raw in [(verified, verified_raw), (proxy, proxy_raw)]:
            store.put(chunk_document(doc, parse_document(doc, raw)), id_key="chunk_id", text_key="text")
        retriever = Retriever(store)
        args = {"ticker": TASK.symbol, "cutoff": cutoff, "partition": "development",
                "mode": "R3", "top_k": 1, "candidate_k": 1}
        assert retriever.retrieve("revenue", **args)["items"][0]["pit_level"] == "verified_known_at"
        assert retriever.retrieve("revenue", pit=("historical_proxy",), **args)["items"] == []


@pytest.mark.parametrize("failure", [True, False])
def test_agent_rag_failure_and_citation_gate(failure):
    class Store:
        def retrieve(self, *args, **kwargs):
            if failure:
                raise TimeoutError
            return {"items": [{"chunk_id": "seen", "source_type": "news", "page": None,
                     "eligible_at": "2024-01-01T00:00:00Z", "pit_level": "historical_proxy",
                     "document_id": "doc", "text": "retrieved"}], "stages": {}}

    class Agent(MomentumBackend):
        def analyst(self, messages, tools):
            if not any(m["role"] == "tool" for m in messages):
                assert any(t["function"]["name"] == "rag_search" for t in tools)
                return {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "x", "type": "function", "function": {
                        "name": "rag_search", "arguments": json.dumps({"query": "cash flow"})}}]}
            return {"content": MarketReport(view="claim", evidence_ids=["chunk:unseen"]).model_dump_json()}

    result = build_workflow(MarketView(TASK, bars()), Agent(Budget()), Budget(),
        rag=RagView(Store(), TASK, "development", RagConfig())).invoke({})
    assert result["errors"][0]["code"] == "invalid_citation"
    assert result["tool_trace"][0]["status"] == ("error" if failure else "ok")
    if not failure:
        assert list(result["evidence"]) == ["chunk:seen"]
