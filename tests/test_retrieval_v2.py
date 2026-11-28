from types import SimpleNamespace

import pytest

from tradingagents_lite.rag import rrf
from tradingagents_lite.retrieval_v2 import BASE, GRID, Ranking, bootstrap, fuse, percentiles, score
from tradingagents_lite.retrieval_v2_chunks import table_blocks


def metadata():
    return {"a": {"document_id": "annual", "page": 0},
            "b": {"document_id": "annual", "page": 0},
            "c": {"document_id": "quarterly", "page": 0},
            "d": {"document_id": "annual", "page": 1}}


def test_equal_rrf_matches_legacy_and_page_quota_keeps_document_identity():
    bm = [("a", 100), ("b", 90), ("c", 80), ("d", 70)]
    dense = [("a", .9), ("b", .8), ("c", .7), ("d", .6)]
    rows = metadata()
    assert fuse(bm, dense, rows, BASE) == [key for key, _ in rrf([bm, dense])]
    assert fuse(bm, dense, rows, Ranking(per_page=1)) == ["a", "c", "d"]
    assert fuse(bm, dense, rows, Ranking(per_page=2)) == ["a", "b", "c", "d"]
    assert len(GRID) == len(set(GRID)) == 81


def test_weighting_uses_rank_and_stable_ties():
    rows = metadata()
    bm, dense = [("a", 900)], [("b", .01)]
    assert fuse(bm, dense, rows, BASE) == ["a", "b"]
    assert fuse(bm, dense, rows, Ranking(dense_weight=.65)) == ["b", "a"]
    assert fuse(bm, dense, rows, Ranking(per_page=1)) == ["a"]


def test_page_quota_does_not_expand_top_five_and_missing_gold_stays_in_denominator():
    rows = {str(i): {"document_id": "annual", "page": i} for i in range(12)}
    bm = [(str(i), float(12 - i)) for i in range(12)]
    results = score(bm, [], rows, {("annual", 4), ("annual", 5), ("missing", 4)}, Ranking(per_page=1))
    assert results["page"]["recall@5"] == 1 / 3
    assert results["page"]["recall@10"] == 2 / 3
    assert results["candidate_page_recall"] == 2 / 3
    assert results["indexed_gold_recall"] == 2 / 3
    assert results["returned_chunks@5"] == 5
    assert results["unique_pages@5"] == 5


def test_company_bootstrap_keeps_full_clusters_and_reports_paired_changes():
    def row(k, company, score):
        return {"id": k, "company_group": company, "page": {"recall@5": score}}
    before = [row("1", "a", 0), row("2", "a", 0), row("3", "b", 1)]
    after = [row("1", "a", 1), row("2", "a", 1), row("3", "b", 0)]
    stats = bootstrap(after, before)
    assert stats["mean_delta"] == 1 / 3
    assert stats["ci95"] == [-1, 1]
    assert (stats["better"], stats["worse"], stats["tied"]) == (2, 1, 0)
    with pytest.raises(ValueError):
        bootstrap(list(reversed(after)), before)
    assert percentiles([0, 10, 20]) == {"mean": 10, "p50": 10, "p95": 19}


def test_table_preserves_complete_source_row_and_locators():
    row = "Operating cash flow                   899          741"
    noise = ("\n" + "word " * 16) * 14
    layout = "CONSOLIDATED STATEMENTS OF CASH FLOWS\n(in millions)\n2024                 2023" + noise + "\n" + row
    page = SimpleNamespace(extract_text=lambda **_: layout)
    reader = SimpleNamespace(pages=[page, page])
    doc = SimpleNamespace(version_id="variant", title="heading", eligible_at=SimpleNamespace(timestamp=lambda: 0),
                          model_dump=lambda **_: {"document_id": "test-doc", "ticker": "TEST"})
    chunks = list(table_blocks(doc, reader))
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert {c["page"] for c in chunks} == {0, 1}
    assert len(chunks) > 2
    assert all(c["document_id"] == "test-doc" and c["version_id"] == "variant" for c in chunks)
    assert all(c["start"][0].startswith(f"pypdf-layout:{c['page']}:line:") for c in chunks)
    assert {c["page"] for c in chunks if "Operating cash flow 899 741" in c["parsed_body"]} == {0, 1}
    assert all(len(c["text"].split()) <= 220 for c in chunks)


def test_formal_rejects_freeze_mismatch_before_loading_test_labels(tmp_path, monkeypatch):
    from tradingagents_lite import retrieval_v2

    retrieval_v2.write_json(tmp_path / "freeze.json", {"inputs": {"data_sha256": "frozen"}})
    monkeypatch.setattr(retrieval_v2, "inputs", lambda _: {"data_sha256": "changed"})
    monkeypatch.setattr(retrieval_v2, "load_split", lambda *_: pytest.fail("test labels read before verification"))
    with pytest.raises(ValueError, match="Frozen inputs"):
        retrieval_v2.formal(tmp_path, tmp_path)
    assert not (tmp_path / "test").exists()
