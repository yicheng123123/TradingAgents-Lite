"""A1-V5 compression invariants; all fixtures are synthetic and offline."""
from types import SimpleNamespace

import numpy as np
import pytest

from tradingagents_lite import financebench_compression_v5 as v5


class _Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(max(1, len(text.split()))))


def _candidate(index, text, *, doc="SYN_2023_10K", page=7):
    chunk = f"chunk-{index}"
    return {
        "candidate_rank": index + 1, "chunk_hash": f"hash-{index}", "chunk_id": chunk,
        "evidence_id": "chunk:" + chunk, "document_id": doc, "page": page, "text": text,
    }


def test_financial_row_keeps_currency_sign_parentheses_and_year_columns():
    text = "STATEMENT OF OPERATIONS\nUSD millions 2023 2022\nNet income................ $(1,234) $ 900"
    units = v5._units(text, 0)
    row = next(unit for unit in units if unit["table_row"] and "Net income" in unit["text"])
    assert "$(1,234)" in row["text"]
    assert "$ 900" in row["text"]
    assert {"2023", "2022"} <= row["years"]
    assert row["header_status"] == "bound"
    assert "USD millions 2023 2022" in text[slice(*row["header_span"])]


def test_c4_binds_header_to_exact_source_row_and_preserves_offsets(monkeypatch):
    text = "USD millions 2023 2022 Net income 100 90"
    start = text.index("Net income")
    unit = {
        "candidate_index": 0, "start": start, "end": len(text), "text": text[start:],
        "kind": "financial_row", "table_row": True, "metric": "net income",
        "years": {"2023", "2022"}, "header_span": (0, start), "header_status": "bound",
        "vector": np.asarray([1.0, 0.0], dtype=np.float32), "relevance": 1.0,
    }
    monkeypatch.setattr(v5, "ACTIVE", SimpleNamespace(tokenizer=_Tokenizer()))
    result = v5._semantic("net income 2023", [_candidate(0, text)], [unit], 3200,
                          "C4", 0.65, 0.0)
    item, = result["items"]
    assert item["excerpt"] == text
    assert result["source_exact"] is True
    assert result["header_binding"] == {"bound": 1, "attempts": 1, "uncertain": 0}


def test_duplicate_evidence_units_do_not_consume_multiple_slots(monkeypatch):
    text = "Operating income was USD 480 million in 2023."
    candidates = [_candidate(0, text, page=7), _candidate(1, text, page=7)]
    units = []
    for index in range(2):
        units.append({
            "candidate_index": index, "start": 0, "end": len(text), "text": text,
            "kind": "sentence", "table_row": False, "metric": "operating income",
            "years": {"2023"}, "header_span": None, "header_status": "not_required",
            "vector": np.asarray([1.0, 0.0], dtype=np.float32), "relevance": 1.0,
        })
    monkeypatch.setattr(v5, "ACTIVE", SimpleNamespace(tokenizer=_Tokenizer()))
    result = v5._semantic("operating income 2023", candidates, units, 3200, "C2", None, 0.0)
    assert result["selected_evidence_count"] == 1
    assert result["source_exact"] is True


def test_mmr_does_not_penalize_complementary_metrics_or_years():
    base = {"table_row": True, "metric": "net income", "years": {"2023"}}
    different_metric = {"table_row": True, "metric": "operating cash flow", "years": {"2023"}}
    different_year = {"table_row": True, "metric": "net income", "years": {"2022"}}
    duplicate = {"table_row": True, "metric": "net income", "years": {"2023"}}
    assert v5._complementary(base, different_metric)
    assert v5._complementary(base, different_year)
    assert not v5._complementary(base, duplicate)


def test_prefix_and_semantic_outputs_obey_both_context_budgets(monkeypatch):
    sentences = [f"Revenue was USD {i}.0 million for {2023 - i % 4}."
                 for i in range(120)]
    text = " ".join(sentences)
    candidate = _candidate(0, text)
    units = v5._units(text, 0)
    for unit in units:
        unit["vector"] = np.asarray([1.0, 0.0], dtype=np.float32)
        unit["relevance"] = 1.0
    monkeypatch.setattr(v5, "ACTIVE", SimpleNamespace(tokenizer=_Tokenizer()))
    for budget in (3200, 1600):
        prefix = v5._c0("What was revenue in 2023?", [candidate], budget)
        semantic = v5._semantic("What was revenue in 2023?", [candidate], units,
                                 budget, "C2", None, 0.0)
        for result in (prefix, semantic):
            assert result["context_bytes"] <= v5._cap(budget)
            assert result["request_bytes"] <= 8000
            assert result["source_exact"] is True
        assert semantic["selected_evidence_count"] > 0


def test_candidate_scope_enforces_partition_pit_and_provenance():
    payload = {"dataset_partition": v5.PARTITION, "eligible_ts": v5.CUTOFF - 1,
               "pit_level": "historical_proxy", "chunk_id": "c", "document_id": "d",
               "page": 0, "text": "Revenue 10", "ticker": "SYN"}
    v5._validate(payload)
    with pytest.raises(ValueError, match="dataset or point-in-time"):
        v5._validate({**payload, "eligible_ts": v5.CUTOFF + 1})
    with pytest.raises(ValueError, match="dataset or point-in-time"):
        v5._validate({**payload, "dataset_partition": "private_test"})


def test_source_span_audit_rejects_changed_citation_coordinates():
    candidate = _candidate(0, "Revenue 10")
    item = {"evidence_id": candidate["evidence_id"], "document_id": candidate["document_id"],
            "page": candidate["page"], "excerpt": "Revenue 10", "_chunk_id": candidate["chunk_id"],
            "_source_spans": [[0, len(candidate["text"])]], "_chunk_hash": candidate["chunk_hash"]}
    assert v5._exact(item, [candidate])
    assert not v5._exact({**item, "page": candidate["page"] + 1}, [candidate])
