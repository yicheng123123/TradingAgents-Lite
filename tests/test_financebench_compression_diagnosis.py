"""Synthetic regression cases for the isolated Calibration diagnosis and repair."""
import numpy as np
import pytest

from tradingagents_lite import financebench_compression_diagnosis as audit
from tradingagents_lite import financebench_compression_repair as repair
from tradingagents_lite import financebench_compression_v5 as v5


def candidate(text, rank=1, **overrides):
    return {
        "candidate_rank": rank, "chunk_hash": f"h{rank}", "chunk_id": f"c{rank}",
        "evidence_id": f"chunk:c{rank}", "document_id":"fictional-10K", "page":0,
        "ticker":"SYN", "dataset_partition":v5.PARTITION, "eligible_ts":v5.CUTOFF-1,
        "pit_level":"historical_proxy", "text":text, **overrides,
    }


def test_repair_extends_numeric_option_to_complete_row(monkeypatch):
    text = "Years ended 2024 2023 2022 2021\nOperating income $(1,234.75) 98 42 17"
    original = [{"segments":[(6,9)], "score":110, "excerpt":"income $(1,234.75) 98",
                 "selection_reason":"query_phrase_numeric_row"}]
    monkeypatch.setattr(repair, "ORIGINAL_SEGMENTS", lambda *args:original)
    options = repair.repaired_segments("Operating income", "Operating income", text)
    assert options[-1] == original[0]
    assert options[0]["score"] == original[0]["score"]
    assert options[0]["excerpt"] == text


def test_plural_period_and_units_bind_only_when_columns_align():
    prefix = "(USD thousands) Years ended December 31, 2024 December 31, 2023 "
    text = prefix + "Operating income (1,234) 512"
    span = audit.period_span(text,len(prefix),2)
    assert text[slice(*span)].strip() == prefix.strip(" ()")
    assert audit.period_span(text,len(prefix),3) is None
    changed = prefix + "other metric 17 "
    assert audit.period_span(changed+"Operating income 5 6",len(changed),2) is None


def test_year_and_value_matching_preserve_numeric_boundaries_and_sign():
    source = "2024 Operating income… $(1,234.25) 2.50"
    needle = "Operating   income....... $ (1,234.25) 2.50"
    start,end = audit.row_spans(needle,source)[0]
    assert source[start:end] == "Operating income… $(1,234.25) 2.50"
    assert audit.row_spans("Operating income 1,234.25",source) == []
    assert audit.row_spans("2.5",source) == []
    assert repair.number_key("(1,234.25)") == repair.number_key("−1234.25")
    assert repair.number_key("-1234.25") != repair.number_key("1234.25")


def test_disjoint_spans_are_not_stitched_into_a_financial_row():
    case = {"candidates":[candidate("Revenue 82 91 net profit 15 14")]}
    assert not audit.cover({0:[[0,7],[8,13]]},0,0,13)
    item = v5._render(case["candidates"],{0:[(0,7),(8,13)]})[0]
    assert v5._exact(item,case["candidates"])
    assert not v5._exact({**item,"page":1},case["candidates"])


def test_repair_preserves_wire_limits_spans_and_citation_scope():
    block = "Years ended 2024 2023\nOperating income (1,234.25) 520\n"
    items = [candidate(block+("other disclosed metric 145.0 221.0 "*60),i+1,page=i) for i in range(5)]
    for b in v5.BUDGETS:
        with repair.use_repair(True):
            result = v5._c1("Operating income 2024",items,b)
        assert result["items"] and result["source_exact"]
        assert result["request_bytes"] <= v5.MAX_REQUEST
        assert result["context_bytes"] <= v5._cap(b)
        for item in result["items"]:
            orig = next(c for c in items if c["chunk_id"] == item["_chunk_id"])
            assert item["evidence_id"] == orig["evidence_id"]
            assert item["page"] == orig["page"]
            assert all(0 <= s < e <= len(orig["text"]) for s,e in item["_source_spans"])


@pytest.mark.parametrize("overrides", [
    {"eligible_ts": v5.CUTOFF+1}, {"dataset_partition": "other"},
    {"pit_level":"unverified"},
])
def test_verified_inputs_refuse_future_and_out_of_scope_candidates(tmp_path, overrides):
    case = {"case_id":"synthetic", "candidates":[candidate("Revenue 17",i+1) for i in range(5)]}
    cases = [{**case, "case_id":str(i)} for i in range(49)]
    cases[0]["candidates"][0] = candidate("Revenue 17",**overrides)
    inp, labels = tmp_path/"calibration-inputs.jsonl", tmp_path/"calibration-scoring-labels.jsonl"
    v5.writerows(inp,cases)
    labels.write_text("", encoding="utf-8")
    freeze = {"calibration":{"inputs_sha256":v5.digest(inp), "labels_sha256":v5.digest(labels)}}
    with pytest.raises(ValueError, match="time scope"):
        audit.validated_cases(tmp_path,freeze)


def test_greedy_replay_accounts_for_per_chunk_share():
    text = "Operating margin 18% 14% " + "footnotes and assumptions "*17
    chunk = candidate(text)
    unit = v5._unit(text,0,0,len(text),"financial_row",True)
    unit["vector"] = np.array([1,0],dtype=np.float32)
    unit["relevance"] = 1
    docs = [chunk]+[candidate("other results",i+2) for i in range(4)]
    items, chosen, events = audit.trace_semantic("Operating margin",docs,[unit],3200,"C2",None)
    assert not items and not chosen
    assert events[0] == {"per_chunk_share"}


def test_reported_binding_events_track_all_units_before_selection():
    text = "USD millions 2024 2023 Net income ...... 12 10"
    units = [u for u in v5._units(text,0) if u["table_row"]]
    counts = audit.binding_decisions(text,units)
    assert sum(counts[k] for k in ("local_bind_attempt","inherited_binding","no_nearby_v5_cue")) == len(units)
    assert any(u["header_status"] == "bound" for u in units)


def test_reference_dates_and_section_titles_are_not_financial_rows():
    assert audit.nonfinancial_target("november 27,") == "date_fragment"
    assert audit.nonfinancial_target("years ended december 31, 2022 and 2021") == "date_fragment"
    assert audit.nonfinancial_target("item 8. financial statements and supplementary data") == "section_heading"
    assert audit.nonfinancial_target("Revenue for December 31, 2022 was $2.1 million") is None
