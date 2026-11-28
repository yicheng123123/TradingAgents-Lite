"""Financial reranking, position, and isolation regression coverage."""
from datetime import timedelta

import numpy as np
import pytest

from tradingagents_lite.rag import bm25, rrf
from tradingagents_lite.rag_benchmark import CUTOFF, PARTITION
from tradingagents_lite.retrieval_v6.engine import Search, fused, scope, union
from tradingagents_lite.retrieval_v6.experiment import package
from tradingagents_lite.retrieval_v6.features import (NUMBER, document_features, expansions, features,
                                                    find_metrics, parse, rank, top5)
from tradingagents_lite.retrieval_v6.scoring import literal
from tradingagents_lite.retrieval_v6.structure import header_binding, source_units


def test_metrics_distinguish_related_but_non_equivalent_accounting_concepts():
    assert find_metrics("total revenue, net sales") == ("revenue",)
    assert find_metrics("operating cash flow versus free cash flow") == ("operating cash flow","free cash flow")
    assert find_metrics("Operating income, EBIT, and EBITDA") == ("operating income","ebit","ebitda")
    assert find_metrics("R&D and SG&A expenses") == ("research and development","selling general and administrative")
    assert find_metrics("Income from operations versus net earnings") == ("operating income","net income")


def test_fiscal_periods_come_from_text_and_support_cross_year_comparisons():
    intent=parse("What was growth in net sales year over year between FY2022 and FY2023?")
    assert intent.metrics==("revenue",) and intent.years==("2022","2023")
    assert intent.calculation and intent.comparison=="yoy" and intent.period=="annual"
    old=document_features({"text":"Net sales $837", "title":"FOO_2023_10K","section":"Filings 2022"})
    assert features(intent,old,2,None,None)["Y"]==0
    current=document_features({"text":"Net sales $837 for 2022 and $900 for 2023"})
    assert features(intent,current,2,None,None)["Y"]==1
    quarter=parse("What was revenue in Q1 2024 and the fourth quarter of 2023?")
    assert quarter.quarters==("1","4") and quarter.period=="quarter"


def test_fallback_and_query_expansion_accept_only_original_question():
    no=parse("What was notable?")
    assert expansions("What was notable?", no)==["What was notable?"]
    q="What was capex in FY2023?"
    generated=expansions(q,parse(q))
    assert generated[0]==q and len(generated)<=3
    assert any("property" in x for x in generated)
    assert all("2024" not in x and "operating cash flow" not in x for x in generated)
    assert rank({"a":{"S":0},"b":{"S":2}}, {}, no, 5,"weighted","distinct")==["b","a"]


def layout_line(data):
    label,left,right=data
    return f"{label:<28}{left:>13}{right:>14}"


def test_layout_preserves_full_row_numbers_and_binds_verified_columns():
    header=layout_line(("Years ended","2024","2023"))
    row=layout_line(("Operating cash flow","$(1,234.50)","$405.25"))
    text=f"Consolidated statements of cash flows\nIn USD millions\n{header}\n{row}\n"
    column=header_binding(header,row)
    assert [c["period"] for c in column]==["2024","2023"]
    assert [row[slice(*c["row_columns"])] for c in column]==["$(1,234.50)","$405.25"]
    segment=next(unit for unit in source_units(text) if unit[1]=="row")
    assert text[slice(*segment[0])]==row
    assert segment[2]["binding_status"]=="verified_layout_alignment"
    assert text[slice(*segment[2]["header_span"])]==header
    assert "USD millions" in text[slice(*segment[2]["unit_span"])]
    assert header_binding(header,"Operating cash flow   1   2")==[]


def test_signs_and_delimiters_in_line_proxy():
    assert literal("Cash … $(12.30) USD")!=literal("Cash ... $12.30 USD")
    assert literal("Cash ... −12.30") == literal("Cash ... -12.30")
    assert NUMBER.search("Operating income $(1,234.50)").group()=="$(1,234.50)"


def test_same_page_control_preserves_a_permutation_and_limits_delivery():
    keys=list("abcdefg")
    rows={key:{"document_id":"same","page":i//2} for i,key in enumerate(keys)}
    maps={key:{"S":1-i/10} for i,key in enumerate(keys)}
    q=parse("Revenue?")
    baseline=rank(maps,rows,q)
    representative=rank(maps,rows,q,0,"weighted","distinct")
    assert baseline==keys and representative[:4]==list("aceg")
    assert len(top5(representative))==5 and set(representative)==set(keys)
    assert any(representative.index(k)>3 for k in "bdf")
    with pytest.raises(ValueError,match="permutation"):
        package(keys,keys,keys[:-1],{})


def row(key,text="total revenue",**override):
    return {"chunk_id":key,"text":text,"ticker":"A","dataset_partition":PARTITION,
            "eligible_ts":CUTOFF.timestamp(),"source_type":"sec_filing","pit_level":"historical_proxy",
            "version_id":key,"document_id":key,"page":0,**override}


def test_scope_filters_future_wrong_partition_and_non_matching_company():
    rows=[row("ok"),row("future",eligible_ts=(CUTOFF+timedelta(seconds=1)).timestamp()),
          row("other",ticker="B"),row("private",dataset_partition="synthetic_other"),
          row("unknown",pit_level="unavailable")]
    assert scope(rows,"A")==["ok"]


def test_candidate_dedup_exact_bm25_parity_and_dense_order():
    source=[row("a","total revenue grew"),row("b","inventory and net sales"),row("c","operating cash flow")]
    search=Search({r["chunk_id"]:(r,vector) for r,vector in zip(source,np.eye(3),strict=True)})
    expected=bm25("total revenue and net sales",source,50)
    assert search.lexical("total revenue and net sales","A",50)==expected
    dense=search.dense([1,0,0],"A",50)
    assert dense[0]==("a",1.)
    assert union(expected,dense)==["b","a","c"] or set(union(expected,dense))=={"a","b","c"}
    assert len(union(expected,dense))==3
    shortlist,all_keys=fused([(1,expected),(1,dense)],2)
    assert all_keys==[key for key,_ in rrf([expected,dense])]
    assert shortlist==all_keys[:2]
