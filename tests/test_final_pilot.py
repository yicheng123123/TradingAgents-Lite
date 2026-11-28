"""Offline tests of real pilot integration with a fake HTTP completion endpoint."""
from contextlib import closing
from datetime import timedelta
import json

import pytest

from test_stage1 import TASK, bars
from tradingagents_lite.data import close_available, read_tasks
from tradingagents_lite.documents import Document, chunk_document, parse_document, digest
from tradingagents_lite.final_pilot.data import PARTITION, TASKS, atomic, select_tasks, unpack
from tradingagents_lite.final_pilot.retrieval import PilotRag
from tradingagents_lite.final_pilot.run import EXPERIMENT, PHASE, predict, record_db, reconcile
from tradingagents_lite.final_pilot.smoke import mock_backend
from tradingagents_lite.token_budget import TokenBudget


@pytest.fixture
def packet():
    return {"task":TASK.model_dump(mode="json"),
            "bars":[bar.model_dump(mode="json") for bar in bars() if bar.day<=TASK.decision_date],
            "evidence":[],"xbrl":{"filings":{},"facts":[],"content_hash":""}}


def pilot_config():
    return {"retriever":"S3","context":"C1-segments"}


def test_pilot_sampler_is_one_per_stock_stable_and_chronological():
    rows=read_tasks(TASKS/"development.jsonl")
    selected=select_tasks(rows)
    assert selected==select_tasks(reversed_list:=list(reversed(rows)))
    assert len(reversed_list)==80 and len(selected)==len({row.symbol for row in selected})==20
    assert len({row.task_id for row in selected})==20
    assert selected==sorted(selected,key=lambda row:row.decision_date)


def test_packet_refuses_future_and_does_not_expose_settlement(packet):
    view,_,_=unpack(packet)
    assert not hasattr(view,"outcome")
    day=TASK.decision_date+timedelta(days=1)
    leaked={**packet["bars"][-1],"day":str(day),"available_at":close_available(day).isoformat()}
    with pytest.raises(ValueError,match="future"):
        unpack({**packet,"bars":[*packet["bars"],leaked]})


class RankedRows:
    def __init__(self,task,empty=False):
        content=b"Revenue in 2024 was USD (12.5) million and financial results changed."
        doc=Document(document_id="synthetic",ticker=task.symbol,source_type="news",
            source_url="https://example.com/source",eligible_at=close_available(task.decision_date),
            pit_level="historical_proxy",dataset_partition=PARTITION,raw_hash=digest(content),
            format="headline",use_basis="synthetic",title="Revenue")
        self.rows=[] if empty else chunk_document(doc,parse_document(doc,content))

    def search(self,query,task):
        return self.rows,{"pool":len(self.rows),"reranker_ms":0.,"latency_ms":0.}


def test_extract_preserves_reference_boundaries_with_empty_result():
    ranked=RankedRows(TASK)
    result=PilotRag(ranked,TASK,"S3","C1-segments").search("financial results revenue")
    item,=result["items"]
    assert "(12.5)" in item["excerpt"]
    assert item["start"]==ranked.rows[0]["start"] and item["evidence_id"]=="chunk:"+ranked.rows[0]["chunk_id"]
    assert item["token_spans"]
    empty=PilotRag(RankedRows(TASK,empty=True),TASK,"S3","C1-segments").search("financial results revenue")
    assert empty["status"]=="unavailable" and empty["items"]==[]


def test_flash_trader_receives_real_context_shape_and_no_historical_citations(packet,tmp_path):
    ledger=TokenBudget(tmp_path/"mock.sqlite")
    backend=mock_backend(ledger,"T3")
    class FrozenMemory:
        def before(self,task,query):
            return {"context":[{"memory_id":"history-1","historical_evidence_ids":["old-only"],"lesson":"history"}],
                    "injected_ids":["history-1"],"latency_ms":0,"eligible":1}
    result=predict("T3",packet,backend,RankedRows(TASK),FrozenMemory(),pilot_config())
    assert result["status"]=="ok" and result["memory"]["passed_to_backend"]
    assert all(row["phase"]==PHASE for row in ledger.attempts())
    trader=[call for call in backend.client.calls if "Allowed actions are buy" in call["messages"][0]["content"]
            or "Actions: buy, sell" in call["messages"][0]["content"]]
    inputs=json.loads(trader[-1]["messages"][1]["content"])
    assert inputs["historical_experience"][0]["memory_id"]=="history-1"
    assert "old-only" not in inputs["evidence"] and "old-only" not in result["decision"]["evidence_ids"]


def test_recovery_retries_a_tool_timeout_and_schema_repair(packet,tmp_path):
    ledger=TokenBudget(tmp_path/"mock.sqlite")
    observed=[]
    def faults(runtime,original,name,raw,local,call_id):
        def test_call(tool,args):
            if tool=="market_indicators":
                observed.append(tool)
                if len(observed)==1:
                    return {"error":"tool_timeout"}
            return local(tool,args)
        return original(name,raw,test_call,call_id)
    backend=mock_backend(ledger,"repair",malformed=True)
    result=predict("T1",packet,backend,None,None,pilot_config(),faults=faults)
    assert result["status"]=="ok" and len(observed)==2
    assert any(row["recovery_status"]=="recovered" for row in result["agent_trace"])
    assert any(row["attempt"]=="repair" and row["valid"] for row in result["structured_trace"])


def test_resume_stops_unmatched_attempt_and_preserves_finished_cell(tmp_path):
    ledger=TokenBudget(tmp_path/"ledger.sqlite")
    with closing(record_db(tmp_path/"prediction.sqlite")) as db, db:
        db.execute("INSERT INTO cells VALUES (?,?,?)",("T0:case",json.dumps({"calls":[]}),"frozen"))
    with closing(record_db(tmp_path/"prediction.sqlite")) as db:
        assert "T0:case" in reconcile(ledger,db,"frozen")
        ledger.reserve(experiment_id=EXPERIMENT,task_id="T1:case",phase=PHASE,model="mock",
                       logical_call=1,attempt_number=1,request_sha256="x",request_bytes=100,max_output_tokens=100)
        with pytest.raises(ValueError,match="Unmatched"):
            reconcile(ledger,db,"frozen")
