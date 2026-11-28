"""Offline harness and known-scope runtime smoke with HTTP-free provider doubles."""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace

from ..memory import MemoryStore
from ..token_budget import TokenBudget
from ..workflow import MomentumBackend
from .data import BUDGET, OUT, WORK, atomic, load, unpack
from .retrieval import QUERY, PilotRag
from .run import EXPERIMENT, MODEL, PHASE, RecallOnly, load_search, predict, provider, read_packets


def identities(value):
    if isinstance(value, dict):
        return ([value["evidence_id"]] if "evidence_id" in value else []) + [
            key for item in value.values() for key in identities(item)]
    if isinstance(value, list):
        return [key for item in value for key in identities(item)]
    return []


class MockEndpoint:
    def __init__(self, malformed=False, invalid_citation=False):
        self.malformed=malformed
        self.invalid_citation=invalid_citation
        self.calls=[]
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.calls.append(request)
        system=request["messages"][0]["content"]
        objects=[]
        for msg in request["messages"][1:]:
            try:
                value=json.loads(msg.get("content") or "")
            except ValueError:
                continue
            objects.append(value)
        messages=SimpleNamespace(tool_calls=[],content="")
        finish="stop"
        if request.get("tools"):
            messages.tool_calls=[SimpleNamespace(id="mock-indicators",type="function",
                        function=SimpleNamespace(name="market_indicators",arguments="{}"))]
            finish="tool_calls"
        elif "Provide an independent argument" in system:
            inputs=objects[-1]
            ids=inputs["report"]["evidence_ids"]
            side=inputs["stance"]
            messages.content=json.dumps({"stance":side,"action":"buy" if side=="bull" else "sell",
                "claims":[{"kind":"inference","statement":"Synthetic workflow test only.","evidence_ids":ids[:1]}]})
        elif "MarketReport" in system or "Market Analyst" in system or "historical task" in system:
            ids=list(dict.fromkeys(key for value in objects for key in identities(value)))
            if self.malformed:
                self.malformed=False
                messages.content="not json"
            elif self.invalid_citation:
                self.invalid_citation=False
                messages.content=json.dumps({"view":"Malformed reference","evidence_ids":["nonexistent-citation"]})
            else:
                messages.content=json.dumps({"view":"Offline mock retrieval and volatility check.",
                                             "evidence_ids":ids[-2:]})
        else:
            ids=objects[0]["report"]["evidence_ids"]
            messages.content=json.dumps({"action":"buy","rationale":"Synthetic workflow test only.","evidence_ids":ids[:1]})
        return SimpleNamespace(id=f"offline-{len(self.calls)}",model=MODEL,
                  choices=[SimpleNamespace(message=messages,finish_reason=finish)],
                  usage=SimpleNamespace(prompt_tokens=160,completion_tokens=30))


def mock_backend(ledger,key,**options):
    backend=provider(MockEndpoint(**options),ledger)
    backend.set_budget_context(experiment_id="offline-"+EXPERIMENT,task_id=key,phase=PHASE)
    return backend


def offline(work=WORK,out=OUT,device="cuda"):
    data=read_packets(work,out)
    manifest=load(out/"pilot-manifest.json")
    memory_coverage=load(out/"memory-coverage.json")["candidate_counts"]
    main=manifest["smoke_tasks"][0]
    cold=next(key for key in manifest["execution_order"]["T0"] if not memory_coverage[key])
    ledger=TokenBudget(work/"offline-budget.sqlite",experiment_limit=1_000_000)
    fallback=[]
    config={"retriever":"S3","context":"C1-segments"}
    with ExitStack() as resources:
        try:
            emb, search=load_search(resources,work,config,device)
            task=unpack(data[main])[0].task
            check=PilotRag(search,task,"S3","C1-segments").search(QUERY)
            if check["retrieval_trace"]["reranker_ms"]>8000 or check["retrieval_trace"]["pool"]==0:
                raise RuntimeError("S3 resource or compatibility gate failed")
        except (ValueError,RuntimeError,OSError,ImportError) as exc:
            fallback.append(type(exc).__name__)
            resources.close()
            config["retriever"]="R3"
            emb,search=load_search(resources,work,config,device)
        trial=PilotRag(search,unpack(data[main])[0].task,config["retriever"],"C1-segments").search(QUERY)
        if not trial["items"]:
            config["context"]="legacy-prefix"
        memory={arm:RecallOnly(resources.enter_context(MemoryStore(work/f"memory-{arm}",emb)),arm)
                for arm in ("T3","T4")}
        rows={}
        for arm in ("T0","T1","T2","T3","T4"):
            backend=MomentumBackend(BUDGET) if arm=="T0" else mock_backend(ledger,arm+":"+main)
            rows[arm]=predict(arm,data[main],backend,search,memory.get(arm),config)
        nopath=predict("T4",data[cold],mock_backend(ledger,"cold"),search,memory["T4"],config)
        malformed=predict("T1",data[main],mock_backend(ledger,"invalid",malformed=True),search,None,config)
        citations=predict("T1",data[main],mock_backend(ledger,"citation",invalid_citation=True),search,None,config)
        blocked=TokenBudget(work/"offline-denied.sqlite",experiment_limit=1)
        denial=predict("T1",data[main],mock_backend(blocked,"denied"),search,None,config)
        failed=[arm for arm,row in rows.items() if row["status"]!="ok"]
        facts={
            "all_arms_ok":not failed,"fixed_rag_delivered":any((r.get("retrieval_trace") or {}).get("selected_count")
                for r in rows["T2"]["tool_trace"]),
            "memory_injected":bool(rows["T3"]["memory"] and rows["T3"]["memory"]["passed_to_backend"]),
            "cold_memory_empty":bool(nopath["memory"] is not None and not nopath["memory"].get("context")),
            "debate_triggered":rows["T4"]["role_calls"]==2,
            "debate_not_triggered":nopath["role_calls"]==0,
            "structured_repair_ok":malformed["status"]=="ok" and any(r["attempt"]=="repair" for r in malformed["structured_trace"]),
            "citation_repair_ok":citations["status"]=="ok" and any(r["attempt"]=="repair" for r in citations["structured_trace"]),
            "budget_refused":denial["system_fault"]=="token_budget_exhausted",
        }
    atomic(work/"offline-smoke-debug.json",{"arms":rows,"cold":nopath,"malformed":malformed,
           "citation":citations,"denied":denial})
    summary={**config,"passed":all(facts.values()),"checks":facts,"fallback":fallback,
             "main_task":main,"cold_task":cold,"real_glm_requests":0,"real_glm_tokens":0}
    atomic(work/"offline-smoke.json",summary)
    print(summary,flush=True)
