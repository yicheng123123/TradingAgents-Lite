"""Separate settlement and de-identified Pilot summaries."""
from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256 as hash_bytes
import json
import math
from random import Random
from statistics import mean,median

from ..data import close_available,sha256
from ..schema import Action,Task
from .data import DevelopmentMarket,OUT,PUBLIC,WORK,atomic,load
from .run import EXPERIMENT,PHASE,open_budget,previous_smoke,reconcile,record_db

ARMS=("T0","T1","T2","T3","T4")
PAIRS=(("T1","T0"),("T2","T1"),("T3","T2"),("T4","T3"),("T4","T1"))


def fingerprint(value):
    return hash_bytes(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def percentile(values,p):
    return sorted(values)[min(len(values)-1,math.ceil(p*len(values))-1)] if values else None


def cost(row):
    return (row.get("usage") or {}).get("total_tokens_charged",0)


def deployment_estimate(groups,snapshot,selected):
    if selected is None:
        return {"available":False,"b2_remaining":snapshot["phase_remaining"]}
    basic=groups["T1"]["mean_tokens"]
    other=groups[selected]["mean_tokens"] if selected!="T1" else 0
    estimate=60*(basic+other)
    return {"available":True,"b2_remaining":snapshot["phase_remaining"],"selected":selected,
            "expected_60_observed_mean":estimate,"two_times_observed_mean":2*estimate,
            "estimated_fit":estimate<=min(snapshot["phase_remaining"],snapshot["global_remaining"])}


def summarize(rows,settled):
    passed=[row for row in rows if row["status"]=="ok"]
    acts=Counter(row["decision"]["action"] for row in passed)
    cited=[row for row in passed if row["decision"]["evidence_ids"]]
    valid=sum(bool(row["verification"].get("passed")) for row in cited)
    alpha=[row["alpha"] for row in rows if row.get("alpha") is not None]
    direction=[row["direction_hit"] for row in rows if row.get("direction_hit") is not None]
    calls=[c for row in rows for c in row["calls"] if c.get("attempt",0)>0 and c.get("status")!="blocked"]
    tools=[tool for row in rows for tool in row["tool_trace"]]
    structures=[e for row in rows for e in row["structured_trace"]]
    recovery=[e for row in rows for e in row["agent_trace"] if e["recovery_status"] in {"failed","recovered"}]
    proposed=Counter()
    for row in rows:
        for trace in row["citations"]:
            proposed["attempted"]+=len(trace["cited_ids"])
            proposed["valid"]+=len(trace["valid_ids"])
        for stance in row["debate"].values():
            for claim in stance["claims"]:
                proposed["attempted"]+=len(claim["evidence_ids"])
                proposed["valid"]+=len(claim["evidence_ids"])
    lat=[row["latency_seconds"] for row in rows]
    memory_rows=[row.get("memory") or {} for row in rows]
    injected=[m for m in memory_rows if m.get("passed_to_backend") and m.get("injected_ids")]
    debate_rows=[row for row in rows if any(trace["triggered"] for trace in row["debate_trace"])]
    bull_bear=[event for row in rows for event in row["agent_trace"] if event["node"] in {"bull","bear","adjudicate"}]
    rag=[tool["retrieval_trace"] for tool in tools if tool.get("retrieval_trace")]
    return {"n":20,"observed":len(rows),"ok":len(passed),"completion_rate":len(passed)/20,"actions":dict(acts),
            "normal_hold":sum(r["decision"]["action"]=="hold" and not r["decision"]["risk_veto"] for r in passed),
            "risk_veto_hold":sum(r["decision"]["risk_veto"]=="volatility_limit" for r in passed),
            "failure_hold":sum(r["status"]!="ok" and r["decision"]["action"]=="hold" for r in rows),
            "nonhold_coverage":(acts["buy"]+acts["sell"])/20,"settled":len(alpha),
            "direction_denominator":len(direction),"conditional_direction_accuracy":mean(direction) if direction else None,
            "mean_alpha":mean(alpha) if alpha else None,"median_alpha":median(alpha) if alpha else None,
            "citation_task_denominator":len(cited),"citation_task_passed":valid,
            "citation_validity":valid/len(cited) if cited else None,"no_citation_tasks":len(passed)-len(cited),
            "attempted_citations":dict(proposed),
            "future_evidence_violations":sum(any(e["code"]=="future_evidence" for e in r["errors"])
                                            or "future_evidence" in r["verification"].get("violations",[])
                                            for r in rows),
            "tools":len(tools),"successful_tools":sum(t.get("status")=="ok" for t in tools),
            "tool_call_success_rate":sum(t.get("status")=="ok" for t in tools)/len(tools) if tools else None,
            "structure_events":len(structures),"valid_structure_events":sum(bool(e["valid"]) for e in structures),
            "structured_output_validity":sum(bool(e["valid"]) for e in structures)/len(structures) if structures else None,
            "recovery_events":len(recovery),"successful_recoveries":sum(e["recovery_status"]=="recovered" for e in recovery),
            "recovery_success_rate":sum(e["recovery_status"]=="recovered" for e in recovery)/len(recovery) if recovery else None,
            "requests":len(calls),"total_tokens":sum(map(cost,rows)),
            "mean_tokens":mean(map(cost,rows)) if rows else None,
            "latency_mean_seconds":mean(lat) if lat else None,"latency_median_seconds":median(lat) if lat else None,
            "latency_p95_seconds":percentile(lat,.95),
            "rag_calls":len(rag),"rag_delivered":sum(r.get("selected_count",0) for r in rag),
            "reranker_latency_ms":mean(r["reranker_ms"] for r in rag) if rag else None,
            "memory_eligible_tasks":sum(bool(m.get("eligible")) for m in memory_rows),
            "memory_used_tasks":len(injected),"memory_use_rate":len(injected)/20,
            "memory_empty_rate_among_observed":1-len(injected)/len(rows) if rows else None,
            "memory_injected_total":sum(len(m["injected_ids"]) for m in injected),
            "memory_context_bytes_mean":mean(m["context_bytes"] for m in injected) if injected else None,
            "memory_latency_ms_mean":mean(m["latency_ms"] for m in memory_rows if "latency_ms" in m)
                                     if any("latency_ms" in m for m in memory_rows) else None,
            "debate_trigger_count":len(debate_rows),"debate_trigger_rate":len(debate_rows)/20,
            "debate_completed_roles":sum(set(row["debate"])=={"bull","bear"} for row in debate_rows),
            "debate_conflict_count":sum(any(t.get("conflict") for t in row["debate_trace"]) for row in debate_rows),
            "debate_tokens":sum(row["token_usage"]["charged_tokens"] for row in bull_bear),
            "debate_latency_seconds":sum(row["latency_ms"] for row in bull_bear)/1000,
            "settlement_enabled":settled}


def paired(rows,first,second,sessions):
    primary={r["task_id"]:r for r in rows if r["arm"]==first}
    control={r["task_id"]:r for r in rows if r["arm"]==second}
    blocks=defaultdict(list)
    deltas=[]
    for key in sorted(primary.keys() & control.keys()):
        a,b=primary[key],control[key]
        if a.get("alpha") is None or b.get("alpha") is None:
            continue
        d=a["alpha"]-b["alpha"]
        deltas.append(d)
        blocks[sessions[a["decision_date"]]//10].append(d)
    interval=None
    if len(blocks)>=2:
        rng=Random(42031)
        samples=list(blocks.values())
        draws=[mean(delta for block in rng.choices(samples,k=len(samples)) for delta in block)
               for _ in range(2000)]
        interval=[percentile(draws,.025),percentile(draws,.975)]
    return {"paired":len(deltas),"excluded_from_full_20":20-len(deltas),"time_blocks":len(blocks),
            "mean_alpha_difference":mean(deltas) if deltas else None,"median_alpha_difference":median(deltas) if deltas else None,
            "bootstrap_95_ci":interval,"exploratory":True}


def admission(groups,pairs):
    baseline=groups["T1"]
    conditions={}
    for arm in ("T2","T3","T4"):
        g=groups[arm]
        pair=pairs[f"{arm}-T1"]
        conditions[arm]={
            "completion":g["ok"]>=18,"paired":pair["paired"]>=18,
            "citations":g["citation_validity"] is not None and g["citation_validity"]>=.95,
            "PIT":g["future_evidence_violations"]==0,
            "nonhold":abs(g["nonhold_coverage"]-baseline["nonhold_coverage"])<=.15+1e-12,
            "tokens":g["mean_tokens"]<=2*baseline["mean_tokens"],
            "p95":g["latency_p95_seconds"]<=2*baseline["latency_p95_seconds"],
            "alpha":pair["mean_alpha_difference"] is not None and pair["mean_alpha_difference"]>0,
        }
    approved=[arm for arm,value in conditions.items() if all(value.values())]
    if not approved:
        return "T1",conditions
    highest=max(pairs[f"{arm}-T1"]["mean_alpha_difference"] for arm in approved)
    compatible=[arm for arm in approved if highest-pairs[f"{arm}-T1"]["mean_alpha_difference"]<.001]
    return min(compatible,key=lambda arm:(groups[arm]["mean_tokens"],ARMS.index(arm))),conditions


def redact(row):
    return {"task_hash":hash_bytes(row["task_id"].encode()).hexdigest()[:24],
            "arm":row["arm"],"status":row["status"],"action":row["decision"]["action"],
            "risk_veto":row["decision"]["risk_veto"],"alpha":row.get("alpha"),"direction_hit":row.get("direction_hit"),
            "evidence_ids":[hash_bytes(key.encode()).hexdigest()[:24] for key in row["decision"]["evidence_ids"]],
            "tools":[{k:event.get(k) for k in ("name","status","dispatch","code")}
                     for event in row["tool_trace"]],
            "rag_count":sum((event.get("retrieval_trace") or {}).get("selected_count",0) for event in row["tool_trace"]),
            "memory_ids":[hash_bytes(key.encode()).hexdigest()[:24] for key in (row.get("memory") or {}).get("injected_ids",[])],
            "debate_trace":row["debate_trace"],"verification":row["verification"],
            "recovery_count":sum(event["retry_count"] for event in row["agent_trace"]),
            "errors":row["errors"],"system_fault":row["system_fault"],"requests":len(row["calls"]),
            "usage":row["usage"],"latency_seconds":row["latency_seconds"]}


def export(allow_settlement=False,work=WORK,out=OUT):
    ledger=open_budget()
    snapshot=ledger.snapshot(experiment_id=EXPERIMENT,phase=PHASE)
    manifest=load(out/"pilot-manifest.json")
    config=load(out/"freeze.json") if (out/"freeze.json").is_file() else {}
    smoke=load(work/"real-smoke.json") if (work/"real-smoke.json").is_file() else None
    stop=load(work/"stop.json") if (work/"stop.json").is_file() else None
    rows={}
    if config and (work/"predictions.sqlite").exists():
        with closing(record_db(work/"predictions.sqlite")) as db:
            rows=reconcile(ledger,db,sha256(out/"freeze.json"),previous_smoke(work,out))
    assembled=list(rows.values())
    selected=None
    pairs={}
    admission_tests={}
    if allow_settlement:
        seal=load(work/"prediction-seal.json")
        requested={(arm,task["task_id"]) for arm in ARMS for task in manifest["tasks"]}
        if (seal["count"]!=100 or seal["freeze_sha256"]!=sha256(out/"freeze.json")
                or seal["records_sha256"]!=fingerprint(rows)
                or {(r["arm"],r["task_id"]) for r in assembled}!=requested
                or any(r["system_fault"] for r in assembled)):
            raise ValueError("Cannot settle without all frozen terminal Pilot predictions")
        tasks={raw["task_id"]:Task.model_validate(raw) for raw in manifest["tasks"]}
        market=DevelopmentMarket(PUBLIC/"market",[t.symbol for t in tasks.values()],manifest["fingerprints"]["market_manifest"])
        if market.prefix_hashes!=manifest["fingerprints"]["market_prefixes"]:
            raise ValueError("Market settlement source differs from preparation")
        outcomes={key:market.outcome(task) for key,task in tasks.items()}
        for row in assembled:
            task=tasks[row["task_id"]]
            outcome=outcomes[task.task_id]
            if outcome.available_at<=close_available(task.decision_date):
                raise ValueError("Invalid decision to outcome ordering")
            action=Action(row["decision"]["action"])
            if row["status"]=="ok":
                row["alpha"]=action.direction*(outcome.asset_return-outcome.benchmark_return)
                row["direction_hit"]=action.direction*outcome.asset_return>0 if action.direction else None
        calendar={str(bar.day):i for i,bar in enumerate(market.series[market.manifest["benchmark"]])}
        pairs={f"{a}-{b}":paired(assembled,a,b,calendar) for a,b in PAIRS}
    groups={arm:summarize([r for r in assembled if r["arm"]==arm],allow_settlement) for arm in ARMS}
    if allow_settlement:
        for arm in ("T2","T3"):
            pairs[f"{arm}-T1"]=paired(assembled,arm,"T1",calendar)
        selected,admission_tests=admission(groups,pairs)
    forecasts=deployment_estimate(groups,snapshot,selected)
    costs={"pilot":snapshot,"smoke":smoke,"remaining_development":forecasts,
           "actual_b2_requests":snapshot["http_attempts"],"actual_glm_tokens":snapshot["conservatively_charged_tokens"]}
    totals={"status":"complete" if allow_settlement else "incomplete_no_settlement","denominator":100,
            "observed":len(assembled),"glm_instances":sum(row["arm"]!="T0" for row in assembled),
            "configuration":{k:config.get(k) for k in ("retriever","context","model","code_git_sha")},
            "groups":groups,"pairs":pairs,"selected":selected,"gate":admission_tests,
            "budget":costs,"stop":stop}
    atomic(out/"summary.json",totals)
    atomic(out/"budget-summary.json",costs)
    atomic(out/"paired-ablation.json",{"scored":allow_settlement,"comparisons":pairs,"admission":admission_tests})
    memory=load(out/"memory-coverage.json")
    memory["actual"]={arm:{k:groups[arm][k] for k in groups[arm] if k.startswith("memory_")} for arm in ("T3","T4")}
    atomic(out/"memory-coverage.json",memory)
    t3={r["task_id"]:r for r in assembled if r["arm"]=="T3" and r["status"]=="ok"}
    debate=[]
    for row in assembled:
        if row["arm"]!="T4":
            continue
        prior=t3.get(row["task_id"])
        debate.append({"task_hash":redact(row)["task_hash"],"trace":row["debate_trace"],
                       "role_success":{key:key in row["debate"] for key in ("bull","bear")},
                       "paired_t3_action_changed":prior["decision"]["action"]!=row["decision"]["action"]
                           if prior and row["status"]=="ok" else None})
    atomic(out/"debate-analysis.json",{"note":"Observed cross-arm action contrast, not a one-instance counterfactual",
                                      "totals":{k:v for k,v in groups["T4"].items() if k.startswith("debate_")},
                                      "tasks":debate})
    failures=Counter(f'{r["arm"]}:{entry["stage"]}:{entry["code"]}' for r in assembled for entry in r["errors"])
    atomic(out/"failure-attribution.json",{"task_error_counts":dict(failures),"system_stop":stop,
                                           "unexecuted":100-len(assembled)})
    (out/"predictions.jsonl").write_text("".join(json.dumps(redact(row),ensure_ascii=False,sort_keys=True)+"\n"
                                                  for row in assembled),encoding="utf-8")
    from .reports import generate
    generate(out,totals)
    print(totals["status"], totals["observed"], "selected",selected,"GLM",costs["actual_glm_tokens"],flush=True)
