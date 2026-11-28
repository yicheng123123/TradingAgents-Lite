"""Offline A1-V5 Financial RAG extractive compression experiment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json, math, pickle, re, sqlite3
from pathlib import Path
from statistics import mean
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from . import financebench
from .embedding_v4 import BGEM3Encoder
from .rag import RagConfig

ROOT=Path(__file__).resolve().parents[1]
V3=ROOT.parent/"tradingagents-a1-retrieval-v3"
V4=ROOT.parent/"tradingagents-a1-retrieval-v4"
SOURCE=ROOT.parent/"tradingagents-lite"
RUN=ROOT/"data/a1-v5-context-compression/001"
OUT=ROOT/"results/a1-v5-context-compression"
MODEL=V4/"data/a1-retrieval-v4/001/models/baai-bge-m3"
BUDGETS=(3200,1600); RESERVE=.35; MAX_REQUEST=8000
MMR=(.35,.50,.65,.80); CUTOFF=1767225600.0
PARTITION="financebench_public"; PIT={"historical_proxy","verified_known_at"}
OMIT="\n[… omitted text …]\n"; CUT="\n[… omitted remainder …]"
HEADER_NOTE="\n[Table header not reliably bound; row preserved as printed.]"
ID_RE=re.compile(rb'"financebench_id"\s*:\s*"([^"]+)"')
YEAR=re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
NUM=re.compile(r"(?<![A-Za-z0-9])(?:[$€£]\s*)?(?:\(\s*[-+]?\d[\d,]*(?:\.\d+)?%?\s*\)|[-+]?\d[\d,]*(?:\.\d+)?%?)(?![A-Za-z0-9])")
ATOM=r"(?:[$€£]\s*)?(?:\(\s*[-+]?\d[\d,]*(?:\.\d+)?%?\s*\)|[-+]?\d[\d,]*(?:\.\d+)?%?|—|–|-)"
DOT=re.compile(r"\.{3,}\s*(?=(?:[$€£]\s*)?(?:\(?[-+]?\d|[—–]))")
VALUES=re.compile(rf"{ATOM}(?:\s+{ATOM}){{1,}}")
UNIT=re.compile(r"(?<!\w)(?:USD|EUR|GBP|CAD|JPY|CNY|thousands?|millions?|billions?|trillions?|percent(?:age)?|per\s+share|shares?|basis\s+points?|bps)(?!\w)|[$€£%]",re.I)
METRICS=tuple(sorted({"net income","net earnings","operating income","income from operations","operating cash flow",
 "cash from operations","net cash provided by operating activities","revenue","revenues","net sales","total sales",
 "total revenue","total assets","total liabilities","current liabilities","cash and cash equivalents","free cash flow",
 "capital expenditures","capital expenditure","gross profit","gross margin","operating margin","debt ratio",
 "long-term debt","shareholders equity","stockholders equity","earnings per share","diluted eps","basic eps",
 "accounts receivable","depreciation","operating expenses","cost of revenue"},key=len,reverse=True))
ACTIVE=None

def compact(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def norm(x): return " ".join(str(x).casefold().split())
def digest(path):
 h=sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def tree(path):
 return {p.relative_to(path).as_posix():digest(p) for p in sorted(path.rglob("*")) if p.is_file() and p.name!=".lock"}
def readjson(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def readrows(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
def writejson(path,x):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def writerows(path,xs):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n" for x in xs),encoding="utf-8")
def pseudo(x): return sha256(str(x).encode()).hexdigest()[:24]

def _args_paths(a):
 return {"v3":Path(a.v3_root).resolve(),"v4":Path(a.v4_root).resolve(),"source":Path(a.source_root).resolve(),
         "run":Path(a.run_dir).resolve(),"out":Path(a.out_dir).resolve(),"model":Path(a.model_root).resolve()}

def _source_rows(path,wanted):
 found={}
 with Path(path).open("rb") as f:
  for line in f:
   m=ID_RE.search(line)
   if m:
    key=m.group(1).decode()
    if key in wanted:
     row=json.loads(line)
     if row.get("financebench_id")!=key: raise ValueError("FinanceBench id mismatch")
     found[key]=row
 return found

def _rank_inputs(v3,split):
 archive=v3/"results/a1-retrieval-v3"
 scores={r["id"]:r for r in readrows(archive/f"{split}-S3.jsonl")}
 traces={r["id"]:r for r in readrows(archive/f"{split}-ranking-trace.jsonl") if r.get("arm")=="S3" and not r.get("failure")}
 expected=49 if split=="calibration" else 99
 if len(scores)!=expected or set(scores)!=set(traces): raise ValueError(f"invalid {split} denominator")
 ranked={}
 for key,t in traces.items():
  rows=sorted((r for r in t["candidates"] if r[5] is not None),key=lambda r:r[5])
  ranked[key]=[r[0] for r in rows[:5]]
  if split=="calibration" and ranked[key]!=scores[key]["top5"]: raise ValueError("S3 trace/top5 mismatch")
 return scores,ranked

def _load_index(index,expected,wanted):
 if tree(index)!=expected: raise ValueError("V3 original index fingerprint mismatch")
 db=index/"collection/chunks/storage.sqlite"
 con=sqlite3.connect(f"file:{db}?mode=ro",uri=True); result={}
 try:
  for (blob,) in con.execute("SELECT point FROM points"):
   point=pickle.loads(blob)
   if type(point).__name__!="PointStruct": raise ValueError("unexpected Qdrant point blob")
   key=sha256(repr((str(point.id),)).encode()).hexdigest()[:24]
   if key in wanted:
    result[key]=dict(point.payload)
    if len(result)==len(wanted): break
 finally: con.close()
 if set(result)!=wanted: raise ValueError(f"missing {len(wanted-set(result))} archived chunks")
 return result

def _validate(p):
 if p.get("dataset_partition")!=PARTITION or float(p.get("eligible_ts",math.inf))>CUTOFF or p.get("pit_level") not in PIT:
  raise ValueError("frozen candidate violates dataset or point-in-time scope")
 if any(k not in p for k in ("chunk_id","document_id","page","text","ticker")) or not p["text"] or p["page"]<0:
  raise ValueError("invalid candidate provenance")

def _labels(key,raw,score):
 evidence=[]
 for e in raw.get("evidence",[]):
  if type(e.get("evidence_page_num")) is int and e["evidence_page_num"]>=0:
   evidence.append({"document_id":e["doc_name"],"page":e["evidence_page_num"],
     "evidence_text":e.get("evidence_text",""),"evidence_text_full_page":e.get("evidence_text_full_page","")})
 return {"case_id":key,"gold_pages":sorted({(e["document_id"],e["page"]) for e in evidence}),
  "evidence":evidence,"retrieval_recall_at_5":score.get("page",{}).get("recall@5")}

def _partition(paths,split):
 v3=paths["v3"]; source=paths["source"]
 freeze=readjson(v3/"results/a1-retrieval-v3/freeze.json")
 scores,ranks=_rank_inputs(v3,split)
 source_file=source/"data/financebench/data/financebench_open_source.jsonl"
 source_hash=digest(source_file)
 if source_hash!=freeze["inputs"]["A1_V2_inputs"]["dataset_sha256"]["financebench_open_source.jsonl"]:
  raise ValueError("public FinanceBench dataset hash differs from V3 freeze")
 rows=_source_rows(source_file,set(scores))
 if set(rows)!=set(scores): raise ValueError("frozen question projection missing")
 index=v3/"data/a1-retrieval-v3/002/original-index"
 expected=freeze["inputs"]["A1_V2_inputs"]["original_index"]
 payloads=_load_index(index,expected,{h for hs in ranks.values() for h in hs})
 qs=[]; labels=[]
 for key,score in scores.items():
  raw=rows[key]; candidates=[]
  for rank,h in enumerate(ranks[key],1):
   p=payloads[h]; _validate(p)
   candidates.append({"candidate_rank":rank,"chunk_hash":h,"chunk_id":str(p["chunk_id"]),
    "evidence_id":"chunk:"+str(p["chunk_id"]),"document_id":str(p["document_id"]),
    "page":int(p["page"]),"ticker":p["ticker"],"eligible_ts":float(p["eligible_ts"]),
    "pit_level":p["pit_level"],"dataset_partition":p["dataset_partition"],"text":str(p["text"])})
  if len({x["ticker"] for x in candidates})!=1: raise ValueError("mixed company filters")
  qs.append({"case_id":key,"company":raw["company"],"question":raw["question"],
             "retrieval_query":raw["question"],"candidates":candidates})
  labels.append(_labels(key,raw,score))
 return qs,labels,{"n":len(qs),"source_sha256":source_hash,
  "s3_results_sha256":digest(v3/f"results/a1-retrieval-v3/{split}-S3.jsonl"),
  "s3_trace_sha256":digest(v3/f"results/a1-retrieval-v3/{split}-ranking-trace.jsonl"),
  "candidate_input_sha256":sha256(compact(qs).encode()).hexdigest()}

def prepare(a):
 p=_args_paths(a); v3freeze=readjson(p["v3"]/"results/a1-retrieval-v3/freeze.json")
 v4freeze=readjson(p["v4"]/"results/a1-retrieval-v4/freeze.json")
 qs,labels,fingerprints=_partition(p,"calibration")
 if len(qs)!=49 or any(len(q["candidates"])!=5 for q in qs): raise ValueError("expected 49 fixed Top-5 inputs")
 if p["run"].exists() and any(p["run"].iterdir()): raise FileExistsError("local run directory is not empty")
 p["run"].mkdir(parents=True,exist_ok=True)
 writerows(p["run"]/"calibration-inputs.jsonl",qs); writerows(p["run"]/"calibration-scoring-labels.jsonl",labels)
 expected=v4freeze["model"]["files_sha256"]; actual={k:digest(p["model"]/k) for k in expected}
 if actual!=expected: raise ValueError("BGE-M3 weights differ from pinned V4 files")
 writejson(p["run"]/"manifest.json",{"schema":"a1-v5-inputs","branch":"experiment/a1-v5-context-compression",
  "base_commit":"8b561a9d28b23aba73f2ecb900b1e757f89c8027","calibration_n":49,"candidate_k":5,
  "budgets":BUDGETS,"followup_reserve":RESERVE,"max_request_bytes":MAX_REQUEST,
  "v3_freeze_sha256":digest(p["v3"]/"results/a1-retrieval-v3/freeze.json"),
  "v4_freeze_sha256":digest(p["v4"]/"results/a1-retrieval-v4/freeze.json"),
  "index_sha256":tree(p["v3"]/"data/a1-retrieval-v3/002/original-index"),
  "model_sha256":actual,"inputs_sha256":digest(p["run"]/"calibration-inputs.jsonl"),
  "labels_sha256":digest(p["run"]/"calibration-scoring-labels.jsonl"),"fingerprints":fingerprints,
  "test_data_opened":False,"private_test_or_v3_holdout_accessed":False,"glm_requests":0,"glm_tokens":0})
 print(json.dumps({"prepared":49,"candidate_k":5,"test_data_opened":False},indent=2))

def _metric(text):
 low=norm(text)
 for x in METRICS:
  if x in low: return x
 prefix=NUM.split(text,maxsplit=1)[0]
 return " ".join(re.findall(r"[a-z]{3,}",prefix.casefold())[:5]) or None

def _unit(text,ci,s,e,kind,table=False):
 body=text[s:e]
 return {"candidate_index":ci,"start":s,"end":e,"text":body,"kind":kind,"table_row":table,
  "metric":_metric(body),"years":set(YEAR.findall(body)),"header_span":None,
  "header_status":"not_required","vector":None,"relevance":0.0}

def _units(text,ci):
 rows=[]
 for m in DOT.finditer(text):
  vals=VALUES.match(text,m.end())
  if not vals or not NUM.search(vals.group()): continue
  nums=list(NUM.finditer(text,0,m.start()))
  s=nums[-1].end() if nums else max(0,m.start()-180)
  if m.start()-s>240:
   cuts=[x.end() for x in re.finditer(r"[:;.!?]\s+",text[s:m.start()])]
   if cuts:s+=cuts[-1]
  while s<m.start() and text[s].isspace():s+=1
  e=vals.end()
  while e>s and text[e-1].isspace():e-=1
  rows.append(_unit(text,ci,s,e,"financial_row",True))
 off=0
 for line in text.splitlines(keepends=True):
  body=line.rstrip("\r\n")
  if body and NUM.search(body) and re.search("[A-Za-z]",body) and len(list(NUM.finditer(body)))<=16:
   rows.append(_unit(text,ci,off,off+len(body),"financial_row",True))
  off+=len(line)
 rows.sort(key=lambda x:(x["start"],x["end"]))
 headers=list(financebench._HEADER_CUE.finditer(text)); unique=[]; seen=set(); prev=None
 for row in rows:
  span=(row["start"],row["end"])
  if span in seen:continue
  seen.add(span)
  intervening=[h for h in headers if prev and prev["end"]<=h.start()<row["start"]]
  if prev and not intervening and prev.get("header_span"):
   row["header_span"],row["header_status"]=prev["header_span"],prev["header_status"]
  else:
   near=[h for h in headers if 0<=row["start"]-h.end()<=700]
   if near:
    start=max(0,near[0].start()-48); header=text[start:row["start"]]
    if len(header)<=700 and (YEAR.search(header) or re.search(r"\b(thousands?|millions?|billions?|trillions?|year ended|years ended|as of|fiscal year)\b",header,re.I)):
     row["header_span"],row["header_status"]=(start,row["start"]),"bound"
    else: row["header_status"]="uncertain"
   else:row["header_status"]="uncertain"
  if row["header_span"]:row["years"].update(YEAR.findall(text[slice(*row["header_span"])]))
  prev=row;unique.append(row)
 units=list(unique)
 starts=[0]+[m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z(])",text)]
 ends=[m.start() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z(])",text)]+[len(text)]
 for s,e in zip(starts,ends,strict=True):
  while s<e and text[s].isspace():s+=1
  if e<=s or e-s>900:continue
  piece=text[s:e]; n=len(list(NUM.finditer(piece)))
  if n<=14:units.append(_unit(text,ci,s,e,"financial_row" if n and _metric(piece) else "sentence"))
 if not units and text:units=[_unit(text,ci,0,len(text),"candidate_fallback")]
 unique={}
 for u in units:unique.setdefault((u["start"],u["end"],u["kind"]),u)
 return sorted(unique.values(),key=lambda u:(u["start"],u["end"]))

def _encode(q,query,candidates,encoder):
 units=[u for i,c in enumerate(candidates) for u in _units(c["text"],i)]
 if not units:return [],0
 vectors,stats=encoder.encode_with_stats([q,query,*[u["text"] for u in units]])
 v=(np.asarray(vectors[0],dtype=np.float32)+np.asarray(vectors[1],dtype=np.float32))/2
 v/=max(float(np.linalg.norm(v)),1e-12)
 for u,vec in zip(units,vectors[2:],strict=True):
  u["vector"]=np.asarray(vec,dtype=np.float32);u["relevance"]=float(np.dot(v,u["vector"]))
 return units,float(stats["milliseconds"])

def _complementary(a,b):
 return a["table_row"] and b["table_row"] and (a["metric"]!=b["metric"] or
  (a["years"] and b["years"] and a["years"]!=b["years"]))
def _mmr(u,chosen,lam):
 rel=max(0,u["relevance"])
 if not chosen:return rel
 red=max((float(np.dot(u["vector"],x["vector"])) for x in chosen if not _complementary(u,x)),default=0)
 return lam*rel-(1-lam)*max(0,red)
def _merge(spans):
 out=[]
 for s,e in sorted(spans):
  if out and s<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],e))
  elif s<e:out.append((s,e))
 return out
def _render(candidates,mapping,uncertain=()):
 items=[]
 for ci,spans in sorted(mapping.items()):
  spans=_merge(spans);c=candidates[ci]
  text=OMIT.join(c["text"][s:e] for s,e in spans)+(HEADER_NOTE if ci in uncertain else "")
  items.append({"evidence_id":c["evidence_id"],"document_id":c["document_id"],"page":c["page"],
   "excerpt":text,"_chunk_id":c["chunk_id"],"_chunk_hash":c["chunk_hash"],
   "_source_spans":[[s,e] for s,e in spans]})
 return items
def _wire(items):
 return [{"evidence_id":x["evidence_id"],"document_id":x["document_id"],"page":x["page"],"excerpt":x["excerpt"]} for x in items]
def _payload(items):return {"status":"available" if items else "unavailable","reason":None if items else "no_complete_unit","items":_wire(items)}
def _cap(b):return max(0,b-math.ceil(b*RESERVE))
def _ctx(items):return financebench._message_size(_payload(items),"user")
def _request(q,items):
 msgs=[{"role":"system","content":financebench.SYSTEM},{"role":"user","content":q},
  {"role":"user","content":financebench.OBSERVATION_PREFIX+compact(_payload(items))},
  {"role":"user","content":financebench.FINAL}]
 return financebench._prompt_body_size(msgs,())
def _fits(q,items,b):return _ctx(items)<=_cap(b) and _request(q,items)<=MAX_REQUEST
def _exact(item,candidates):
 c=next((x for x in candidates if x["chunk_id"]==item["_chunk_id"]),None)
 if c is None or (c["document_id"],c["page"])!=(item["document_id"],item["page"]):return False
 spans=item["_source_spans"]
 if any(s<0 or s>=e or e>len(c["text"]) for s,e in spans):return False
 source=OMIT.join(c["text"][s:e] for s,e in spans)
 got=item["excerpt"].removesuffix(HEADER_NOTE).removesuffix(CUT)
 return source==got

def _result(q,candidates,arm,b,started,items,units=(),embed=0,lam=None):
 raw=sum(len(c["text"].encode()) for c in candidates)
 source=sum(len(next(c["text"] for c in candidates if c["chunk_id"]==i["_chunk_id"])[s:e].encode()) for i in items for s,e in i["_source_spans"])
 enc=ACTIVE
 tokens=sum(len(enc.tokenizer.encode(i["excerpt"],add_special_tokens=True)) for i in items) if enc else None
 attempts=sum(bool(u.get("table_row")) for u in units);bound=sum(u.get("header_status")=="bound" for u in units)
 ctx=_ctx(items);req=_request(q,items)
 return {"arm":arm,"budget_bytes":b,"mmr_lambda":lam,"candidate_count":len(candidates),
  "selected_evidence_count":len(items),"context_bytes":ctx,"effective_capacity_bytes":_cap(b),
  "request_bytes":req,"context_under_budget":ctx<=_cap(b),"request_under_8000":req<=MAX_REQUEST,
  "raw_candidate_text_bytes":raw,"delivered_source_bytes":source,"compression_ratio":source/raw if raw else None,
  "proxy_evidence_tokens":tokens,"items":items,"selected_units":list(units),
  "source_exact":all(_exact(i,candidates) for i in items),
  "header_binding":{"bound":bound,"attempts":attempts,"uncertain":sum(u.get("header_status")=="uncertain" for u in units)},
  "embedding_latency_ms":embed,"selection_latency_ms":(perf_counter()-started)*1000+embed}

def _c0(q,candidates,b):
 start=perf_counter();items=[]
 for c in candidates:
  ends=[m.end() for m in re.finditer(r"\S+",c["text"])]
  lo,hi,best=0,len(ends),None
  while lo<=hi:
   mid=(lo+hi)//2
   trial=list(items)
   if mid:
    end=ends[mid-1];text=c["text"][:end]+(CUT if mid<len(ends) else "")
    trial.append({"evidence_id":c["evidence_id"],"document_id":c["document_id"],"page":c["page"],
     "excerpt":text,"_chunk_id":c["chunk_id"],"_chunk_hash":c["chunk_hash"],"_source_spans":[[0,end]]})
   ok=True if not mid else _fits(q,trial,b)
   if ok:
    if mid:best=mid
    lo=mid+1
   else:hi=mid-1
  if best:
   end=ends[best-1]
   items.append({"evidence_id":c["evidence_id"],"document_id":c["document_id"],"page":c["page"],
    "excerpt":c["text"][:end]+(CUT if best<len(ends) else ""),"_chunk_id":c["chunk_id"],
    "_chunk_hash":c["chunk_hash"],"_source_spans":[[0,end]]})
  if _ctx(items)>=_cap(b):break
 return _result(q,candidates,"C0",b,start,items)

def _c1(q,candidates,b):
 start=perf_counter();old=financebench.RAG
 cfg=old.model_dump() if hasattr(old,"model_dump") else vars(old).copy();cfg["context_bytes"]=b
 financebench.RAG=RagConfig(**cfg)
 try:
  ctx=financebench.CaseSearch(SimpleNamespace(question=q),None,{})
  payload=ctx._deliver([{"evidence_id":c["evidence_id"],"document_id":c["document_id"],"page":c["page"],"excerpt":c["text"],"chunk_id":c["chunk_id"]} for c in candidates],
   query=q,requested_limit=len(candidates),preserve_followup=True,
   base_messages=[{"role":"system","content":financebench.SYSTEM},{"role":"user","content":q}],
   request_tools=(),expect_final=True)
  decisions={d["evidence_id"]:d for d in ctx.trace[-1]["candidates"]}
 finally:financebench.RAG=old
 items=[];units=[]
 for item in payload["items"]:
  c=next(x for x in candidates if x["evidence_id"]==item["evidence_id"])
  tokens=list(re.finditer(r"\S+",c["text"]))
  spans=_merge([(tokens[s].start(),tokens[e-1].end()) for s,e in decisions[item["evidence_id"]].get("selection_segments",[])
                if 0<=s<e<=len(tokens)])
  items.append({"evidence_id":item["evidence_id"],"document_id":item["document_id"],"page":item["page"],
   "excerpt":item["excerpt"],"_chunk_id":c["chunk_id"],"_chunk_hash":c["chunk_hash"],"_source_spans":[[s,e] for s,e in spans]})
  units.append({"candidate_rank":c["candidate_rank"],"kind":"query_aware_segments","header_status":None,"table_row":False})
 result=_result(q,candidates,"C1",b,start,items,units);result["status"]=payload["status"];return result

def _semantic(q,candidates,units,b,arm,lam,embed):
 start=perf_counter();units=list(units);chosen=[];mapping={};uncertain=set();seen=set()
 share=max(256,_cap(b)//max(1,len(candidates)))
 while units:
  proposals=[]
  for u in units:
   ci=u["candidate_index"];text=norm(u["text"])
   if not text or text in seen:continue
   same=[x for x in chosen if x["candidate_index"]==ci]
   if any(max(0,min(u["end"],x["end"])-max(u["start"],x["start"]))/
          max(1,min(u["end"]-u["start"],x["end"]-x["start"]))>.90 for x in same):continue
   score=max(0,u["relevance"]) if arm=="C2" else _mmr(u,chosen,lam)
   if score<=0:continue
   tmap={};tu=set()
   for x in [*chosen,u]:
    spans=tmap.setdefault(x["candidate_index"],[]);spans.append((x["start"],x["end"]))
    if arm=="C4":
     if x["header_span"] and x["header_status"]=="bound":spans.append(x["header_span"])
     elif x["table_row"] and x["header_status"]=="uncertain":tu.add(x["candidate_index"])
   items=_render(candidates,tmap,tu)
   if not _fits(q,items,b):continue
   item=next(x for x in items if x["_chunk_id"]==candidates[ci]["chunk_id"])
   if arm!="C4" and len(compact(_wire([item])).encode())>share:continue
   prior=_ctx(_render(candidates,mapping,uncertain))
   density=score/max(1,_ctx(items)-prior) if arm=="C4" else score
   proposals.append((density,score,-ci,-u["start"],u,tmap,tu))
  if not proposals:break
  _,_,_,_,u,mapping,uncertain=max(proposals,key=lambda x:x[:4]);chosen.append(u);seen.add(norm(u["text"]));units.remove(u)
 items=_render(candidates,mapping,uncertain)
 unit_rows=[{"candidate_rank":u["candidate_index"]+1,"span":[u["start"],u["end"]],
  "kind":u["kind"],"metric":u["metric"],"years":sorted(u["years"]),
  "header_span":list(u["header_span"]) if u["header_span"] else None,
  "header_status":u["header_status"],"table_row":u["table_row"]} for u in chosen]
 return _result(q,candidates,arm,b,start,items,unit_rows,embed,lam)

def _expected_rows(label):
 rows=set()
 for e in label["evidence"]:
  anchor=norm(e.get("evidence_text","")); full=e.get("evidence_text_full_page") or e.get("evidence_text","")
  if not anchor:continue
  units=_units(full,0); found=[u["text"] for u in units if u["table_row"] and NUM.search(u["text"])]
  matched=[x for x in found if norm(x) in anchor]
  if not matched:matched=[u["text"] for u in _units(e["evidence_text"],0) if u["table_row"] and NUM.search(u["text"])]
  rows.update(((e["document_id"],e["page"]),norm(x)) for x in matched)
 return sorted(rows)

def _score(case,result,label,encoder):
 delivered={}
 for item in result["items"]:
  delivered.setdefault((item["document_id"],item["page"]),[]).append(item["excerpt"].removesuffix(HEADER_NOTE).removesuffix(CUT))
 gold={tuple(x) for x in label["gold_pages"]}; pages=set(delivered)
 page_recall=len(gold&pages)/len(gold) if gold else None
 rows=_expected_rows(label)
 row_hits=sum(any(norm(row) in norm(text) for text in delivered.get(page,())) for page,row in rows)
 texts={page:norm(" ".join(value)) for page,value in delivered.items()}
 evidence=[e for e in label["evidence"] if e["evidence_text"]]
 years=set();values=set();units=set()
 for e in evidence:
  text=e["evidence_text"];years.update(YEAR.findall(text))
  values.update(norm(m.group()) for m in NUM.finditer(text) if not YEAR.fullmatch(m.group().strip().replace(",","")))
  units.update(norm(m.group()) for m in UNIT.finditer(text))
 ret_y={x for x in years if any(x in texts.get((e["document_id"],e["page"]),"") for e in evidence)}
 ret_v={x for x in values if any(x in texts.get((e["document_id"],e["page"]),"") for e in evidence)}
 ret_u={x for x in units if any(x in texts.get((e["document_id"],e["page"]),"") for e in evidence)}
 by_id={x["chunk_id"]:x for x in case["candidates"]}
 cite=all(i["_chunk_id"] in by_id and i["evidence_id"]==by_id[i["_chunk_id"]]["evidence_id"]
  and i["document_id"]==by_id[i["_chunk_id"]]["document_id"] and i["page"]==by_id[i["_chunk_id"]]["page"] for i in result["items"])
 rawtok=sum(encoder.token_lengths([x["text"]])[0] for x in case["candidates"]);tok=result["proxy_evidence_tokens"]
 return {"case_key":pseudo(case["case_id"]),"company_key":pseudo(case["company"]),
  "arm":result["arm"],"budget_bytes":result["budget_bytes"],"mmr_lambda":result.get("mmr_lambda"),
  "candidate_count":len(case["candidates"]),"retrieval_page_recall_at_5":label["retrieval_recall_at_5"],
  "delivered_page_recall":page_recall,"gold_page_count":len(gold),"delivered_gold_page_count":len(gold&pages),
  "financial_rows_covered":row_hits,"financial_rows_total":len(rows),
  "complete_financial_row_coverage":row_hits/len(rows) if rows else None,
  "years_retained":len(ret_y),"years_total":len(years),"year_retention":len(ret_y)/len(years) if years else None,
  "values_retained":len(ret_v),"values_total":len(values),"value_retention":len(ret_v)/len(values) if values else None,
  "units_retained":len(ret_u),"units_total":len(units),"unit_retention":len(ret_u)/len(units) if units else None,
  "header_binding_bound":result["header_binding"]["bound"],"header_binding_attempts":result["header_binding"]["attempts"],
  "header_binding_uncertain":result["header_binding"]["uncertain"],"citation_integrity":cite,
  "source_fact_consistency":result["source_exact"],"context_bytes":result["context_bytes"],
  "effective_capacity_bytes":result["effective_capacity_bytes"],"request_bytes":result["request_bytes"],
  "request_under_8000":result["request_under_8000"],"raw_candidate_text_bytes":result["raw_candidate_text_bytes"],
  "delivered_source_bytes":result["delivered_source_bytes"],"compression_ratio":result["compression_ratio"],
  "proxy_evidence_tokens":tok,"raw_proxy_tokens":rawtok,
  "proxy_token_reduction":1-tok/rawtok if tok is not None and rawtok else None,
  "selection_latency_ms":result["selection_latency_ms"],"embedding_latency_ms":result["embedding_latency_ms"],
  "selected_evidence_count":result["selected_evidence_count"],"context_under_budget":result["context_under_budget"]}

def _groups(rows,split):
 grouped={}
 for r in rows:grouped.setdefault((r["arm"],r["budget_bytes"],r.get("mmr_lambda")),[]).append(r)
 out={}
 for (arm,b,lam),items in grouped.items():
  def avg(k):
   vals=[x[k] for x in items if x.get(k) is not None];return mean(vals) if vals else None
  def ratio(n,d):
   a=sum(x.get(n,0) for x in items);z=sum(x.get(d,0) for x in items);return a/z if z else None
  lat=sorted(x["selection_latency_ms"] for x in items);label=f"{arm}|lambda={lam}" if lam is not None else arm
  out[f"{label}|budget={b}"]={"split":split,"n":len(items),"arm":arm,"mmr_lambda":lam,
   "configured_context_budget_bytes":b,"effective_first_round_capacity_bytes":max(0,b-math.ceil(b*RESERVE)),
   "retrieval_recall_at_5_mean":avg("retrieval_page_recall_at_5"),"delivered_page_recall_mean":avg("delivered_page_recall"),
   "complete_financial_row_coverage_micro":ratio("financial_rows_covered","financial_rows_total"),
   "financial_rows_covered":sum(x["financial_rows_covered"] for x in items),
   "financial_rows_total":sum(x["financial_rows_total"] for x in items),
   "year_retention_micro":ratio("years_retained","years_total"),"value_retention_micro":ratio("values_retained","values_total"),
   "unit_retention_micro":ratio("units_retained","units_total"),"header_binding_accuracy":ratio("header_binding_bound","header_binding_attempts"),
   "header_binding_attempts":sum(x["header_binding_attempts"] for x in items),
   "citation_integrity_rate":avg("citation_integrity"),"source_fact_consistency_rate":avg("source_fact_consistency"),
   "mean_context_bytes":avg("context_bytes"),"mean_effective_capacity_utilization":avg("context_bytes")/max(1,b-math.ceil(b*RESERVE)),
   "mean_request_bytes":avg("request_bytes"),"max_request_bytes_observed":max(x["request_bytes"] for x in items),
   "mean_source_compression_ratio":avg("compression_ratio"),"mean_proxy_evidence_tokens":avg("proxy_evidence_tokens"),
   "mean_raw_proxy_tokens":avg("raw_proxy_tokens"),"proxy_token_reduction":avg("proxy_token_reduction"),
   "embedding_latency_mean_ms":avg("embedding_latency_ms"),
   "selection_latency_mean_ms":mean(lat),"selection_latency_p50_ms":lat[(len(lat)-1)//2],
   "selection_latency_p95_ms":lat[min(len(lat)-1,math.ceil(.95*len(lat))-1)],
   "budget_failures":sum(not x["context_under_budget"] or not x["request_under_8000"] for x in items)}
 return out

def _key(a,b,lam=None):return (f"{a}|lambda={lam}" if lam is not None else a)+f"|budget={b}"
def _order(g):
 return tuple(g.get(k) if g.get(k) is not None else -1 for k in
  ("complete_financial_row_coverage_micro","delivered_page_recall_mean","source_fact_consistency_rate","header_binding_accuracy"))+(-(g.get("selection_latency_mean_ms") or math.inf),)
def _choose(groups):
 b=3200;c3=max(((groups[_key("C3",b,x)],x) for x in MMR),key=lambda p:_order(p[0]))
 c4=max(((groups[_key("C4",b,x)],x) for x in MMR),key=lambda p:_order(p[0]))
 new=[(groups[_key("C2",b)],None),(c3[0],c3[1]),(c4[0],c4[1])]
 best,lam=max(new,key=lambda p:_order(p[0]));overall=[groups[_key("C0",b)],groups[_key("C1",b)],*[x[0] for x in new]]
 over=max(overall,key=_order);c1=groups[_key("C1",b)]
 saved=(c1["mean_context_bytes"] or 0)-(best["mean_context_bytes"] or 0)
 return {"best_new":{"arm":best["arm"],"mmr_lambda":lam},"best_new_metrics_key":_key(best["arm"],b,lam),
  "best_c3_mmr_lambda":c3[1],"best_c4_mmr_lambda":c4[1],
  "best_overall":{"arm":over["arm"],"mmr_lambda":over["mmr_lambda"]},
  "best_overall_metrics_key":_key(over["arm"],b,over["mmr_lambda"]),
  "compression_space_saved_vs_c1_bytes":saved,
  "top10_expansion_triggered":bool(saved>=100 and
    (best["delivered_page_recall_mean"] or 0)>=(c1["delivered_page_recall_mean"] or 0)),
  "top10_trigger_rule":"Best new config saves >=100 serialized context bytes vs C1 and does not lower delivered page recall."}

def calibrate(a):
 p=_args_paths(a);qs=readrows(p["run"]/"calibration-inputs.jsonl")
 labels={x["case_id"]:x for x in readrows(p["run"]/"calibration-scoring-labels.jsonl")}
 if len(qs)!=49:raise ValueError("Calibration denominator must be 49")
 import torch
 enc=BGEM3Encoder(p["model"],device=a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
 global ACTIVE;ACTIVE=enc;rows=[];deliveries=[]
 for q in qs:
  units,ems=_encode(q["question"],q["retrieval_query"],q["candidates"],enc)
  for b in BUDGETS:
   results=[_c0(q["question"],q["candidates"],b),_c1(q["question"],q["candidates"],b),
    _semantic(q["question"],q["candidates"],units,b,"C2",None,ems)]
   for lam in MMR:
    results.extend((_semantic(q["question"],q["candidates"],units,b,"C3",lam,ems),
                    _semantic(q["question"],q["candidates"],units,b,"C4",lam,ems)))
   for r in results:
    rows.append(_score(q,r,labels[q["case_id"]],enc))
    deliveries.append({"case_id":q["case_id"],"arm":r["arm"],"budget_bytes":b,"mmr_lambda":r.get("mmr_lambda"),
      "context_bytes":r["context_bytes"],"request_bytes":r["request_bytes"],"items":r["items"],
      "selected_units":r["selected_units"],"source_exact":r["source_exact"]})
  print("Calibration compressed "+q["case_id"],flush=True)
 ACTIVE=None
 writerows(p["run"]/"calibration-deliveries.jsonl",deliveries);writerows(p["out"]/"calibration-rows.jsonl",rows)
 groups=_groups(rows,"calibration");selected=_choose(groups)
 top10={"executed":False,"trigger":selected["top10_expansion_triggered"],
  "reason":"requires Top-10 extension" if selected["top10_expansion_triggered"] else
  "skipped: no >=100-byte saving against C1 at non-inferior page recall",
  "candidate_expansion_contribution":None,"compression_contribution":None}
 summary={"schema":"a1-v5-context-compression-calibration","n":49,"groups":groups,"selected":selected,
  "mmr_grid":MMR,"model":enc.metadata,"candidate_scope":{"source":"A1-V3 S3 frozen final rank",
  "top_k":5,"query":"original question"},"selection_rule":["financial row coverage","delivered page recall",
  "source fact consistency","header binding","selection latency"],"top10_expansion":top10,
  "test_data_opened":False,"glm_requests":0,"glm_tokens":0}
 writejson(p["out"]/"calibration-summary.json",summary)
 print(json.dumps({"rows":len(rows),"selected":selected,"top10":top10},indent=2))

def _code_hashes():
 return {f:digest(ROOT/f) for f in ("tradingagents_lite/financebench_compression_v5.py",
  "tradingagents_lite/financebench.py","tradingagents_lite/embedding_v4.py",
  "tests/test_financebench_compression_v5.py")}

def freeze(a):
 p=_args_paths(a);calpath=p["out"]/"calibration-summary.json"
 if not calpath.exists():raise FileNotFoundError("run Calibration first")
 cal=readjson(calpath);m=readjson(p["run"]/"manifest.json")
 f={"schema":"a1-v5-context-compression-freeze","experiment":"A1-V5 Financial RAG Context Compression",
  "branch":"experiment/a1-v5-context-compression","base_commit":"8b561a9d28b23aba73f2ecb900b1e757f89c8027",
  "selected_configuration":cal["selected"]["best_new"],"calibration_selection":cal["selected"],
  "budgets_bytes":BUDGETS,"followup_reserve_ratio":RESERVE,"max_request_bytes":MAX_REQUEST,"mmr_grid":MMR,
  "fixed_retrieval":{"baseline":"A1-V3 S3","candidate_count":5,"query":"original FinanceBench question"},
  "calibration":{"n":49,"summary_sha256":digest(calpath),"rows_sha256":digest(p["out"]/"calibration-rows.jsonl"),
  "inputs_sha256":m["inputs_sha256"],"labels_sha256":m["labels_sha256"]},
  "top10_expansion":cal["top10_expansion"],"fingerprints":m,"source_code_sha256":_code_hashes(),
  "formal_test_plan":{"n":99,"arms":["C1","frozen selected compression"],"budgets_bytes":BUDGETS},
  "test_data_opened":False,"private_test_or_v3_holdout_accessed":False,"glm_requests":0,"glm_tokens":0}
 writejson(p["out"]/"freeze.json",f)
 print(json.dumps({"frozen":True,"freeze_sha256":digest(p["out"]/"freeze.json"),
  "selected":f["selected_configuration"],"test_data_opened":False},indent=2))

def test_frozen(a):
 p=_args_paths(a);fp=p["out"]/"freeze.json"
 if not fp.exists():raise FileNotFoundError("freeze before public Test")
 frozen=readjson(fp);fh=digest(fp)
 if frozen["test_data_opened"] or frozen["source_code_sha256"]!=_code_hashes():raise ValueError("freeze mismatch")
 if (p["run"]/"test-started.json").exists() or (p["out"]/"test-rows.jsonl").exists():
  raise FileExistsError("public Test may run only once")
 writejson(p["run"]/"test-started.json",{"freeze_sha256":fh,"started_utc":datetime.now(timezone.utc).isoformat(),
  "glm_requests":0,"glm_tokens":0})
 qs,labels,fingerprints=_partition(p,"test")
 if len(qs)!=99:raise ValueError("keep all 99 public Test questions in denominator")
 writerows(p["run"]/"test-inputs.jsonl",qs);writerows(p["run"]/"test-scoring-labels.jsonl",labels)
 import torch
 enc=BGEM3Encoder(p["model"],device=a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
 global ACTIVE;ACTIVE=enc;rows=[];deliveries=[];labelmap={x["case_id"]:x for x in labels}
 config=frozen["selected_configuration"]
 for q in qs:
  units,ems=_encode(q["question"],q["retrieval_query"],q["candidates"],enc)
  for b in BUDGETS:
   variants=[_c1(q["question"],q["candidates"],b)]
   if config["arm"]=="C2":variants.append(_semantic(q["question"],q["candidates"],units,b,"C2",None,ems))
   elif config["arm"]!="C1":variants.append(_semantic(q["question"],q["candidates"],units,b,config["arm"],config.get("mmr_lambda"),ems))
   for r in variants:
    rows.append(_score(q,r,labelmap[q["case_id"]],enc))
    deliveries.append({"case_id":q["case_id"],"arm":r["arm"],"budget_bytes":b,"mmr_lambda":r.get("mmr_lambda"),
     "context_bytes":r["context_bytes"],"request_bytes":r["request_bytes"],"items":r["items"],"selected_units":r["selected_units"]})
 ACTIVE=None
 writerows(p["run"]/"test-deliveries.jsonl",deliveries);writerows(p["out"]/"test-rows.jsonl",rows)
 summary={"schema":"a1-v5-context-compression-test","n":99,"denominator":99,"groups":_groups(rows,"test"),
  "freeze_sha256":fh,"selected_configuration":config,"fingerprints":fingerprints,"model":enc.metadata,"test_data_opened":True,"private_test_or_v3_holdout_accessed":False,
  "glm_requests":0,"glm_tokens":0}
 writejson(p["out"]/"test-summary.json",summary);writejson(p["run"]/"test-complete.json",{"freeze_sha256":fh,"n":99})
 _export(p);print(json.dumps({"test_n":99,"freeze_sha256":fh,"glm_requests":0,"glm_tokens":0},indent=2))

def _charts(p,cal,test,freeze):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 groups=cal["groups"];sel=freeze["calibration_selection"]
 variants=[("C0",None,"C0 raw prefix","#748094"),("C1",None,"C1 query aware","#38BDF8"),
  ("C2",None,"C2 semantic","#FBBF24"),("C3",sel["best_c3_mmr_lambda"],"C3 + MMR","#A78BFA"),
  ("C4",sel["best_c4_mmr_lambda"],"C4 + header/global","#34D399")]
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
 for ax,metric,title in zip(axes,("delivered_page_recall_mean","complete_financial_row_coverage_micro"),
  ("Delivered page recall","Complete financial row coverage"),strict=True):
  for arm,lam,label,color in variants:
   pts=sorted((_cap(b),groups[_key(arm,b,lam)].get(metric)) for b in BUDGETS)
   if all(y is not None for _,y in pts):ax.plot([x for x,_ in pts],[y for _,y in pts],marker="o",color=color,label=label)
  ax.set_title(title,loc="left");ax.set_xlabel("First-round evidence capacity (UTF-8 bytes)")
  ax.set_ylabel("Recall / coverage");ax.set_ylim(0,1.02);ax.grid(axis="y",alpha=.25)
 axes[0].legend(frameon=False,fontsize=8,loc="lower right")
 fig.savefig(p/"quality-budget.svg",metadata={"Date":None});plt.close(fig)
 groups=test["groups"];config=freeze["selected_configuration"];fig,ax=plt.subplots(figsize=(8.5,4.7),layout="constrained")
 for metric,label,color in (("delivered_page_recall_mean","Delivered page recall","#38BDF8"),
  ("complete_financial_row_coverage_micro","Financial row coverage","#34D399")):
  x=list(BUDGETS);ax.plot(x,[groups[_key("C1",b)].get(metric) for b in x],marker="o",ls="--",color=color,label=label+" · C1")
  ax.plot(x,[groups[_key(config["arm"],b,config.get("mmr_lambda"))].get(metric) for b in x],
   marker="o",color=color,label=label+" · frozen")
 ax.set_title("Frozen public Test · 99 questions",loc="left");ax.set_xlabel("Configured context budget (UTF-8 bytes)")
 ax.set_ylabel("Recall / coverage");ax.set_ylim(0,1.02);ax.grid(axis="y",alpha=.25);ax.legend(frameon=False,fontsize=8)
 fig.savefig(p/"frozen-test-comparison.svg",metadata={"Date":None});plt.close(fig)

def _report(p,s,f):
 cal=s["calibration"];test=s["test"];cfg=f["selected_configuration"];b=3200
 c1=cal["groups"][_key("C1",b)];new=cal["groups"][_key(cfg["arm"],b,cfg.get("mmr_lambda"))]
 c1t=test["groups"][_key("C1",b)];newt=test["groups"][_key(cfg["arm"],b,cfg.get("mmr_lambda"))]
 def row(name,g):
  pct=lambda k:"Unavailable" if g.get(k) is None else f"{g[k]:.1%}"
  val=lambda k:"Unavailable" if g.get(k) is None else f"{g[k]:.1f}"
  return f"| {name} | {pct('delivered_page_recall_mean')} | {pct('complete_financial_row_coverage_micro')} | {pct('year_retention_micro')} | {pct('value_retention_micro')} | {pct('unit_retention_micro')} | {pct('citation_integrity_rate')} | {pct('source_fact_consistency_rate')} | {val('mean_context_bytes')} | {val('selection_latency_mean_ms')} |"
 lines=["# A1-V5 Financial RAG Context Compression","","## Calibration（49题；3200-byte配置）","",
  f"冻结新压缩：**{cfg['arm']}**"+(f"（MMR λ={cfg['mmr_lambda']:.2f}）" if cfg.get("mmr_lambda") is not None else ""),
  f"综合 Calibration 指标优先级 winner：**{cal['selected']['best_overall']['arm']}**。固定检索来自 A1-V3 S3 Top-5，压缩没有改变检索 Recall@5。",
  "C1 使用 FinanceBench V2 CaseSearch._deliver；C2 为 BGE-M3 extractive selection；C3 加 MMR；C4 加来源内表头绑定与全局预算分配。","",
  "| 配置 | 送达页召回 | 完整行覆盖 | 年份保留 | 数值保留 | 单位保留 | 引用完整 | 原文一致 | 上下文 bytes | 选择 ms |",
  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",row("C1",c1),
  row(cfg["arm"]+(f" λ={cfg['mmr_lambda']:.2f}" if cfg.get("mmr_lambda") else ""),new),"",
  f"冻结压缩比 C1 平均少使用 {(c1['mean_context_bytes'] or 0)-(new['mean_context_bytes'] or 0):.1f} serialized bytes。检索 Recall@5 两组相同，为 {(c1['retrieval_recall_at_5_mean'] or 0):.1%}。","",
  "## 冻结公开 Test（99题）","",
  "| Configuration | Delivered page recall | Full row coverage | Year retention | Value retention | Unit retention | Context bytes | Selection ms |",
  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",row("C1",c1t),
  row(cfg["arm"]+(f" λ={cfg['mmr_lambda']:.2f}" if cfg.get("mmr_lambda") else ""),newt),"",
  "FinanceBench 公共 Test 在历史 A1 迭代中多次复评，并非独立未见测试集；本轮只作一次冻结复评。","",
  "## 预算与数据隔离","",
  "- Evidence 上限 3200 bytes，35% 首轮预留后有效容量2080；1600-byte压力测试有效容量1040。",
  "- 完整 System、问题、观察消息和结束指令最多8000 UTF-8 bytes。",
  "- Token 数使用 BGE-M3 tokenizer 代理，不代表 GLM Token 节省。",
  "- 事实片段与表头均通过原文字符 offset 回溯；逐题结果不含原文、公司名或来源 URL。",
  "- 公共 Calibration 的 answer/evidence 投影只进入评分，压缩选择接收的问题、query和候选 Chunk。",
  "- Private Test 与 V3 Holdout 未访问；索引只读且逐文件指纹验证；GLM请求0、GLM Token0。",
  "", "## Top-10 候选池",cal["top10_expansion"]["reason"],"",
  "![Calibration quality-budget curve](quality-budget.svg)","",
  "![Frozen Test comparison](frozen-test-comparison.svg)","",
  "运行命令见 COMMANDS.md；冻结清单见 freeze.json；脱敏逐题数据见 calibration-rows.jsonl 和 test-rows.jsonl。","",
  "## 限制","",
  "- 财务行覆盖以 FinanceBench 证据全文和摘录进行字符串匹配，属于离线代理指标。",
  "- BGE-M3 相似度排序不代表 GLM 实际可读性或问答准确率；真实 GLM 验证需要另行批准。",
  "- 本轮没有重新检索候选，也没有更换 reranker。"]
 (p/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def _export(p):
 cal=readjson(p["out"]/"calibration-summary.json");test=readjson(p["out"]/"test-summary.json")
 freeze=readjson(p["out"]/"freeze.json")
 writejson(p["out"]/"summary.json",{"schema":"a1-v5-context-compression-summary","calibration":cal,
  "test":test,"selected_configuration":freeze["selected_configuration"],"freeze_sha256":digest(p["out"]/"freeze.json"),
  "glm_requests":0,"glm_tokens":0,"test_public_reused_not_unseen":True})
 _charts(p["out"],cal,test,freeze);_report(p["out"],{"calibration":cal,"test":test},freeze)

def parser():
 x=argparse.ArgumentParser(description=__doc__);x.add_argument("command",choices=("prepare","calibrate","freeze","test"))
 x.add_argument("--v3-root",default=str(V3));x.add_argument("--v4-root",default=str(V4))
 x.add_argument("--source-root",default=str(SOURCE));x.add_argument("--run-dir",default=str(RUN));x.add_argument("--out-dir",default=str(OUT))
 x.add_argument("--model-root",default=str(MODEL));x.add_argument("--device",choices=("cuda","cpu"),default=None);return x
def main(argv=None):
 a=parser().parse_args(argv)
 {"prepare":prepare,"calibrate":calibrate,"freeze":freeze,"test":test_frozen}[a.command](a)
if __name__=="__main__":main()
