"""Build task-bound prediction packets and a separate Development settlement input."""
from collections import Counter
from datetime import date, datetime
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
from time import perf_counter

from ..data import HistoricalData, MarketView, close_available, read_tasks, sha256
from ..data_v2_market import ISSUERS
from ..documents import Document, chunk_document, digest, parse_document
from ..evidence import Evidence, EvidenceStore
from ..fundamentals import FundamentalsView
from ..memory import MemoryConfig, MemoryReplay, MemoryStore
from ..rag import compact_json
from ..schema import Bar, Budget, Decision, Task
from ..vectors import LocalEmbedding, VectorStore
from ..workflow import MomentumBackend, build_workflow

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "tradingagents-lite"
PUBLIC = SOURCE / "data/data-expansion-v3"
WORK = ROOT / "data/final-agent-pilot-001"
OUT = ROOT / "results/final-agent-pilot-001"
TASKS = ROOT / "tasks/data-expansion-v3"
PARTITION = "development-v3"
PILOT_KEY = "final-agent-pilot-001:"
WARMUP_ARM = "deterministic_momentum_warmup"
LAYOUT_STOP = date(2026,8,7)
BUDGET = Budget(max_tokens=768, max_tool_rounds=1)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    temp.replace(path)


def select_tasks(tasks):
    grouped = {}
    for task in tasks:
        if not task.task_id.startswith("v3-development-") or task.horizon_sessions!=5 or task.decision_date>=LAYOUT_STOP:
            raise ValueError("Only the frozen V3 Development schedule is allowed")
        grouped.setdefault(task.symbol,[]).append(task)
    if len(tasks)!=80 or len(grouped)!=20 or any(len(items)!=4 for items in grouped.values()):
        raise ValueError("V3 manifest population differs")
    result = [min(items,key=lambda t:hash_bytes((PILOT_KEY+t.task_id).encode()).digest()) for items in grouped.values()]
    return sorted(result,key=lambda t:(t.decision_date,t.task_id))


class DevelopmentMarket(HistoricalData):
    """Read the fixed public Development market prefix, ending before Holdout."""

    def __init__(self, directory, symbols, expected):
        path = directory/"manifest.json"
        if sha256(path)!=expected:
            raise ValueError("V3 market manifest changed")
        self.manifest=load(path)
        self.series={}
        self.split_events={}
        self.prefix_hashes={}
        for symbol in sorted(set(symbols)|{self.manifest["benchmark"]}):
            if self.manifest["assets"][symbol]["split_events"]:
                raise ValueError("Pilot price reader requires a split-free series")
            h=hash_bytes()
            bars=[]
            with (directory/f"{symbol}.bars.jsonl").open("rb",buffering=0) as f:
                for line in f:
                    bar=Bar.model_validate_json(line)
                    if bar.day>LAYOUT_STOP or (bars and bar.day<=bars[-1].day) or bar.available_at<close_available(bar.day):
                        raise ValueError("Market prefix dates or availability differ")
                    bars.append(bar)
                    h.update(line)
                    if bar.day==LAYOUT_STOP:
                        break
            if not bars or bars[-1].day!=LAYOUT_STOP:
                raise ValueError("Development market prefix is incomplete")
            self.series[symbol]=tuple(bars)
            self.prefix_hashes[symbol]=h.hexdigest()


def source_inputs():
    task_manifest=load(TASKS/"manifest.json")
    development=TASKS/"development.jsonl"
    if sha256(development)!=task_manifest["partitions"]["development"]["sha256"]:
        raise ValueError("V3 Development tasks changed")
    evidence=[]
    fingerprints={"development_tasks":sha256(development),"task_manifest":sha256(TASKS/"manifest.json")}
    for folder, manifest_name, manifest_key, filename, file_key in (
        ("sec","source_manifest.json","sec_manifest_sha256","evidence.jsonl","evidence_sha256"),
        ("ir","manifest.json","ir_manifest_sha256","headlines.jsonl","headlines_sha256"),
        ("ir-releases","manifest.json","ir_releases_manifest_sha256","headlines.jsonl","headlines_sha256"),
    ):
        manifest=PUBLIC/folder/manifest_name
        path=PUBLIC/folder/filename
        if sha256(manifest)!=task_manifest[manifest_key] or sha256(path)!=load(manifest)[file_key]:
            raise ValueError(f"Frozen evidence source changed: {folder}")
        fingerprints[folder]=sha256(path)
        evidence.extend(Evidence.model_validate_json(line) for line in path.read_text().splitlines() if line)
    return read_tasks(development),EvidenceStore(evidence),task_manifest,fingerprints


def xbrl_packet(task, root):
    manifest=load(root/"manifest.json")
    entry=manifest["issuers"][task.symbol]
    path=root/f"{task.symbol}.json"
    if sha256(path)!=entry["sha256"]:
        raise ValueError("XBRL source changed")
    bundle=load(path)
    cutoff=close_available(task.decision_date)
    filings={key:item for key,item in bundle["filings"].items()
             if item["ticker"]==task.symbol and item.get("eligible_at") and item.get("report_date")
             and date.fromisoformat(item["report_date"])<=task.decision_date
             and date.fromisoformat(item["filing_date"])<=task.decision_date
             and datetime.fromisoformat(item["eligible_at"])<=cutoff
             and (not item.get("acceptance_datetime") or datetime.fromisoformat(item["acceptance_datetime"])<=cutoff)}
    facts=[row for row in bundle["facts"] if row["accession"] in filings and date.fromisoformat(row["end"])<=task.decision_date]
    return {"facts":facts,"filings":filings,"content_hash":entry.get("company_facts_sha256") or entry["sha256"]}


def unpack(packet):
    task=Task.model_validate(packet["task"])
    cutoff=close_available(task.decision_date)
    bars=tuple(Bar.model_validate(bar) for bar in packet["bars"])
    records=tuple(Evidence.model_validate(row) for row in packet["evidence"])
    if any(bar.day>task.decision_date or bar.available_at>cutoff for bar in bars):
        raise ValueError("Forbidden future market data in packet")
    if any(row.ticker!=task.symbol or not row.eligible_at or row.eligible_at>cutoff for row in records):
        raise ValueError("Forbidden evidence outside packet scope")
    x=packet["xbrl"]
    if any(datetime.fromisoformat(row["eligible_at"])>cutoff or row["ticker"]!=task.symbol
           for row in x["filings"].values()) or any(date.fromisoformat(row["end"])>task.decision_date for row in x["facts"]):
        raise ValueError("Forbidden future XBRL data in packet")
    return MarketView(task,bars),EvidenceStore(list(records)).view(task),FundamentalsView(
        task,x["content_hash"],x["facts"],x["filings"],PARTITION)


def encoder(root):
    location=root/"embedding"
    location.mkdir(exist_ok=True)
    link=location/"models"
    if not link.is_symlink():
        link.symlink_to(SOURCE/"data/local-models/models",target_is_directory=True)
    return LocalEmbedding(location,threads=2,batch=24,allow_download=False)


def document_chunks(item):
    title=item.source_metadata.get("title") or item.source_metadata.get("form","")
    kind="html" if item.source_type=="sec_filing" else "headline"
    url=str(item.source_url)
    raw=(PUBLIC/"sec/cache"/(digest(url.encode())+".body")).read_bytes() if kind=="html" else title.encode()
    verified=digest(raw)
    if kind=="html" and verified!=item.content_hash:
        raise ValueError("SEC snapshot content version does not match the cache")
    if not raw:
        raise ValueError("Empty authorized document")
    doc=Document(document_id=item.document_id,ticker=item.ticker,source_type=item.source_type,source_url=url,
                 eligible_at=item.eligible_at,pit_level=item.pit_level,dataset_partition=PARTITION,
                 raw_hash=verified,accession=item.document_id if kind=="html" else None,title=title,
                 first_seen_at=item.first_seen_at,version_attestation_sha256=item.source_metadata.get("version_attestation_sha256"),
                 version_attestation_source=item.source_metadata.get("version_attestation_source"),
                 format=kind,use_basis="Frozen SEC filing or IR headline from V3 Development snapshot")
    return chunk_document(doc,parse_document(doc,raw))


def prepare(work=WORK,out=OUT):
    work.mkdir(parents=True,exist_ok=False)
    out.mkdir(parents=True,exist_ok=True)
    all_tasks,store,manifest,fingerprints=source_inputs()
    selected=select_tasks(all_tasks)
    ids={t.task_id for t in selected}
    v3=DevelopmentMarket(PUBLIC/"market",[t.symbol for t in selected],manifest["market_manifest_sha256"])
    fingerprints["market_manifest"]=manifest["market_manifest_sha256"]
    fingerprints["market_prefixes"]=v3.prefix_hashes
    snapshots={}
    corpus={}
    for task in selected:
        view=store.view(task)
        records=[row for label in ("sec_filing","news") for row in view.visible(label)]
        corpus.update({row.evidence_id:row for row in records})
        body={"task":task.model_dump(mode="json"),
              "bars":[b.model_dump(mode="json") for b in v3.view(task).history],
              "evidence":[row.model_dump(mode="json") for row in records],
              "xbrl":xbrl_packet(task,PUBLIC/"sec/fundamentals")}
        unpack(body)
        path=work/"packets"/(task.task_id+".json")
        atomic(path,body)
        snapshots[task.task_id]=sha256(path)
    # Warm-up settlement is limited to non-Pilot tasks for the same stock.
    targets={task.symbol:close_available(task.decision_date) for task in selected}
    warm=[]
    unavailable=Counter()
    used=set()
    for task in all_tasks:
        if task.task_id in ids or close_available(task.decision_date)>=targets[task.symbol]:
            continue
        outcome=v3.outcome(task)
        if outcome.available_at>targets[task.symbol]:
            unavailable[task.symbol]+=1
            continue
        state=build_workflow(v3.view(task),MomentumBackend(BUDGET),BUDGET).invoke({})
        decision=Decision.model_validate(state["decision"])
        if state["errors"]:
            raise ValueError("Non-Pilot warm-up rule execution failed")
        warm.append((task,decision,outcome,state["evidence"]))
        used.add(task.task_id)
    emb=encoder(work)
    availability={}
    for arm in ("T3","T4"):
        with MemoryStore(work/f"memory-{arm}",emb) as db:
            replay=MemoryReplay(db,MemoryConfig("M3",PARTITION,arm,limit=3,context_bytes=1800))
            for record in warm:
                replay.enqueue(*record)
            replay.finish()
            if any(db.consistency().values()):
                raise ValueError("Warm-up memory index differs from its SQLite authority")
            counts={task.task_id:len(db.visible(ticker=task.symbol,cutoff=close_available(task.decision_date),
                                               partition=PARTITION,arm=arm)) for task in selected}
            if arm=="T3":
                availability=counts
            elif counts!=availability:
                raise ValueError("Both memory arms must start with identical episode content")
            material=[ep.model_dump(mode="json",exclude={"experiment_arm"}) for ep in db._all()]
            content=hash_bytes(compact_json(material).encode()).hexdigest()
            if arm=="T3":
                common_memory=content
            elif content!=common_memory:
                raise ValueError("Warm-up episode content differs by arm")
    atomic(out/"memory-coverage.json",{"source":"Non-Pilot deterministic MomentumBackend; no real LLM Reflection",
        "candidate_counts":availability,"histogram":dict(sorted(Counter(availability.values()).items())),
        "warmup_episodes":len(warm),"warmup_task_ids":sorted(used),"source_content_sha256":common_memory,
        "future_settlement_excluded":sum(unavailable.values()),"prediction_cold_start_count":sum(not n for n in availability.values())})
    # Fixed smoke selection uses only legal historical availability, after the 20 tasks are frozen.
    candidates=[task for task in selected if availability[task.task_id]>0
                and abs(v3.view(task).indicators()["return_5"])<=BUDGET.momentum_threshold]
    if not candidates:
        raise ValueError("No fixed Smoke task covers both Memory and D2 eligibility")
    smoke_ids=[candidates[0].task_id]
    arms=("T0","T1","T2","T3","T4")
    plan={arm:[task.task_id for task in selected] for arm in arms}
    atomic(out/"pilot-manifest.json",{"development_count":80,"pilot_count":20,"stocks":20,
        "sectors":sorted({ISSUERS[task.symbol][0] for task in selected}),"hash_key":PILOT_KEY+"<task_id>",
        "tasks":[task.model_dump(mode="json") for task in selected],"fingerprints":fingerprints,
        "prediction_packets":snapshots,"warmup_ids":sorted(used),"arm_order":list(arms),
        "smoke_tasks":smoke_ids,"execution_order":plan,"market_read_through":str(LAYOUT_STOP)})
    build_index(work,out)


def _init_worker(root):
    global _worker_encoder
    _worker_encoder=encoder(root)


def _encode_job(batch):
    return _worker_encoder.encode([row["text"] for row in batch])


def build_index(work=WORK,out=OUT):
    """Resume only the isolated index and encode missing rows with bounded workers."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from qdrant_client import models
    manifest=load(out/"pilot-manifest.json")
    corpus={}
    for key,expected in manifest["prediction_packets"].items():
        packet_path=work/"packets"/(key+".json")
        if sha256(packet_path)!=expected:
            raise ValueError("Frozen packet fingerprint differs")
        for raw in load(packet_path)["evidence"]:
            doc=Evidence.model_validate(raw)
            corpus[doc.evidence_id]=doc
    start=perf_counter()
    counts=Counter()
    chunks=[]
    for item in sorted(corpus.values(),key=lambda item:item.evidence_id):
        batch=document_chunks(item)
        if not batch:
            raise ValueError("Authorized V3 document produced no chunks")
        chunks.extend(batch)
        counts[item.source_type]+=1
    print("Pilot index", dict(counts), "chunks", len(chunks),flush=True)
    emb=encoder(work)
    expected={row["chunk_id"]:row for row in chunks}
    if len(expected)!=len(chunks):
        raise ValueError("Duplicate deterministic chunk IDs")
    with VectorStore(work/"rag",emb,"pilot_chunks") as index:
        existing={str(p.id):p.payload for p in index.scan()}
        if not existing.keys()<=expected.keys() or any(expected[key]!=row for key,row in existing.items()):
            raise ValueError("Existing Pilot index does not match frozen evidence")
        missing=[row for row in chunks if row["chunk_id"] not in existing]
        batches=[missing[i:i+128] for i in range(0,len(missing),128)]
        with ProcessPoolExecutor(max_workers=5,mp_context=get_context("spawn"),
                                 initializer=_init_worker,initargs=(work,)) as pool:
            for i,(batch,vectors) in enumerate(zip(batches,pool.map(_encode_job,batches),strict=True),1):
                index.client.upsert(index.name,[models.PointStruct(id=row["chunk_id"],vector=value,payload=row)
                                               for row,value in zip(batch,vectors,strict=True)],wait=True)
                if i%5==0 or i==len(batches):
                    print("Indexed",len(existing)+min(i*128,len(missing)),"/",len(chunks),flush=True)
        check={str(p.id):p.payload for p in index.scan()}
        if check!=expected:
            raise ValueError("Pilot index/source payload mismatch")
    atomic(work/"index-audit.json",{"corpus":"V3 Development SEC and issuer releases","documents":dict(counts),
        "chunks":len(chunks),"positions_valid":all(row["start"] and row["end"] for row in chunks),
        "index_elapsed_seconds":perf_counter()-start,"embedding_model":emb.metadata})
    print("Prepared",len(manifest["tasks"]),"warmup episodes",len(manifest["warmup_ids"]),flush=True)
