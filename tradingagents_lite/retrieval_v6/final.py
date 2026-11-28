"""Structure ablation and one-shot frozen public evaluation."""
from collections import Counter, defaultdict
from hashlib import sha256 as hash_bytes
from pathlib import Path
from statistics import fmean
import subprocess
from time import perf_counter

from ..data import sha256
from ..embedding_v4 import BGEM3Encoder
from ..rag import rrf
from ..retrieval_v4 import jsonl_read, legacy_rows, paired_comparison
from ..vectors import LocalEmbedding
from .engine import ROOT, SOURCE, V3_INDEX, V4_INDEX, M3, Search, fused, scope_fingerprint, union
from .experiment import (ARCHIVE_V4, LEGACY_V3, LEGACY_V4, eligible_stop, export_phase, feature_map,
                         inputs, load_models, package, project, read, score_all, score_logits,
                         split_ids, write, write_rows)
from .features import expansions, parse, rank
from .scoring import select, summarize


def model_for_small(work):
    root = work/"small-query-cache"
    root.mkdir(exist_ok=True)
    models = root/"models"
    if not models.is_symlink():
        models.symlink_to(SOURCE/"data/local-models/models",target_is_directory=True)
    return LocalEmbedding(root,device="cpu",threads=2,allow_download=False)


def audit_fixed(out, work):
    grouped = read(work/"feature-ablation-rows.json")["groups"]
    original = legacy_rows(LEGACY_V3/"calibration-S3.jsonl")
    m3 = [r for r in jsonl_read(LEGACY_V4/"calibration-rows.jsonl") if r["arm"]=="E3-m3"]
    parity = {}
    for arm,historical in (("S3",original),("E3",m3)):
        prior = {r["question_sha256"]:r for r in historical}
        current = grouped[arm+"/F0-residual-none"]
        parity[arm] = {"n":len(current),"matching_page_metrics":sum(r["page"]==prior[r["question_sha256"]]["page"] for r in current),
                       "matching_oracle":sum(r["candidate_oracle_at_5"]==prior[r["question_sha256"]]["candidate_oracle_at_5"] for r in current)}
    structure = None
    if (work/"structure-build.json").exists():
        from .engine import read_points
        from pypdf import PdfReader
        rows = {key:row for key,row,_ in read_points(Path(read(work/"structure-build.json")["database"]))}
        assert len(rows)==read(work/"structure-build.json")["audit"]["indexed"]
        for row in rows.values():
            assert row["start"][0]==row["end"][0]==f"pdf:{row['page']}:layout"
            assert len(row["source_spans"])==len(row["source_span_hashes"])
            assert all(a<b for a,b in row["source_spans"])
        sampled = sorted(rows.values(),key=lambda row:(row["document_id"],row["page"],row["chunk_id"]))
        sampled = sampled[::max(1,len(sampled)//24)]
        for row in sampled:
            pdf = PdfReader(SOURCE/"data/financebench-pdfs"/(row["document_id"]+".pdf"))
            layout = pdf.pages[row["page"]].extract_text(extraction_mode="layout") or ""
            assert hash_bytes(layout.encode()).hexdigest()==row["page_layout_sha256"]
            source = [layout[a:b] for a,b in row["source_spans"]]
            assert [hash_bytes(text.encode()).hexdigest() for text in source]==row["source_span_hashes"]
            assert "\n[… omitted …]\n".join(source)==row["text"]
        structure={"metadata_checked":len(rows),"independent_pdf_reextraction_checked":len(sampled),"violations":0}
    write(out/"baseline-audit.json", {**parity, **read(work/"baseline-parity.json"),"source_locator_audit":structure,
        "exact_search":read(work/"retrieval-parity.json") if (work/"retrieval-parity.json").exists() else None})


def structure_build(args):
    from .structure import extract_doc, structure_jobs, build_index
    work,out = Path(args.work),Path(args.output)
    reports = [read(out/(name+".json")) for name in ("feature-ablation","query-expansion-ablation")
               if (out/(name+".json")).exists()]
    if any(eligible_stop(r) for r in reports):
        return
    legacy = Search.from_sqlite(V4_INDEX)
    known = read(work/"input-fingerprints.json")["pdf_sha256"]
    questions = project("calibration", split_ids("calibration"))
    jobs = structure_jobs(legacy,known)
    if not (work/"structure-smoke.json").exists():
        counts = Counter()
        cal_companies = {q["company"] for q in questions.values()}
        subset = [j for j in jobs if j[2]["ticker"] in cal_companies][:6]
        for job in subset:
            _,audit = extract_doc(job)
            counts.update(audit)
        write(work/"structure-smoke.json",{"documents":len(subset),"counts":dict(counts)})
    smoke = read(work/"structure-smoke.json")
    if smoke["counts"].get("verified_bindings",0)<1 or smoke["counts"].get("row",0)<10:
        write(out/"chunking-ablation.json",{"status":"stopped_after_smoke","smoke":smoke})
        return
    if not (work/"structure-build.json").exists():
        db,audit = build_index(jobs,work,BGEM3Encoder(M3,device=args.device))
        write(work/"structure-build.json",{"database":str(db),"audit":audit})


def parse_arm(arm):
    prefix, _, recipe = arm.partition("/")
    stage,fusion,page_policy = (0,"residual","none")
    if recipe.startswith("F"):
        number,fusion,page_policy = recipe.split("-")
        stage=int(number[1:])
    return prefix, (stage,fusion,page_policy)


def structure_layers(extra, q, vector):
    bm = extra.lexical(q["query"],q["company"],160)
    dense = extra.dense(vector,q["company"],160)
    return bm,dense


def assemble(arm, base, extras=()):
    prefix, _ = parse_arm(arm)
    lex,dense = base
    channels = [(2,lex[0]),(2,dense[0])]
    raw = [key for key,_ in rrf([lex[0],dense[0]])]
    if prefix in ("S3","E3","Q0","T0"):
        return raw,raw,raw
    if prefix.startswith(("Q1","Q3","Q4")) or prefix=="T4":
        channels.extend((1,layer) for layer in lex[1:])
    if prefix.startswith(("Q2","Q3","Q4")) or prefix=="T4":
        channels.extend((1,layer) for layer in dense[1:])
    if prefix.startswith("T"):
        channels.extend((1,layer) for layer in extras)
        cap=150
    else:
        cap=int(prefix.split("-")[1])
    pool,pre = fused(channels,cap)
    return pool,pre,raw


def structure_compare(args):
    work,out=Path(args.work),Path(args.output)
    if not (work/"structure-build.json").exists():
        structure_build(args)
    if not (work/"structure-build.json").exists():
        return
    built=read(work/"structure-build.json")
    old=Search.from_sqlite(V4_INDEX)
    new=Search.from_sqlite(Path(built["database"]))
    rows={**old.rows,**new.rows}
    if set(old.rows)&set(new.rows) or not {r["document_id"] for r in new.rows.values()} <= {r["document_id"] for r in old.rows.values()}:
        raise ValueError("Structured index is not an isolated overlay of the frozen corpus")
    model,encoder,load=load_models(args.device)
    prior=read(work/"phase1-unscored.json")
    expanded=read(work/"phase2-unscored.json") if (work/"phase2-unscored.json").exists() else {}
    frozen=read(ARCHIVE_V4/"calibration-m3-candidates.json")["measurements"]
    best_exp=read(out/"query-expansion-ablation.json")["best"] if expanded else "E3/F0-residual-none"
    candidate_recipe=best_exp.split("/")[-1]
    t4="T4/"+(candidate_recipe if candidate_recipe.startswith("F") else "F0-residual-none")
    ids=split_ids("calibration")
    questions=project("calibration",ids)
    predictions,memo={},{}
    for i,q in enumerate(questions.values(),1):
        qid=q["id"]
        intent=parse(q["query"])
        start=perf_counter()
        forms=expansions(q["query"],intent)
        lex,dense,elapsed=old.retrieve(q,forms,encoder)
        frozen_arms=frozen[qid]["rankings"]["50"]
        lex[0],dense[0]=frozen_arms["bm25"],frozen_arms["dense"]
        vector=encoder.encode([q["query"]],query=True)[0]
        additions=structure_layers(new,q,vector)
        candidates={}
        for arm,types in (("T1",{"paragraph"}),("T2",{"paragraph","row"}),("T3",{"paragraph","row","block"}),
                          (t4,{"paragraph","row","block"})):
            extra=([pair for pair in ranking if new.rows[pair[0]]["unit_type"] in types][:35] for ranking in additions)
            channels=(lex,dense) if arm.startswith("T4") else ([lex[0]],[dense[0]])
            candidates[arm]=assemble(arm,channels,extra)
        shared=prior[qid]["E3/F0-residual-none"]["features"]
        scores={key:value["S"] for key,value in shared.items()}
        for observation in expanded.get(qid,{}).values():
            scores.update({key:value["S"] for key,value in (observation["features"] or {}).items()})
        missing=sorted({key for pool,_,_ in candidates.values() for key in pool}-scores.keys())
        found,usage=score_logits(model,q["query"],missing,rows)
        scores.update(found)
        predictions[qid]={"T0":prior[qid]["E3/F0-residual-none"]}
        for arm,(pool,pre,raw) in candidates.items():
            old.check(q,[key for key in pre if key in old.rows])
            new.check(q,[key for key in pre if key in new.rows])
            vectors=feature_map(intent,pool,scores,lex[0],dense[0],rows,memo)
            ranked=rank(vectors,rows,intent,*parse_arm(arm)[1])
            cost={"estimate_ms":(perf_counter()-start)*1000,"feature_ms":0.,
                  "peak_allocated_bytes":usage.get("peak_allocated_bytes")}
            predictions[qid][arm]=package(pool,raw,ranked,cost,pre_cap=pre,vectors=vectors)
        if i%10==0:
            print("Structure pools",i,flush=True)
    write(work/"structure-unscored.json",predictions)
    labels=project("calibration",ids,labels=True)
    groups=score_all(questions,predictions,rows,labels)
    report=export_phase("chunking-ablation",groups,out,work)
    report.update({"smoke":read(work/"structure-smoke.json"),"index":built["audit"],
                   "index_fingerprint":scope_fingerprint(Path(built["database"])),
                   "index_mode":"three logical overlays filtering a single independent Qdrant layout collection",
                   "models_load_ms":load})
    write(out/"chunking-ablation.json",report,replace=True)


def make_prediction(q, arm, old, new, encoder, model, temperature):
    prefix, recipe = parse_arm(arm)
    intent = parse(q["query"])
    if temperature=="cold":
        old.legal.cache_clear()
    tick=perf_counter()
    forms=expansions(q["query"],intent) if prefix.startswith("Q") and prefix!="Q0" or prefix=="T4" else [q["query"]]
    lex,dense,timing=old.retrieve(q,forms,encoder)
    extra=()
    rows=old.rows
    if prefix.startswith("T") and prefix!="T0":
        vector=encoder.encode([q["query"]],query=True)[0]
        types={"paragraph"} if prefix=="T1" else {"paragraph","row"} if prefix=="T2" else {"paragraph","row","block"}
        extra=([pair for pair in ranking if new.rows[pair[0]]["unit_type"] in types][:35]
               for ranking in structure_layers(new,q,vector))
        rows={**old.rows,**new.rows}
    pool,pre,raw=assemble(arm,(lex,dense),extra)
    old.check(q,[key for key in pre if key in old.rows])
    if new:
        new.check(q,[key for key in pre if key in new.rows])
    fusion_done=perf_counter()
    use_bge=prefix in {"S3","E3","Q0"} or prefix.startswith(("Q4","T")) or arm.endswith("/BGE")
    if use_bge:
        logits,usage=score_logits(model,q["query"],pool,rows)
        vectors=feature_map(intent,pool,logits,lex[0],dense[0],rows,{}) if recipe[0] else {
            key:{"S":value,"M":0,"Y":0,"F":0,"H":0,"N":0,"D":0} for key,value in logits.items()}
        ranked=rank(vectors,rows,intent,*recipe)
    else:
        usage={"milliseconds":0.,"peak_allocated_bytes":0,"peak_reserved_bytes":0}
        ranked=pool
        vectors={}
    timing.update({"total_ms":(perf_counter()-tick)*1000,"retrieval_with_fusion_ms":(fusion_done-tick)*1000,
                   "reranker_ms":usage["milliseconds"],"peak_allocated_bytes":usage["peak_allocated_bytes"],
                   "peak_reserved_bytes":usage["peak_reserved_bytes"]})
    return package(pool,[key for key in raw if key in set(pool)],ranked,timing,pre_cap=pre,vectors=vectors)


def measure_deployment(args, choice):
    work=Path(args.work)
    old=Search.from_sqlite(V4_INDEX)
    small=Search.from_sqlite(V3_INDEX)
    new=Search.from_sqlite(Path(read(work/"structure-build.json")["database"])) if choice.startswith("T") and not choice.startswith("T0") else None
    model,encoder,load=load_models(args.device)
    small_encoder=model_for_small(work)
    _,warm=model.score("What was operating income?",["Historical statement excerpt for warm-up only."])
    questions=list(project("calibration",split_ids("calibration")).values())
    selected={}
    for q in questions:
        selected.setdefault(q["company"],q)
    from .scoring import percentiles
    result={"startup_ms":load,"warmup_ms":warm["milliseconds"],"n":len(selected),"arms":{}}
    timing_arms = {"S3":(small,small_encoder),"E3":(old,encoder)}
    if (out := Path(args.output)/"query-expansion-ablation.json").exists():
        reduced = "Q4-100/F5-weighted-distinct"
        if reduced in read(out)["groups"]:
            timing_arms[reduced] = (old,encoder)
    timing_arms[choice] = (small,small_encoder) if choice.startswith("S3") else (old,encoder)
    for arm,(search,enc) in timing_arms.items():
        phases={}
        for temperature in ("cold","warm"):
            timings=[make_prediction(q,arm,search,new,enc,model,temperature)["latency"] for q in selected.values()]
            phases[temperature]=percentiles([row["total_ms"] for row in timings])
        result["arms"][arm]=phases
    result["within_latency_ceiling"] = [
        arm for arm,phases in result["arms"].items()
        if all(phases[temp]["mean"]<=1.25*result["arms"]["S3"][temp]["mean"] for temp in ("cold","warm"))
    ]
    result["passes"] = choice in result["within_latency_ceiling"]
    write(work/"deployment.json",result)
    return result


def fingerprints(work):
    answer = {"inputs":inputs(), "code":{file.name:sha256(file) for file in Path(__file__).parent.glob("*.py")},
              "references":{name:sha256(ROOT/"tradingagents_lite"/name) for name in (
                  "retrieval_v4.py","retrieval_v3.py","reranker_v3.py","rag.py","evidence.py","embedding_v4.py",
                  "rag_benchmark.py","retrieval_v2_chunks.py","vectors.py")}}
    if (work/"structure-build.json").exists():
        answer["layout"] = scope_fingerprint(Path(read(work/"structure-build.json")["database"]))
    return answer


def freeze(args):
    work,out=Path(args.work),Path(args.output)
    if (out/"freeze.json").exists():
        raise FileExistsError("Frozen configuration already exists")
    groups={}
    phases={}
    for name in ("feature-ablation","query-expansion-ablation","chunking-ablation"):
        fp=work/(name+"-rows.json")
        if fp.exists():
            phases[name]=read(fp)
            groups.update(phases[name]["groups"])
    if not groups:
        raise ValueError("Calibration results are missing")
    if not any(eligible_stop(read(out/(name+".json"))) for name in phases) and "chunking-ablation" not in phases:
        if not (out/"chunking-ablation.json").exists():
            raise ValueError("Complete the predefined structure gate before freezing")
    choice, held, folds=select(groups)
    audit_fixed(out,work)
    # Decide complementary hand-crafted gains on company clusters, without looking at Test.
    financial=phases["feature-ablation"]["groups"]
    complement=[]
    for arm,rows in financial.items():
        baseline=financial[arm.split("/")[0]+"/F0-residual-none"]
        improvement=defaultdict(float)
        for prev,cur in zip(baseline,rows,strict=True):
            improvement[cur["company_group"]]+=cur["page"]["recall@5"]-prev["page"]["recall@5"]
        complement.append(sum(v>0 for v in improvement.values()))
    largest=max(complement,default=0)
    ready=largest>=3 and len(folds)>=8 and summarize(held)["page"]["recall@5"]<.70
    if ready:
        raise ValueError("The conditional learning-to-rank gate passed; evaluate it before freezing")
    deployment=measure_deployment(args,choice)
    git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    write_rows(out/"calibration-rows.jsonl",[
        *({**r,"evaluation":"fixed_recipe"} for arm in groups.values() for r in arm),
        *({**r,"evaluation":"leave_one_company_out_selection"} for r in held)])
    profile={name:summarize(rows) for name,rows in groups.items()}
    selected_s=summarize(groups[choice])
    summary={"choice":choice,"calibration":selected_s,"out_of_fold":summarize(held),
             "folds":folds,"ranking_model":{"executed":False,"max_improved_companies":largest,
                      "companies":len(folds),"reason":"pre-registered gate not passed"},
             "deployment":deployment,"ranked_recipes":profile}
    write(work/"selection.json",summary)
    write(out/"freeze.json",{"git_sha":git_sha,"protocol_sha256":sha256(out/"protocol-freeze.json"),
        "fingerprints":fingerprints(work),"calibration_rows_sha256":sha256(out/"calibration-rows.jsonl"),
        "selection":choice,"out_of_fold_recall_at_5":summary["out_of_fold"]["page"]["recall@5"],
        "final_order":["load frozen models and indexes","project 99 question IDs without labels",
                       "rank first per-query pass","repeat for warm timing and compare",
                       "archive predictions locally","load scoring evidence","score all 99"],
        "return_k":5,"public_test_opened":False,"glm_requests":0,"glm_tokens":0})
    print({"frozen":choice,"recall_at_5":selected_s["page"]["recall@5"],"oof":summary["out_of_fold"]["page"]["recall@5"],
           "deployment_gate":deployment["passes"],"freeze_sha256":sha256(out/"freeze.json")},flush=True)


def test_once(args):
    work,out=Path(args.work),Path(args.output)
    frozen=read(out/"freeze.json")
    if (out/"public-test-rows.jsonl").exists() or (work/"public-test-started.json").exists():
        raise FileExistsError("Public Test was already started")
    if frozen["fingerprints"]!=fingerprints(work) or frozen["calibration_rows_sha256"]!=sha256(out/"calibration-rows.jsonl"):
        raise ValueError("Frozen inputs or source changed")
    if frozen["protocol_sha256"] != sha256(out/"protocol-freeze.json"):
        raise ValueError("Protocol differs")
    choice=frozen["selection"]
    opening=perf_counter()
    old=Search.from_sqlite(V3_INDEX if choice.startswith("S3") else V4_INDEX)
    new=Search.from_sqlite(Path(read(work/"structure-build.json")["database"])) if choice.startswith("T") and not choice.startswith("T0") else None
    index_load_ms=(perf_counter()-opening)*1000
    model,encoder,load=load_models(args.device,dense=not choice.startswith("S3"))
    if encoder is None:
        encoder=model_for_small(work)
    _,warmup=model.score("What was net income?",["Warm-up only."])
    write(work/"public-test-startup.json", {"index_load_ms":index_load_ms, **load, "warmup_ms":warmup["milliseconds"]})
    # Set the marker immediately before the question projection, after runtime preflight.
    write(work/"public-test-started.json",{"freeze_sha256":sha256(out/"freeze.json")})
    ids=split_ids("test")
    questions=project("test",ids)
    predictions={}
    for counter,q in enumerate(questions.values(),1):
        cold=make_prediction(q,choice,old,new,encoder,model,"cold")
        warm=make_prediction(q,choice,old,new,encoder,model,"warm")
        cold["latency"]={
            **{f"cold_{k}":v for k,v in cold["latency"].items() if k.endswith("_ms")},
            **{f"warm_{k}":v for k,v in warm["latency"].items() if k.endswith("_ms")},
            "peak_allocated_bytes":max(cold["latency"]["peak_allocated_bytes"] or 0,warm["latency"]["peak_allocated_bytes"] or 0),
            "peak_reserved_bytes":max(cold["latency"]["peak_reserved_bytes"] or 0,warm["latency"]["peak_reserved_bytes"] or 0),
            "ranking_mismatch":int(cold["ranked"][:10]!=warm["ranked"][:10]),
        }
        predictions[q["id"]]={"V6":cold}
        if counter%10==0:
            print("Public Test ranked",counter,flush=True)
    write(work/"public-test-predictions.json",predictions)
    score_existing(args)


def migration(rows, baseline):
    prior = {row["question_sha256"]: row["gold_page_ranks"] for row in baseline}
    counts = Counter()
    for row in rows:
        original = {item.get("key",item.get("page_hash")):item["after"] for item in prior[row["question_sha256"]]}
        for hit in row["gold_page_ranks"]:
            before, after = original.get(hit["key"]), hit["after"]
            if after is None or after>5:
                continue
            if before is None:
                counts["previously_absent"] += 1
            elif before<=5:
                counts["previously_top5"] += 1
            elif before<=10:
                counts["previously_6_10"] += 1
            elif before<=20:
                counts["previously_11_20"] += 1
            else:
                counts["previously_21_plus"] += 1
    return dict(counts)


def score_existing(args):
    work,out=Path(args.work),Path(args.output)
    frozen=read(out/"freeze.json")
    marker=read(work/"public-test-started.json")
    if marker["freeze_sha256"]!=sha256(out/"freeze.json"):
        raise ValueError("Prediction freeze differs")
    if (out/"public-test-rows.jsonl").exists():
        raise FileExistsError("Public scores are already archived")
    choice=frozen["selection"]
    old=Search.from_sqlite(V3_INDEX if choice.startswith("S3") else V4_INDEX)
    new=Search.from_sqlite(Path(read(work/"structure-build.json")["database"])) if choice.startswith("T") and not choice.startswith("T0") else None
    rows={**old.rows,**(new.rows if new else {})}
    predictions=read(work/"public-test-predictions.json")
    ids=split_ids("test")
    questions=project("test",ids)
    scored=score_all(questions,predictions,rows,project("test",ids,labels=True))["V6"]
    archive = {"R3": legacy_rows(LEGACY_V3/"test-S0.jsonl"),"S3":legacy_rows(LEGACY_V3/"test-S3.jsonl"),
               "E3":[r for r in jsonl_read(LEGACY_V4/"test-rows.jsonl") if r["arm"]=="E3-m3"]}
    paired={name:paired_comparison(scored,baseline) for name,baseline in archive.items()}
    deltas = {arm:{r["question_sha256"]:r["delta_recall_at_5"] for r in cmp["paired"]} for arm,cmp in paired.items()}
    for row in scored:
        row["paired_delta"]={arm:values[row["question_sha256"]] for arm,values in deltas.items()}
    stats=summarize(scored)
    from ..retrieval_v4 import summarize_rows
    comparisons={arm:summarize_rows(baseline) for arm,baseline in archive.items()}
    summary={**read(work/"selection.json"),"public_baselines":comparisons,"public_v6":stats,"paired":paired,
             "freeze_sha256":sha256(out/"freeze.json"),"public_test_reused":True,"glm_requests":0,"glm_tokens":0,
             "indexes":frozen["fingerprints"]["inputs"]["indexes"],"layout_index":frozen["fingerprints"].get("layout"),
             "cold_warm_top10_mismatches":sum(r["latency"]["ranking_mismatch"] for r in scored)}
    if (work/"public-test-startup.json").exists():
        summary["public_startup"]=read(work/"public-test-startup.json")
    write(out/"summary.json",summary)
    fixed = read(work/"feature-ablation-rows.json")["groups"]
    fixed_migrations = {arm:migration(group,fixed[arm.split("/")[0]+"/F0-residual-none"])
                        for arm,group in fixed.items()}
    write(out/"failure-attribution.json",{
        "public_counts":stats["loss_attribution"],"public_indicator_proxies":stats["failure_proxies"],
        "note":"Potential mismatch categories use feature contrasts only. They are not causal adjudications.",
        "public_rrf_to_final":stats["rank_migration"],"public_e3_to_v6":migration(scored,archive["E3"]),
        "fixed_pool_raw_bge_to_variant":fixed_migrations,
        "calibration_fixed_pool_migration":read(out/"feature-ablation.json")["groups"],
        "baseline_test":{name:base["loss_attribution"] for name,base in comparisons.items()},
        "pit_violations":0,"source_locator_violations":0,
    })
    write_rows(out/"public-test-rows.jsonl",scored)
    print({"public_n":len(scored),"page":stats["page"],"oracle":stats["candidate_oracle_at_5"]},flush=True)


def finish(args):
    actions={"build":structure_build,"structure":structure_compare,"freeze":freeze,
             "test":test_once,"score-existing":score_existing}
    if args.stage=="report":
        from .report import generate
        generate(Path(args.output))
    else:
        actions[args.stage](args)
