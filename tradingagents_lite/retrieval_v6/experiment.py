"""Phased offline validation. Public Test is gated by a durable frozen plan."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256 as hash_bytes
import json
import os
from pathlib import Path
import re
import subprocess
from time import perf_counter

from ..data import sha256
from ..embedding_v4 import BGEM3Encoder
from ..rag import rrf
from ..rag_benchmark import split
from ..reranker_v3 import BGEReranker
from ..retrieval_v4 import jsonl_read, legacy_rows, paired_comparison
from .engine import ROOT, SOURCE, V3, V4, DATASET, M3, RERANKER, V3_INDEX, V4_INDEX
from .engine import Search, fused, scope_fingerprint, union
from .features import document_features, expansions, features, parse, rank, top5
from .scoring import metrics, select, summarize, truth_for

OUT = ROOT / "results/a1-v6-financial-retrieval"
WORK = ROOT / "data/a1-v6/001"
LEGACY_V3 = ROOT / "results/a1-retrieval-v3"
LEGACY_V4 = ROOT / "results/a1-retrieval-v4"
ARCHIVE_V3 = V3 / "002"
ARCHIVE_V4 = V4 / "003"
ID_KEY = re.compile(rb'"financebench_id"\s*:\s*"([^"]+)"')
FEATURE_RECIPES = [(0,"residual","none")] + [
    (stage, fusion, "none") for stage in range(1,6) for fusion in ("residual","weighted")
] + [(stage,"weighted",page) for stage in (0,5) for page in ("distinct","soft")]


def read(path):
    return json.loads(path.read_text())


def write(path, body, *, replace=False):
    with path.open("w" if replace else "x", encoding="utf-8") as stream:
        json.dump(body, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def split_ids(group):
    if group == "calibration":
        return read(ARCHIVE_V3/"calibration-candidates.json")["question_ids"]
    return [row["id"] for row in jsonl_read(LEGACY_V3/"test-S3.jsonl")]


def project(group, ids, labels=False):
    """Project only the chosen public partition. Never return the answer field."""
    result = {}
    with DATASET.open("rb") as stream:
        for line in stream:
            match = ID_KEY.search(line)
            if not match or match[1].decode() not in ids:
                continue
            raw = json.loads(line)
            qid = raw["financebench_id"]
            if qid in result or split(raw["company"]) != group:
                raise ValueError("Frozen public partition differs")
            result[qid] = raw["evidence"] if labels else {"id": qid, "company": raw["company"], "query": raw["question"]}
    if set(result) != set(ids) or len(result) != (49 if group=="calibration" else 99):
        raise ValueError("Frozen denominator differs")
    return result


def inputs():
    known = read(LEGACY_V3/"freeze.json")["inputs"]["A1_V2_inputs"]
    if sha256(DATASET) != known["dataset_sha256"][DATASET.name]:
        raise ValueError("FinanceBench fingerprint differs")
    if sha256(V3_INDEX) != known["original_index"]["collection/chunks/storage.sqlite"]:
        raise ValueError("V3 index differs")
    v4 = read(LEGACY_V4/"freeze.json")
    files = {name: sha256(M3/name) for name in v4["model"]["files_sha256"]}
    if files != v4["model"]["files_sha256"]:
        raise ValueError("BGE-M3 model differs")
    models = {file.name: sha256(file) for file in RERANKER.iterdir() if file.is_file()}
    return {"dataset": sha256(DATASET),"indexes":{"S3":scope_fingerprint(V3_INDEX),
             "E3":scope_fingerprint(V4_INDEX)}, "m3":files, "reranker":models,
            "historical_freeze":{"v3":sha256(LEGACY_V3/"freeze.json"),"v4":sha256(LEGACY_V4/"freeze.json")},
            "pdf_sha256":known["pdf_sha256"]}


def check_archives(search):
    from ..retrieval_v4 import source_fingerprint
    rows = [{"point_id": key, **search.rows[key]} for key in search.keys]
    expected = read(LEGACY_V4/"freeze.json")["index"]
    # V4's persisted payload and vector hashes use the Qdrant Point representation.
    if (len(rows),len({r["document_id"] for r in rows})) != (36622,83):
        raise ValueError("Unexpected frozen index population")
    if source_fingerprint(rows) != expected["payload_sha256"]:
        raise ValueError("V4 payload fingerprint differs")
    vector_hash = hash_bytes()
    for key, vector in zip(search.keys, search.matrix, strict=True):
        vector_hash.update(key.encode())
        vector_hash.update(vector.astype("<f4", copy=False).tobytes())
    return vector_hash.hexdigest() == expected["vector_sha256"]



def load_models(device, *, dense=True):
    timings = {}
    start = perf_counter()
    model = BGEReranker(RERANKER, device)
    timings["reranker_ms"] = (perf_counter()-start)*1000
    start = perf_counter()
    encoder = BGEM3Encoder(M3, device=device) if dense else None
    timings["encoder_ms"] = (perf_counter()-start)*1000 if dense else 0
    return model, encoder, timings


def score_logits(model, query, keys, rows):
    logits, usage = model.score(query, [rows[key]["text"] for key in keys])
    return dict(zip(keys,logits,strict=True)), usage


def feature_map(intent, pool, logits, bm, dense, rows, memo):
    bm_r = {key: i for i,(key,_) in enumerate(bm,1)}
    de_r = {key: i for i,(key,_) in enumerate(dense,1)}
    result = {}
    for key in pool:
        if key not in memo:
            memo[key] = document_features(rows[key])
        result[key] = features(intent, memo[key], logits[key], bm_r.get(key),de_r.get(key))
    return result


def recipe_name(recipe):
    stage,fusion,pages = recipe
    return f"F{stage}-{fusion}-{pages}"


def package(pool, before, ranked, latency, *, pre_cap=None, vectors=None):
    if len(set(pool)) != len(pool) or set(ranked) != set(pool) or len(ranked) != len(pool):
        raise ValueError("Ranking must be a permutation of the legal candidate set")
    return {"pool":pool, "before":before, "ranked":ranked, "return":top5(ranked),
            "pre_cap":pre_cap if pre_cap is not None else pool, "latency":latency, "features":vectors}


def phase1(questions, search, model, work):
    old = read(ARCHIVE_V3/"calibration-unscored.json")["S3"]
    lexical = read(ARCHIVE_V3/"calibration-candidates.json")["measurements"]
    m3 = read(ARCHIVE_V4/"calibration-m3-candidates.json")["measurements"]
    result, parity, memo = {}, Counter(), {}
    for number,q in enumerate(questions.values(),1):
        result[q["id"]] = {}
        intent = parse(q["query"])
        for base, cache in (("S3",lexical),("E3",m3)):
            record = cache[q["id"]]
            layers = record["rankings"]["50"]
            bm,de = layers["bm25"],layers["dense"]
            pool = union(bm,de)
            before = [key for key,_ in rrf([bm,de])]
            search.check(q,pool)
            if base=="S3":
                saved = old[q["id"]]["rank"]
                if set(saved["pool"]) != set(pool):
                    raise ValueError("S3 candidate set changed")
                logits = saved["scores"]
                usage = {"milliseconds": old[q["id"]]["latency"]["hot"]["model_ms"]}
                parity["s3_archive_order"] += sorted(pool,key=lambda key:(-logits[key],key)) == saved["ranked"]
            else:
                logits,usage = score_logits(model,q["query"],pool,search.rows)
            start = perf_counter()
            vectors = feature_map(intent,pool,logits,bm,de,search.rows,memo)
            feature_ms = (perf_counter()-start)*1000
            for recipe in FEATURE_RECIPES:
                tick = perf_counter()
                ranked = rank(vectors, search.rows,intent,*recipe)
                reference = record["timing"]["hot"]["50"]
                retrieval_ms = reference.get("retrieval_total_ms",sum(reference.get(name,0.) for name in
                    ("scope_ms","bm25_ms","encode_ms","dense_ms")))
                cost = (perf_counter()-tick)*1000 + (feature_ms if recipe[0] else 0)
                latency = {"estimate_ms":retrieval_ms+usage["milliseconds"]+cost,
                           "feature_ms":cost,"peak_allocated_bytes":usage.get("peak_allocated_bytes")}
                result[q["id"]][base+"/"+recipe_name(recipe)] = package(pool,before,ranked,latency,vectors=vectors)
        if number%10 == 0:
            print("Fixed pools",number,flush=True)
    write(work/"phase1-unscored.json", result)
    write(work/"baseline-parity.json", dict(parity))
    return result


def phase2(questions, search, model, encoder, phase1_records, work):
    m3 = read(ARCHIVE_V4/"calibration-m3-candidates.json")["measurements"]
    output, parity, memo = {}, Counter(), {}
    recipes = [r for r in FEATURE_RECIPES if r[1]=="residual" or r[0] in (0,5)]
    for number,q in enumerate(questions.values(),1):
        qid = q["id"]
        intent = parse(q["query"])
        lex,dense,elapsed = search.retrieve(q, expansions(q["query"],intent),encoder)
        base = m3[qid]["rankings"]["50"]
        frozen_lex, frozen_dense = base["bm25"],base["dense"]
        parity["bm25_top50_order"] += [k for k,_ in lex[0]] == [k for k,_ in frozen_lex]
        parity["exact_dense_top50_order"] += [k for k,_ in dense[0]] == [k for k,_ in frozen_dense]
        parity["exact_dense_top50_set"] += set(k for k,_ in dense[0]) == set(k for k,_ in frozen_dense)
        lex[0],dense[0] = frozen_lex,frozen_dense
        groups = {}
        for arm,a,b in (("Q1",True,False),("Q2",False,True),("Q3",True,True)):
            channels = [(2,lex[0]),(2,dense[0])]
            if a:
                channels.extend((1,layer) for layer in lex[1:])
            if b:
                channels.extend((1,layer) for layer in dense[1:])
            for cap in (100,150):
                groups[f"{arm}-{cap}"] = fused(channels,cap)
        full_keys = sorted({key for pool,_ in groups.values() for key in pool})
        frozen = phase1_records[qid]["E3/F0-residual-none"]["features"]
        new_keys = [key for key in full_keys if key not in frozen]
        logits,usage = score_logits(model,q["query"],new_keys,search.rows)
        logits.update({key:row["S"] for key,row in frozen.items()})
        output[qid] = {}
        for arm,(pool,pre) in groups.items():
            search.check(q,pre)
            vectors = feature_map(intent,pool,logits,lex[0],dense[0],search.rows,memo)
            base_cost = phase1_records[qid]["E3/F0-residual-none"]["latency"]["estimate_ms"]
            estimated = {"estimate_ms":sum(elapsed.values())+base_cost+usage["milliseconds"]*len(set(pool)&set(new_keys))/max(1,len(new_keys)),
                         "feature_ms":0.,"peak_allocated_bytes":usage.get("peak_allocated_bytes")}
            output[qid][arm] = package(pool,pool,pool,{"estimate_ms":sum(elapsed.values()),"feature_ms":0.},pre_cap=pre)
            for recipe in (recipes if arm.startswith("Q3") else [recipes[0]]):
                name = arm+"/BGE" if not recipe[0] and recipe[2]=="none" else arm.replace("Q3","Q4")+"/"+recipe_name(recipe)
                ranked = rank(vectors,search.rows,intent,*recipe)
                output[qid][name] = package(pool,pool,ranked,estimated,pre_cap=pre,vectors=vectors)
        if number%10 == 0:
            print("Expansion pools",number,flush=True)
    write(work/"phase2-unscored.json",output)
    write(work/"retrieval-parity.json",dict(parity))
    return output


def score_all(questions, predictions, search_rows, labels):
    truth = truth_for(labels,search_rows,SOURCE)
    grouped = defaultdict(list)
    for qid, q in questions.items():
        for arm, output in predictions[qid].items():
            grouped[arm].append(metrics(q,arm,output,search_rows,truth[qid]))
    return grouped


def export_phase(name, grouped, out, work):
    best,held,folds = select(grouped)
    values = {key:summarize(rows) for key,rows in grouped.items()}
    report = {"groups":values,"best":best,"oof":summarize(held),"folds":folds,
              "oof_company_count":len(folds)}
    write(out/(name+".json"),report)
    write(work/(name+"-rows.json"), {"groups":grouped,"oof":held})
    return report


def eligible_stop(report):
    winner = report["groups"][report["best"]]
    return winner["page"]["recall@5"] >= .70 and report["oof"]["page"]["recall@5"] >= .70 and winner["candidate_oracle_at_5"]>=.90


def calibrate(args):
    out,work = Path(args.output),Path(args.work)
    work.mkdir(parents=True,exist_ok=False)
    manifest = inputs()
    write(work/"input-fingerprints.json",manifest)
    ids = split_ids("calibration")
    questions = project("calibration",ids)
    search = Search.from_sqlite(V4_INDEX)
    print("Frozen payload and vector check",check_archives(search),flush=True)
    model,encoder,load = load_models(args.device)
    write(work/"model-startup.json",load)
    first = phase1(questions,search,model,work)
    labels = project("calibration",ids,labels=True)
    arms = score_all(questions,first,search.rows,labels)
    one = export_phase("feature-ablation",arms,out,work)
    if not eligible_stop(one):
        second = phase2(questions,search,model,encoder,first,work)
        for qid in second:
            second[qid]["Q0"] = first[qid]["E3/F0-residual-none"]
        arms2 = score_all(questions,second,search.rows,labels)
        two = export_phase("query-expansion-ablation",arms2,out,work)
    else:
        two = None
    print({"phase1":one["best"], "phase2":two and two["best"]},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("calibrate","build","structure","freeze","test","score-existing","report"))
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--work",default=str(WORK))
    parser.add_argument("--output",default=str(OUT))
    args = parser.parse_args()
    if args.stage == "calibrate":
        calibrate(args)
    else:
        from .final import finish
        finish(args)

if __name__ == "__main__":
    main()
