"""Frozen C1 and single C1 boundary-repair ablation, Calibration only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
import random
import re
from statistics import mean
from time import perf_counter

from . import financebench as fb
from . import financebench_compression_diagnosis as audit
from . import financebench_compression_v5 as v5

ORIGINAL_SEGMENTS = fb._selection_segments
BOOTSTRAP_SEED = 5101
BOOTSTRAP_DRAWS = 4000


def number_key(text):
    text = text.replace("−", "-")
    value = text.strip().replace(",", "")
    currency = next((x for x in "$€£" if x in value), "")
    value = value.replace(currency, "").strip() if currency else value
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1].strip().lstrip("+-")
    pct = value.endswith("%")
    try:
        return currency, Decimal(value.rstrip("%")), pct
    except InvalidOperation:
        return currency, value, pct


def _financial_rows(text):
    """Use exact physical lines or the existing dotted-table row boundaries."""
    for unit in v5._units(text, 0):
        if not unit["table_row"]:
            continue
        raw = unit["text"]
        numbers = list(v5.NUM.finditer(raw))
        if not 2 <= len(numbers) <= 12:
            continue
        start, end = unit["start"], unit["end"]
        # A flattened prose block that merely contains numbers is not a row.
        physical = "\n" in text and (start == 0 or text[start-1] == "\n") and (
            end == len(text) or text[end] in "\r\n")
        dotted = bool(v5.DOT.search(raw))
        if (physical or dotted) and re.search("[A-Za-z]", raw[:numbers[0].start()]):
            yield start, end, len(numbers)


def repaired_segments(question, query, text):
    original = ORIGINAL_SEGMENTS(question, query, text)
    words = list(re.finditer(r"\S+", text))
    if not original or not words:
        return original
    rows = list(_financial_rows(text))
    if not rows:
        return original

    def word_span(lo, hi):
        members = [i for i, token in enumerate(words) if token.start() < hi and token.end() > lo]
        return (members[0], members[-1]+1) if members else None

    result = []
    for option in original:
        if option["selection_reason"] == "retriever_candidate_fallback":
            result.append(option)
            continue
        spans = list(option["segments"])
        additions = []
        for left, right in spans:
            lo, hi = words[left].start(), words[right-1].end()
            for start, end, count in rows:
                overlap = min(end, hi)-max(start, lo)
                if overlap < 10 or not v5.NUM.search(text[max(lo, start):min(hi, end)]):
                    continue
                header = audit.period_span(text, start, count)
                if header:
                    additions.append(word_span(*header))
                if not (lo <= start and end <= hi):
                    additions.append(word_span(start, end))
        merged = fb._merge_segments([*spans, *(s for s in additions if s)])
        if merged != spans and len(merged) <= 4:
            excerpt, merged = fb._render_segments(text, merged)
            result.append({**option, "excerpt": excerpt, "segments": merged,
                           "selection_reason": "boundary_protected_c1"})
        # The V2 assembler retains its normal fair-share, reserve and fallback rules.
        result.append(option)
    return result


@contextmanager
def use_repair(enabled):
    if enabled:
        fb._selection_segments = repaired_segments
    try:
        yield
    finally:
        fb._selection_segments = ORIGINAL_SEGMENTS


def retention(label, case, result):
    """Match source spans and page-scoped numeric/currency tokens uniformly."""
    spans = audit.delivered_spans(case, result)
    full = total = bound = 0
    for page, row in v5._expected_rows(label):
        if audit.nonfinancial_target(row):
            continue
        total += 1
        matches = [(i, s, e) for i, s, e in audit.row_candidates(case, page, row)
                   if audit.cover(spans, i, s, e)]
        full += bool(matches)
        if any((head := audit.period_span(case["candidates"][i]["text"], s,
                                    len(list(v5.NUM.finditer(case["candidates"][i]["text"][s:e])))))
               and audit.cover(spans, i, *head) for i, s, e in matches):
            bound += 1
    source = defaultdict(list)
    for i, chunk_spans in spans.items():
        chunk = case["candidates"][i]
        source[chunk["document_id"], chunk["page"]].extend(chunk["text"][s:e] for s, e in chunk_spans)

    def keys(body):
        nums = [m.group() for m in v5.NUM.finditer(body.replace("−", "-"))]
        return {
            "year": set(v5.YEAR.findall(body)),
            "value": {number_key(n) for n in nums if not v5.YEAR.fullmatch(n.strip())},
            "unit": {v5.norm(m.group()) for m in v5.UNIT.finditer(body)}
        }
    denominators, retained = defaultdict(set), defaultdict(set)
    for e in label["evidence"]:
        page = e["document_id"], e["page"]
        items = [keys(piece) for piece in source.get(page, ())]
        for kind, target in keys(e["evidence_text"]).items():
            denominators[kind].update((page, val) for val in target)
            retained[kind].update((page, val) for val in target if any(val in x[kind] for x in items))
    return {"rows_corrected":full, "rows_corrected_total":total, "headers_verified":bound,
            **{f"{kind}_corrected_hits":len(retained[kind]) for kind in ("year","value","unit")},
            **{f"{kind}_corrected_total":len(denominators[kind]) for kind in ("year","value","unit")}}


def summarize(rows):
    result = v5._groups(rows, "calibration")
    for key, group in result.items():
        arm, budget = group["arm"], group["configured_context_budget_bytes"]
        subset = [r for r in rows if r["arm"] == arm and r["budget_bytes"] == budget
                  and r["mmr_lambda"] == group["mmr_lambda"]]
        group["rows_corrected"] = sum(r["rows_corrected"] for r in subset)
        group["rows_corrected_total"] = sum(r["rows_corrected_total"] for r in subset)
        group["coverage_corrected"] = group["rows_corrected"] / group["rows_corrected_total"]
        group["headers_verified"] = sum(r["headers_verified"] for r in subset)
        for kind in ("year","value","unit"):
            total = sum(r[f"{kind}_corrected_total"] for r in subset)
            group[f"{kind}_corrected"] = sum(r[f"{kind}_corrected_hits"] for r in subset) / total if total else None
    return result


def paired(rows, budget, metric, arm="D1"):
    grouped = defaultdict(dict)
    for row in rows:
        if row["budget_bytes"] == budget and row["arm"] in ("D0", arm):
            grouped[row["case_key"]][row["arm"]] = row
    by_company = defaultdict(list)
    directions = Counter()
    for pair in grouped.values():
        old, new = pair["D0"], pair[arm]
        if metric == "rows_corrected":
            denominator = old["rows_corrected_total"]
        else:
            denominator = int(old[metric] is not None)
        if not denominator:
            continue
        delta = new[metric]-old[metric]
        directions["better" if delta>0 else "worse" if delta<0 else "tie"] += 1
        by_company[old["company_key"]].append((delta, denominator))
    totals = [(sum(d for d,_ in entries), sum(n for _,n in entries))
              for entries in by_company.values()]
    if not totals:
        return {}
    rng = random.Random(BOOTSTRAP_SEED)
    sample = []
    for _ in range(BOOTSTRAP_DRAWS):
        drawn = rng.choices(totals, k=len(totals))
        sample.append(sum(d for d,_ in drawn)/sum(n for _,n in drawn))
    sample.sort()
    return {"delta":sum(d for d,_ in totals)/sum(n for _,n in totals),
            "ci95":[sample[99],sample[3899]], "clusters":len(totals), "draws":BOOTSTRAP_DRAWS,
            "case_directions":dict(directions), "scorable_cases":sum(directions.values())}


def run(data, output):
    archive = v5.readjson(audit.HISTORY/"freeze.json")
    diag = v5.readjson(output/"failure-attribution.json")
    cases = audit.validated_cases(data, archive)
    archived = {(r["case_id"],r["arm"],r["budget_bytes"]):r
                for r in v5.readrows(data/"calibration-deliveries.jsonl")
                if r["arm"] == "C1" or r["arm"] in diag["lambda_historical_replay"]
                and r["mmr_lambda"] == diag["lambda_historical_replay"][r["arm"]]}
    old_scores = {(r["case_key"],r["arm"],r["budget_bytes"]):r
                  for r in v5.readrows(audit.HISTORY/"calibration-rows.jsonl")
                  if r["arm"] in diag["lambda_historical_replay"]
                  and r["mmr_lambda"] == diag["lambda_historical_replay"][r["arm"]]}
    # Timing excludes tokenizer accounting for both arms. No neural scoring in D1.
    predictions = []
    changed_context = Counter()
    for i, case in enumerate(cases):
        for b in v5.BUDGETS:
            for arm in (("D0","D1") if i%2 == 0 else ("D1","D0")):
                started = perf_counter()
                with use_repair(arm == "D1"):
                    res = v5._c1(case["question"],case["candidates"],b)
                res["selection_latency_ms"] = (perf_counter()-started)*1000
                res["arm"] = arm
                if arm == "D0" and res["items"] != archived[case["case_id"],"C1",b]["items"]:
                    raise ValueError("D0 differs from archived C1")
                if arm == "D1":
                    changed_context[b] += res["items"] != archived[case["case_id"],"C1",b]["items"]
                predictions.append((case,res))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(v5.MODEL, local_files_only=True, trust_remote_code=False)
    class Proxy:
        def token_lengths(self, texts):
            return [len(tokenizer.encode(x, add_special_tokens=True)) for x in texts]
    proxy = Proxy()
    labels = {r["case_id"]:r for r in v5.readrows(data/"calibration-scoring-labels.jsonl")}
    rows = []
    for case, res in predictions:
        label = labels[case["case_id"]]
        res["proxy_evidence_tokens"] = sum(proxy.token_lengths([x["excerpt"] for x in res["items"]]))
        row = v5._score(case,res,label,proxy)
        rows.append({**row, **retention(label,case,res)})
    case_map = {case["case_id"]:case for case in cases}
    for (case_id,arm,b),res in archived.items():
        if arm == "C1":
            continue
        case = case_map[case_id]
        rows.append({**old_scores[v5.pseudo(case_id),arm,b], **retention(labels[case_id],case,res)})
    groups = summarize(rows)
    pairs = {arm:{str(b):{m:paired(rows,b,m,arm) for m in ("rows_corrected","delivered_page_recall")}
                  for b in v5.BUDGETS}
             for arm in ("D1","C2","C3","C4")}
    v5.writerows(output/"ablation-rows.jsonl", rows)
    v5.writejson(output/"summary.json", {"n":len(cases), "groups":groups, "pairs":pairs,
        "glm_requests":0, "glm_tokens":0, "test_opened":False, "d1_context_changed_cases":changed_context,
        "skipped":{"D2":"Semantic-only C2/C3 lost most complete rows and original encoding latency alone exceeded 2x C1. No additional semantic configuration was fit.",
                   "D3":"D2 was not run; no complementary benefit established."}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=audit.DATA)
    parser.add_argument("--out",type=Path,default=audit.OUTPUT)
    args = parser.parse_args()
    run(args.data,args.out)


if __name__ == "__main__":
    main()
