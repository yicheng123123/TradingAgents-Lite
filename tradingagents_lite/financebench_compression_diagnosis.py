"""Calibration-only attribution for the frozen A1-V5 compression experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import re
from statistics import mean
import unicodedata

from . import financebench as fb
from . import financebench_compression_v5 as v5

BASE = "9e659536eba66baa97f3ff868ed678160f79a7bd"
HISTORY = v5.ROOT / "results/a1-v5-context-compression"
DATA = v5.ROOT.parent / "tradingagents-a1-v5-context-compression" / "data/a1-v5-context-compression/001"
OUTPUT = v5.ROOT / "results/a1-v5.1-compression-diagnosis"
TOKEN = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?%?|[^\W\d_]+|\.{2,}|[^\s]", re.UNICODE)
PERIOD = re.compile(r"\b(?:years?\s+ended|fiscal\s+years?|as\s+of)\b", re.I)
SCALE = re.compile(r"\b(?:(?:USD|GBP|EUR)\s+(?:in\s+)?|in\s+)(?:thousands?|millions?|billions?|trillions?)\b"
                   r"|[$€£]\s*(?:in\s+)?(?:thousands?|millions?|billions?|trillions?)\b", re.I)
DELIVERY_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sept?\.?|Oct\.?|Nov\.?|Dec\.?)"
    r"\s+\d{1,2},?\s+((?:19|20)\d{2})\b", re.I)


DATE_LABEL = re.compile(
    r"(?:\(?)(?:(?:for\s+the\s+)?(?:(?:fiscal\s+)?years?|quarters?|months?)\s+ended\s+|as\s+of\s+|at\s+)?"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|sept?\.?|oct\.?|nov\.?|dec\.?)\s+"
    r"(?:[1-9]|[12]\d|3[01])(?:\s*,?\s*(?:19|20)\d{2}(?:\s*(?:,|and|&)\s*(?:19|20)\d{2})*)?[\s,.)]*",
    re.I,
)
SECTION_LABEL = re.compile(
    r"(?:item\s+\d+[a-z]?\.?(?:\s+financial statements and supplementary data)?"
    r"|form\s+10-[kq])[\s.]*", re.I,
)


def nonfinancial_target(row):
    """Exclude unambiguous date/section artifacts in the old reference labels."""
    if DATE_LABEL.fullmatch(row):
        return "date_fragment"
    if SECTION_LABEL.fullmatch(row):
        return "section_heading"
    return None


def canonical(token):
    return unicodedata.normalize("NFKC", token).casefold().replace("−", "-")


def tokens(text):
    # Normalize typographic variants of table leaders. Preserve punctuation,
    # ordering, signed numbers, decimal places and parentheses.
    text = text.replace("−", "-")
    return [(canonical(m.group()), m.start(), m.end()) for m in TOKEN.finditer(text)
            if not re.fullmatch(r"[.·…]+", m.group()) or m.group() == "."]


def row_spans(row, text):
    needle = [x[0] for x in tokens(row)]
    hay = tokens(text)
    if not needle:
        return []
    words = [x[0] for x in hay]
    n = len(needle)
    return [(hay[i][1], hay[i+n-1][2]) for i, word in enumerate(words)
            if word == needle[0] and words[i:i+n] == needle]


def delivered_spans(case, result):
    by_id = {c["chunk_id"]: i for i, c in enumerate(case["candidates"])}
    return {by_id[item["_chunk_id"]]: item["_source_spans"] for item in result["items"]}


def row_candidates(case, page, row):
    return [(i, *span) for i, chunk in enumerate(case["candidates"])
            if (chunk["document_id"], chunk["page"]) == page
            for span in row_spans(row, chunk["text"])]


def cover(spans, ci, start, end):
    return any(left <= start and end <= right for left, right in spans.get(ci, ()))


def period_span(text, start, cells):
    prefix_start = max(0, start-220)
    prefix = text[prefix_start:start]
    for cue in reversed(list(PERIOD.finditer(prefix))):
        remainder = prefix[cue.end():]
        dates = list(DELIVERY_DATE.finditer(remainder))
        years = v5.YEAR.findall(remainder)
        if len(years) != cells or len(set(years)) != cells or len(remainder) > 145:
            continue
        non_dates = DELIVERY_DATE.sub(r"\1", remainder)
        if any(not v5.YEAR.fullmatch(m.group().strip()) for m in v5.NUM.finditer(non_dates)):
            continue
        if dates and len(dates) != cells:
            continue
        # Use only exact same-chunk source positions. Never infer currencies or labels.
        start_cue = cue.start()
        scale = list(SCALE.finditer(prefix, 0, start_cue))
        if scale and start_cue-scale[-1].end() <= 45:
            start_cue = scale[-1].start()
        return prefix_start+start_cue, start
    return None


def trace_semantic(question, candidates, encoded, budget, arm, lam):
    """Replay the V5 greedy selector. Check every final span against V5 archive."""
    chosen, remaining, events, seen = [], list(range(len(encoded))), defaultdict(set), set()
    mapping, uncertain = {}, set()
    share = max(256, v5._cap(budget) // max(1, len(candidates)))
    while remaining:
        viable = []
        for index in remaining:
            unit = encoded[index]
            ci = unit["candidate_index"]
            body = v5.norm(unit["text"])
            if not body or body in seen:
                events[index].add("duplicate_text")
                continue
            if any(old["candidate_index"] == ci and
                   max(0, min(unit["end"], old["end"])-max(unit["start"], old["start"])) /
                   max(1, min(unit["end"]-unit["start"], old["end"]-old["start"])) > .90
                   for old in (encoded[i] for i in chosen)):
                events[index].add("span_overlap")
                continue
            score = max(0, unit["relevance"]) if arm == "C2" else v5._mmr(
                unit, [encoded[i] for i in chosen], lam)
            if score <= 0:
                events[index].add("zero_relevance" if max(0, unit["relevance"]) == 0
                                  else "mmr_nonpositive")
                continue
            trial, notes = defaultdict(list), set()
            for selected in [*(encoded[i] for i in chosen), unit]:
                rank = selected["candidate_index"]
                trial[rank].append((selected["start"], selected["end"]))
                if arm == "C4":
                    if selected["header_status"] == "bound":
                        trial[rank].append(selected["header_span"])
                    elif selected["table_row"] and selected["header_status"] == "uncertain":
                        notes.add(rank)
            rendered = v5._render(candidates, trial, notes)
            size = v5._ctx(rendered)
            if size > v5._cap(budget) or v5._request(question, rendered) > v5.MAX_REQUEST:
                events[index].add("request_or_context_budget")
                continue
            item = next(x for x in rendered if x["_chunk_id"] == candidates[ci]["chunk_id"])
            if arm != "C4" and len(v5.compact(v5._wire([item])).encode()) > share:
                events[index].add("per_chunk_share")
                continue
            prior = v5._ctx(v5._render(candidates, mapping, uncertain))
            density = score / max(1, size-prior) if arm == "C4" else score
            viable.append(((density, score, -ci, -unit["start"]), index, trial, notes))
            events[index].add("feasible")
        if not viable:
            break
        _, index, mapping, uncertain = max(viable, key=lambda x: x[0])
        remaining.remove(index)
        chosen.append(index)
        seen.add(v5.norm(encoded[index]["text"]))
        events[index].add("selected")
    return v5._render(candidates, mapping, uncertain), chosen, events


def expected_scope(case, result, page, row):
    return any(v5.norm(row) in v5.norm(item["excerpt"].removesuffix(v5.HEADER_NOTE).removesuffix(v5.CUT))
               for item in result["items"] if (item["document_id"], item["page"]) == page)


def attribution(case, label, result, units, events):
    spans = delivered_spans(case, result)
    rows = []
    for index, (page, row) in enumerate(v5._expected_rows(label)):
        locations = row_candidates(case, page, row)
        complete = [i for i, u in enumerate(units)
                    if any(u["candidate_index"] == ci and u["start"] <= s and e <= u["end"]
                           for ci, s, e in locations)]
        delivered = [(ci, s, e) for ci, s, e in locations if cover(spans, ci, s, e)]
        diagnosed_headers = [(ci, *hdr) for ci, s, e in delivered
                             if (hdr := period_span(
                                 case["candidates"][ci]["text"], s,
                                 len(list(v5.NUM.finditer(case["candidates"][ci]["text"][s:e])))
                             ))]
        original = expected_scope(case, result, page, row)
        prior_flags = {key for i in complete for key in events.get(i, ())}
        ranks_with_alternate_header = {
            j+1 for ci, _, _ in locations for j, other in enumerate(case["candidates"])
            if ci != j and (other["document_id"], other["page"]) == page
            and PERIOD.search(other["text"]) and v5.YEAR.search(other["text"])
        }
        if delivered:
            if not original:
                stage = "F7"
            elif diagnosed_headers and not any(cover(spans, ci, s, e) for ci, s, e in diagnosed_headers):
                stage = "F6"
            else:
                stage = "delivered"
        elif not locations:
            stage = "F1"
        elif not complete:
            stage = "F2"
        else:
            # Classify by the deepest barrier reached by a complete variant.
            # A feasible unit can later become budget-blocked or redundant.
            final = _final_barriers(complete, events)
            if "budget" in final:
                stage = "F5"
            elif "redundancy" in final:
                stage = "F4"
            else:
                stage = "F3"
        rows.append({
            "row": index + 1, "first_loss": stage, "raw_locations": len(locations),
            "nonfinancial_target": nonfinancial_target(row),
            "source_spans": [[ci+1, s, e] for ci, s, e in locations],
            "complete_units": len(complete), "source_span_delivery": bool(delivered),
            "max_unit_relevance": max((round(units[i]["relevance"], 5) for i in complete), default=None),
            "other_same_page_ranks_with_period_cue": sorted(ranks_with_alternate_header),
            "v5_text_match": original, "observable_period_headers": len(diagnosed_headers),
            "selection_events": sorted(prior_flags),
        })
    return rows


def _final_barriers(indices, events):
    barriers = set()
    for index in indices:
        flags = events.get(index, ())
        if {"request_or_context_budget", "per_chunk_share"}.intersection(flags):
            barriers.add("budget")
        elif {"span_overlap", "duplicate_text", "mmr_nonpositive"}.intersection(flags):
            barriers.add("redundancy")
        else:
            barriers.add("ranking")
    return barriers


def lengths(values):
    if not values:
        return {}
    ordered = sorted(values)
    return {"n": len(values), "min": ordered[0], "mean": mean(values),
            "p50": ordered[(len(ordered)-1)//2], "p95": ordered[math.ceil(.95*len(ordered))-1],
            "max": ordered[-1]}


def binding_decisions(text, row_units):
    """Distinguish an inherited span, a local cue check, and no qualifying cue."""
    cues = list(fb._HEADER_CUE.finditer(text))
    counts = Counter()
    prev = None
    for unit in row_units:
        start = unit["start"]
        intervening = any(prev and prev["end"] <= cue.start() < start for cue in cues)
        nearby = any(0 <= start-cue.end() <= 700 for cue in cues)
        if prev and not intervening and prev["header_span"]:
            counts["inherited_binding"] += 1
        elif nearby:
            counts["local_bind_attempt"] += 1
            counts["local_bind_success"] += unit["header_status"] == "bound"
        else:
            counts["no_nearby_v5_cue"] += 1
        if unit["header_status"] == "bound":
            values = len(list(v5.NUM.finditer(unit["text"])))
            counts["legacy_bound_with_verifiable_columns"] += bool(period_span(text, start, values))
        prev = unit
    return counts


def validated_cases(data, archive):
    for name, field in (("calibration-inputs.jsonl", "inputs_sha256"),
                        ("calibration-scoring-labels.jsonl", "labels_sha256")):
        if v5.digest(data / name) != archive["calibration"][field]:
            raise ValueError(f"hash mismatch in {name}")
    cases = v5.readrows(data / "calibration-inputs.jsonl")
    if len(cases) != 49 or len({c["case_id"] for c in cases}) != 49:
        raise ValueError("expected 49 unique frozen Calibration cases")
    for case in cases:
        cs = case["candidates"]
        if [c["candidate_rank"] for c in cs] != list(range(1, 6)) or len({x["ticker"] for x in cs}) != 1:
            raise ValueError("invalid S3 rank or ticker scope")
        for c in cs:
            v5._validate(c)
    return cases


def diagnose(data, output, device):
    archive = v5.readjson(HISTORY / "freeze.json")
    cases = validated_cases(data, archive)
    wanted = {"C2": None, "C3": archive["calibration_selection"]["best_c3_mmr_lambda"],
              "C4": archive["calibration_selection"]["best_c4_mmr_lambda"]}
    if {k: v5.digest(v5.MODEL/k) for k in archive["fingerprints"]["model_sha256"]} != archive["fingerprints"]["model_sha256"]:
        raise ValueError("pinned model file hash mismatch")
    # Perform all selection before opening the scoring labels.
    archived = {(r["case_id"], r["arm"], r["budget_bytes"]): r
                for r in v5.readrows(data / "calibration-deliveries.jsonl")
                if r["arm"] in wanted and r["mmr_lambda"] == wanted[r["arm"]]}
    enc = v5.BGEM3Encoder(v5.MODEL, device=device)
    records, headers, unit_bytes, unit_tokens = [], Counter(), [], []
    all_units = []
    for case in cases:
        question, cs = case["question"], case["candidates"]
        encoded, _ = v5._encode(question, case["retrieval_query"], cs, enc)
        all_units.append(encoded)
        unit_bytes.extend(len(u["text"].encode()) for u in encoded)
        unit_tokens.extend(enc.token_lengths([u["text"] for u in encoded]))
        headers.update({"chunks":len(cs), "chunks_with_linebreaks":sum("\n" in c["text"] for c in cs),
                        "v5_cues":sum(len(list(fb._HEADER_CUE.finditer(c["text"]))) for c in cs),
                        "period_cues":sum(len(list(PERIOD.finditer(c["text"]))) for c in cs),
                        "unit_cues":sum(len(list(SCALE.finditer(c["text"]))) for c in cs),
                        "row_units":sum(u["table_row"] for u in encoded),
                        "bound_row_units":sum(u["header_status"] == "bound" for u in encoded),
                        "uncertain_row_units":sum(u["header_status"] == "uncertain" for u in encoded)})
        for ci, c in enumerate(cs):
            headers.update(binding_decisions(c["text"], [u for u in encoded
                if u["candidate_index"] == ci and u["table_row"]]))
        runs = []
        for arm, lam in wanted.items():
            for budget in v5.BUDGETS:
                items, chosen, events = trace_semantic(question, cs, encoded, budget, arm, lam)
                old = archived[case["case_id"], arm, budget]
                if items != old["items"]:
                    raise ValueError(f"replay diverged for {v5.pseudo(case['case_id'])} {arm} {budget}")
                runs.append((arm, budget, old, events))
        records.append(runs)
        print(f"Replayed Calibration {len(records)}/{len(cases)}", flush=True)

    labels = {row["case_id"]:row for row in v5.readrows(data / "calibration-scoring-labels.jsonl")}
    public, totals, coverage = [], defaultdict(Counter), Counter()
    corrected_totals = defaultdict(Counter)
    budget_flags = defaultdict(Counter)
    selected_totals = defaultdict(Counter)
    for case, units, runs in zip(cases, all_units, records, strict=True):
        label = labels[case["case_id"]]
        for arm, budget, old, events in runs:
            rows = attribution(case, label, old, units, events)
            key = f"{arm}-{budget}"
            totals[key].update(row["first_loss"] for row in rows)
            for row in rows:
                if not row["nonfinancial_target"]:
                    corrected_totals[key][row["first_loss"]] += 1
                    corrected_totals[key]["rows"] += 1
                    corrected_totals[key]["source_match"] += row["source_span_delivery"]
                if row["first_loss"] == "F5":
                    budget_flags[key].update(row["selection_events"])
            selected_totals[key].update({
                "all":len(old["selected_units"]),
                "table_rows":sum(u["table_row"] for u in old["selected_units"]),
                "bound":sum(u["header_status"] == "bound" for u in old["selected_units"])})
            totals[key].update({"rows":len(rows), "v5_match":sum(x["v5_text_match"] for x in rows),
                                "source_match":sum(x["source_span_delivery"] for x in rows)})
            public.append({"case":v5.pseudo(case["case_id"]), "company":v5.pseudo(case["company"]),
                           "arm":arm, "budget":budget, "rows":rows,
                           "selected_units":len(old["selected_units"]),
                           "selected_table_units":sum(u["table_row"] for u in old["selected_units"]),
                           "selected_bound_units":sum(u["header_status"] == "bound" for u in old["selected_units"])})
        for page, row in v5._expected_rows(label):
            loc = row_candidates(case, page, row)
            coverage["expected"] += 1
            coverage["in_frozen_chunks"] += bool(loc)
            complete = any(u["candidate_index"] == i and u["start"] <= s and e <= u["end"]
                           for i, s, e in loc for u in units)
            coverage["whole_evidence_unit"] += complete
            invalid = nonfinancial_target(row)
            if invalid:
                coverage[f"excluded_{invalid}"] += 1
            else:
                coverage["corrected_expected"] += 1
                coverage["corrected_in_frozen_chunks"] += bool(loc)
                coverage["corrected_whole_evidence_unit"] += complete
    v5.writerows(output/"cases.jsonl", public)
    v5.writejson(output/"failure-attribution.json", {
        "first_loss_counts":totals, "corrected_first_loss_counts":corrected_totals,
        "unit_source_coverage":coverage,
        "evidence_unit_bytes":lengths(unit_bytes), "unit_embedding_tokens":lengths(unit_tokens),
        "units_over_embedding_limit":sum(t > enc.max_length for t in unit_tokens),
        "headers":headers, "budget_events_nonexclusive":budget_flags,
        "selected_units":selected_totals, "matched_archived_selections":len(public),
        "lambda_historical_replay":wanted, "model":enc.metadata,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    args = parser.parse_args()
    diagnose(args.data, args.out, args.device)


if __name__ == "__main__":
    main()
