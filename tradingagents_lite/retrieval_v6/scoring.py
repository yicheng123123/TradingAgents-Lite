"""Label-only scoring. The retrieval entry point never accepts this module's truth objects."""
from collections import Counter, defaultdict
from hashlib import sha256
import re
from statistics import fmean, median
import unicodedata

from ..rag_benchmark import relevance
from ..retrieval_v2_chunks import Audit, normalize, numeric_line
from ..retrieval_v4 import paired_comparison, query_hash, company_hash
from .features import page_key

TOKEN = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?%?|[^\W\d_]+|\.{2,}|[^\s]")
DATE = re.compile(
    r"(?:\(?)(?:(?:for\s+the\s+)?(?:(?:fiscal\s+)?years?|quarters?|months?)\s+ended\s+|as\s+of\s+|at\s+)?"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|sept?\.?|oct\.?|nov\.?|dec\.?)\s+"
    r"(?:[1-9]|[12]\d|3[01])(?:\s*,?\s*(?:19|20)\d{2}(?:\s*(?:,|and|&)\s*(?:19|20)\d{2})*)?[\s,.)]*", re.I)
SECTION = re.compile(r"(?:item\s+\d+[a-z]?\.?(?:\s+financial statements and supplementary data)?|form\s+10-[kq])[\s.]*", re.I)


def literal(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("−", "-")
    return " " + " ".join(part for part in TOKEN.findall(text) if not re.fullmatch(r"\.{2,}", part)) + " "


def truth_for(labels, rows, source):
    """Score PDF source lines using V3 and a stricter V5.1-style span rule.

    These denominators differ from V5.1 compression, which derived targets from excerpts.
    V3 strips punctuation. The stricter proxy preserves signs, parentheses and units.
    """
    audit = Audit(source, None)
    needed = {(item["doc_name"], item["evidence_page_num"]) for evidence in labels.values() for item in evidence}
    grouped = defaultdict(list)
    for key, row in rows.items():
        if page_key(row) in needed:
            grouped[page_key(row)].append((key, normalize(row["text"]), literal(row["text"])))
    truth = {}
    for qid, evidence in labels.items():
        anchors = defaultdict(list)
        for ev in evidence:
            anchors[ev["doc_name"], ev["evidence_page_num"]].append(ev["evidence_text"])
        versions = {"v3": [], "v51_style": []}
        for page, excerpts in anchors.items():
            old_anchors = [normalize(value) for value in excerpts]
            new_anchors = [literal(value) for value in excerpts]
            for line in audit.page(*page):
                if not numeric_line(line):
                    continue
                normalized = normalize(line)
                if not any(normalized in target for target in old_anchors):
                    continue
                versions["v3"].append({key for key, norm, _ in grouped[page] if normalized in norm})
                wording = literal(line)
                if not DATE.fullmatch(line.strip()) and not SECTION.fullmatch(line.strip()) and any(
                    wording in target for target in new_anchors
                ):
                    versions["v51_style"].append({key for key, _, text in grouped[page] if wording in text})
        truth[qid] = set(anchors), versions
    return truth


def metrics(q, arm, output, rows, truth):
    gold, lines = truth
    ranked, candidates = output["ranked"], set(output["pool"])
    pages = [page_key(rows[key]) for key in ranked]
    p5 = set(pages[:5])
    cp = {page_key(rows[key]) for key in candidates}
    pre = {page_key(rows[key]) for key in output["pre_cap"]}
    before = {}
    after = {}
    for position, key in enumerate(output["before"], 1):
        before.setdefault(page_key(rows[key]), position)
    for position, page in enumerate(pages, 1):
        after.setdefault(page, position)
    top = set(ranked[:5])
    diagnostics = {"A_top5": len(gold & p5), "B_in_pool_not_top5": len((gold & cp)-p5),
                   "C_not_in_pool": len(gold-cp)}
    reasons = Counter()
    top_maps = output.get("features") or {}
    for page in gold-p5:
        if page not in cp:
            reasons["candidate_missing"] += 1
        elif len(p5) < min(5, len(ranked)):
            reasons["in_pool_with_duplicate_top5"] += 1
        else:
            relevant = next((key for key in ranked if page_key(rows[key]) == page), None)
            if not relevant or relevant not in top_maps:
                reasons["in_pool_outranked"] += 1
                continue
            winning = [top_maps[key] for key in ranked[:5]]
            candidate = top_maps[relevant]
            flagged = [label for field, label in (("M","metric_proxy"),("Y","period_proxy"),("F","statement_proxy"))
                       if candidate[field] > 0 and sum(row[field] < candidate[field] for row in winning) >= 3]
            reasons[flagged[0] if flagged else "in_pool_outranked"] += 1
    return {
        "question_sha256": query_hash(q["id"]), "company_group": company_hash(q["company"]), "arm": arm,
        "page": relevance(pages, gold),
        "document": relevance(list(dict.fromkeys(doc for doc, _ in pages)), {doc for doc, _ in gold}),
        "candidate_oracle_at_5": min(5,len(gold & cp))/len(gold),
        "pre_cap_oracle_at_5": min(5,len(gold & pre))/len(gold),
        "candidate_page_recall": len(gold & cp)/len(gold),
        "candidate_count": len(candidates), "returned_count_at_5": len(top),
        "unique_pages_at_5": len(p5), "latency": output["latency"],
        "line_proxies": {version: {"identified": len(targets), "index_complete": sum(bool(keys) for keys in targets),
                                    "pool_complete": sum(bool(keys & candidates) for keys in targets),
                                    "top5_complete": sum(bool(keys & top) for keys in targets)}
                         for version, targets in lines.items()},
        "gold_page_ranks": [{"key": sha256(repr(p).encode()).hexdigest()[:24],
                             "before": before.get(p), "after": after.get(p)} for p in sorted(gold)],
        "loss_attribution": diagnostics, "failure_proxies": dict(reasons),
    }


def percentiles(values):
    values = sorted(values)
    def quantile(p):
        x = p*(len(values)-1)
        i = int(x)
        return values[i] + (x-i)*(values[min(i+1,len(values)-1)]-values[i])
    return {"mean": fmean(values), "p50": quantile(.5), "p95": quantile(.95)}


def summarize(rows):
    first = rows[0]
    lines = {v: {k: sum(r["line_proxies"][v][k] for r in rows) for k in stats}
             for v, stats in first["line_proxies"].items()}
    reasons = Counter()
    for row in rows:
        reasons.update(row["failure_proxies"])
    ranks = [p for row in rows for p in row["gold_page_ranks"] if p["after"] is not None]
    paired = [p for p in ranks if p["before"] is not None]
    return {"n": len(rows), "page": {k: fmean(r["page"][k] for r in rows) for k in first["page"]},
            "document": {k: fmean(r["document"][k] for r in rows) for k in first["document"]},
            **{k: fmean(r[k] for r in rows) for k in ("candidate_oracle_at_5", "pre_cap_oracle_at_5",
                         "candidate_page_recall", "candidate_count", "unique_pages_at_5")},
            "line_proxies": lines,
            "loss_attribution": {k: sum(r["loss_attribution"][k] for r in rows) for k in first["loss_attribution"]},
            "failure_proxies": dict(reasons),
            "latency": {kind: percentiles([r["latency"][kind] for r in rows]) for kind in first["latency"] if kind.endswith("_ms")},
            "peak_allocated_bytes": max((r["latency"].get("peak_allocated_bytes") or 0) for r in rows),
            "rank_migration": {"matched": len(paired), "median_before": median(p["before"] for p in paired) if paired else None,
                               "median_after": median(p["after"] for p in paired) if paired else None,
                               "promoted_6_10": sum(6 <= p["before"] <= 10 and p["after"]<=5 for p in paired),
                               "promoted_11_20": sum(11<= p["before"] <=20 and p["after"]<=5 for p in paired),
                               "promoted_21_plus": sum(p["before"]>=21 and p["after"]<=5 for p in paired)}}


def quality(rows):
    return (fmean(r["page"]["recall@5"] for r in rows), fmean(r["page"]["recall@1"] for r in rows),
            fmean(r["page"]["mrr@10"] for r in rows), sum(r["line_proxies"]["v51_style"]["top5_complete"] for r in rows),
            -fmean(r["candidate_count"] for r in rows))


def select(groups):
    choice = max(groups, key=lambda arm: quality(groups[arm]))
    companies = sorted({r["company_group"] for r in groups[choice]})
    test_predictions = []
    fits = {}
    for company in companies:
        train = {arm: [r for r in rows if r["company_group"] != company] for arm, rows in groups.items()}
        fitted = max(train, key=lambda arm: quality(train[arm]))
        fits[company] = fitted
        test_predictions.extend({**r, "oof_selected_arm": fitted} for r in groups[fitted] if r["company_group"] == company)
    return choice, test_predictions, fits
