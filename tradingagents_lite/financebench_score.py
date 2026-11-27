"""Deterministic FinanceBench scoring in a process separate from inference."""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import re
from statistics import mean

from pydantic import Field

from . import b1
from .data import sha256
from .data_v2_market import write_json
from .financebench_data import Case, GoldCase, rows, write_rows
from .schema import Contract


SCALE = {"": 0, "k": 3, "thousand": 3, "m": 6, "million": 6, "b": 9, "billion": 9,
         "t": 12, "trillion": 12}
CURRENCY = {"$": "usd", "€": "eur", "£": "gbp"}
NUMBER = re.compile(
    r"(?P<sign>[+-]?)\s*(?P<currency>USD|EUR|GBP|CAD|\$|€|£)?\s*"
    r"(?P<value>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)\s*"
    r"(?P<scale>thousands?|millions?|billions?|trillions?|[kmbt])?\s*"
    r"(?P<unit>USD|EUR|GBP|CAD|%|percent(?:age points?)?|bps|basis points?|x|times|shares)?",
    re.IGNORECASE,
)
CONTEXT = re.compile(r"\b(USD|EUR|GBP|CAD)\s+(thousands?|millions?|billions?|trillions?)\b", re.I)


class Answer(Contract):
    status: str = Field(pattern=r"^(answered|abstained)$")
    answer: str = Field(max_length=1000)
    evidence_ids: tuple[str, ...] = Field(max_length=4)


def numeric(text, question):
    value = text.strip().replace("−", "-").removesuffix(".")
    negated = value.startswith("(") and value.endswith(")")
    if negated:
        value = value[1:-1].strip()
    match = NUMBER.fullmatch(value)
    if match is None or (negated and match["sign"]):
        return None
    scale = (match["scale"] or "").lower().rstrip("s")
    prefix, unit = (match[k].lower() if match[k] else "" for k in ("currency", "unit"))
    prefix = CURRENCY.get(prefix, prefix)
    hints = {(currency.lower(), magnitude.lower().rstrip("s")) for currency, magnitude in CONTEXT.findall(question)}
    hinted_currency, hinted_scale = next(iter(hints)) if len(hints) == 1 else ("", "")
    if prefix and unit and prefix != unit:
        return None
    dimension = prefix or unit or hinted_currency
    if dimension in {"usd", "eur", "gbp", "cad"} and not scale:
        scale = hinted_scale if dimension == hinted_currency else ""
    if not dimension:
        if re.search(r"(?:%|\bpercent(?:age)?\b)", question, re.I):
            dimension = "%"
        elif re.search(r"\bin (?:basis points|bps)\b", question, re.I):
            dimension = "bps"
    factor = Decimal(10) ** SCALE[scale]
    if dimension in {"bps", "basis point", "basis points"}:
        factor /= 100
        dimension = "%"
    if dimension.startswith("percent"):
        dimension = "%"
    if dimension == "times":
        dimension = "x"
    amount = Decimal(match["value"].replace(",", "")) * factor
    if negated or match["sign"] == "-":
        amount = -amount
    # Floor is 0.01 in the displayed currency magnitude, or 0.01 percentage point.
    floor = Decimal(".01") * max(Decimal(1), factor)
    return amount, dimension, floor


def _text(value):
    return " ".join(value.casefold().split()).rstrip(".!?")


def assess(prediction, gold, question):
    expected = numeric(gold, question)
    if prediction["status"] != "ok" or prediction.get("answer_status") != "answered":
        return "incorrect", "execution_failure_or_abstention", bool(expected)
    answer = prediction["answer"]
    if expected:
        observed = numeric(answer, question)
        if observed is None or observed[1] != expected[1]:
            return "review", "numeric_format_or_unit_ambiguous", True
        tolerance = max(expected[2], abs(expected[0]) * Decimal(".005"))
        return ("correct" if abs(observed[0] - expected[0]) <= tolerance else "incorrect",
                "normalized_decimal_tolerance", True)
    a, b = _text(answer), _text(gold)
    if a == b:
        return "correct", "conservative_exact_match", False
    if a in {"yes", "no"} and b in {"yes", "no"}:
        return "incorrect", "exact_binary_conflict", False
    return "review", "text_unresolved", False


class Review(Contract):
    id: str
    arm: str
    verdict: str = Field(pattern=r"^(correct|incorrect)$")
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(min_length=1, max_length=1000)


def _accuracy(group):
    scores = Counter(row["verdict"] for row in group)
    n, hits, pending = len(group), scores["correct"], scores["review"]
    return {"n": n, "correct": hits, "incorrect": scores["incorrect"], "pending_review": pending,
            "adjudicated_accuracy": hits / (n - pending) if n > pending else None,
            "full_denominator_bounds": [hits / n, (hits + pending) / n] if n else None}


def score_run(run, bundle, gold_path, output, reviews=None):
    run_meta = json.loads((run / "manifest.json").read_text())
    prediction_path = run / "predictions.jsonl"
    if run_meta["status"] != "complete" or sha256(prediction_path) != run_meta["predictions_sha256"]:
        raise ValueError("All planned predictions must be finalized and hash-verified before scoring")
    source_manifest = bundle / "manifest.json"
    source = json.loads(source_manifest.read_text())
    if (sha256(source_manifest) != run_meta["input_manifest_sha256"]
            or sha256(bundle / "questions.jsonl") != source["questions_sha256"]
            or sha256(gold_path) != source["gold_sha256"]):
        raise ValueError("Scoring inputs differ from the frozen benchmark")
    cases = {row["id"]: Case.model_validate(row) for row in rows(bundle / "questions.jsonl")}
    key = {row["id"]: GoldCase.model_validate(row) for row in rows(gold_path)}
    annotations = {}
    for raw in rows(reviews) if reviews else []:
        review = Review.model_validate(raw)
        identity = (review.id, review.arm)
        if identity in annotations:
            raise ValueError("Duplicate human review")
        annotations[identity] = review
    results, seen = [], set()
    for row in rows(prediction_path):
        case = cases[row["id"]]
        reference = key[case.id]
        identity = (case.id, row["arm"])
        if identity in seen:
            raise ValueError("Duplicate prediction")
        seen.add(identity)
        verdict, method, is_numeric = assess(row, reference.answer, case.question)
        review = annotations.pop(identity, None)
        if review:
            if verdict != "review":
                raise ValueError("Human reviews apply only to unresolved rules")
            verdict = review.verdict
        visible = {(item["document_id"], item["page"]) for item in row["exposed"]}
        matches = len(visible & set(reference.anchors))
        results.append({"id": case.id, "arm": row["arm"], "status": row["status"],
                        "verdict": verdict, "method": method, "numerical": is_numeric,
                        "human_review": review.model_dump(mode="json") if review else None,
                        "evidence_hits": matches, "evidence_total": len(reference.anchors),
                        "budget_tokens": row["budget_tokens"], "tool_calls": row["tool_calls"],
                        "latency_seconds": row["latency_seconds"],
                        "citations": len(row["citations"]), "valid_citations": len(row["valid_citations"])})
    if annotations or seen != {(cell["id"], cell["arm"]) for cell in run_meta["plan"]}:
        raise ValueError("Scoring requires exactly one record per planned cell and reviews for known cells only")
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "scores.jsonl", results)
    write_rows(output / "review-needed.jsonl", [row for row in results if row["verdict"] == "review"])
    groups = {}
    for arm in sorted({row["arm"] for row in results}):
        group = [row for row in results if row["arm"] == arm]
        refs = sum(row["citations"] for row in group)
        recall = [row["evidence_hits"] / row["evidence_total"] for row in group if row["evidence_total"]]
        latency = sorted(row["latency_seconds"] for row in group)
        groups[arm] = {
            "answer_accuracy": _accuracy(group), "numerical_accuracy": _accuracy([r for r in group if r["numerical"]]),
            "completion_rate": sum(row["status"] == "ok" for row in group) / len(group),
            "failure_counts": dict(Counter(row["status"] for row in group if row["status"] != "ok")),
            "page_proxy_evidence_recall": mean(recall) if recall else None, "evidence_denominator": len(recall),
            "citation_validity": sum(row["valid_citations"] for row in group) / refs if refs else None,
            "citation_denominator": refs, "total_tokens": sum(row["budget_tokens"] for row in group),
            "mean_tokens": mean(row["budget_tokens"] for row in group),
            "mean_tool_calls": mean(row["tool_calls"] for row in group),
            "mean_latency_seconds": mean(latency), "p95_latency_seconds": latency[max(0, (95 * len(latency) + 99) // 100 - 1)],
        }
    report = {"backend": run_meta["backend"], "scope": run_meta["scope"], "run_manifest_sha256": sha256(run / "manifest.json"),
              "prediction_sha256": sha256(prediction_path), "gold_sha256": sha256(gold_path),
              "input_manifest_sha256": run_meta["input_manifest_sha256"], "question_sha256": source["questions_sha256"],
              "configuration_sha256": run_meta["config_sha256"], "git_sha": run_meta["git_sha"],
              "actual_models": run_meta["actual_models"], "arms": groups,
              "scoring_code_sha256": sha256(Path(__file__)), "reviews_sha256": sha256(reviews) if reviews else None}
    write_json(output / "summary.json", report)
    return report
