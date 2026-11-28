"""Deliver whole fields under the exact JSON UTF-8 budget."""
import re
from time import perf_counter

from ..memory import _context
from ..rag import compact_json

BUDGETS = (900, 1800, 3600)
COMPRESSION = ("C0", "C1", "C2", "C3", "C4")
STOP = {"with", "this", "that", "from", "have", "were", "against", "before"}
THEMES = {
    "momentum": {"momentum", "signal", "directional", "threshold"},
    "risk": {"veto", "volatility", "risk", "exposure", "limit"},
    "settlement": {"outcome", "settled", "return", "relative", "benchmark"},
}


def terms(text):
    return set(re.findall(r"[a-z]{3,}", text.casefold())) - STOP


def optional_fields(episode, reflection):
    return {
        "process_lesson": reflection.notes.lesson,
        "applicability": reflection.notes.applicability,
        "potential_error": reflection.notes.potential_error,
        "decision_rationale": reflection.decision_thesis,
        "historical_outcome": episode.outcome.model_dump(mode="json"),
    }


def required_fields(episode, reflection):
    return {
        "memory_id": episode.memory_id,
        "historical_evidence_ids": list(reflection.source_evidence_ids),
        "situation": reflection.situation,
        "action": episode.action.value,
        "lesson": episode.reflection.lesson,
        "uncertainty": reflection.notes.uncertainty,
        "eligible_at": reflection.eligible_at.isoformat(),
        "decision_time": episode.decision_time.isoformat(),
    }


def size(items):
    return len(compact_json(items).encode("utf-8"))


def deliver(recalled, query, budget, mode):
    if budget < 2 or mode not in COMPRESSION:
        raise ValueError("Invalid memory context configuration")
    started = perf_counter()
    selected = [ep.memory_id for ep, _ in recalled]
    full = [_context(ep, compact=False) for ep, _ in recalled]
    if mode == "C0":
        items = full
    elif mode == "C1":
        items = [_context(ep, compact=True) for ep, _ in recalled]
    else:
        items = [required_fields(ep, ref) for ep, ref in recalled]
    context = []
    assignments = []

    def fits(values):
        return size(values) <= budget

    def attach(index, key, value):
        candidate = [{**item, key: value} if i == index else item for i, item in enumerate(context)]
        if fits(candidate):
            context[index] = candidate[index]

    for (episode, reflection), item in zip(recalled, items, strict=True):
        if not fits([*context, item]):
            continue
        context.append(item)
        if mode in {"C0", "C1"}:
            continue
        field_order = list(optional_fields(episode, reflection).items())
        if mode in {"C3", "C4"}:
            words = terms(query)
            query_themes = {name for name, aliases in THEMES.items() if aliases & words}

            def priority(field):
                position, (key, value) = field
                field_words = terms(key.replace("_", " ") + " " + compact_json(value))
                theme_count = sum(bool(THEMES[name] & field_words) for name in query_themes)
                return 2 * len(field_words & words) + theme_count + (key == "process_lesson"), -position

            field_order = [pair for _, pair in sorted(enumerate(field_order), key=priority, reverse=True)]
        index = len(context) - 1
        for field_rank, (key, value) in enumerate(field_order):
            if mode == "C4":
                assignments.append((field_rank, index, key, value))
            else:
                attach(index, key, value)
    for _, index, key, value in sorted(assignments):
        attach(index, key, value)
    return {
        "selected": selected, "injected": [item["memory_id"] for item in context], "context": context,
        "raw_bytes": size(full), "context_bytes": size(context),
        "latency_ms": (perf_counter() - started) * 1000,
    }


def fidelity(recalled, delivery):
    lookup = {ep.memory_id: (ep, ref) for ep, ref in recalled}
    items = delivery["context"]
    checks = {"lesson": 0, "references": 0, "action": 0, "eligible_date": 0,
              "decision_date_present": 0, "decision_date_exact": 0, "outcome_present": 0,
              "outcome_exact": 0, "uncertainty": 0, "structured_lesson": 0,
              "numeric_content_exact": 0}
    for item in items:
        ep, ref = lookup[item["memory_id"]]
        checks["lesson"] += item["lesson"] == ep.reflection.lesson
        checks["references"] += item["historical_evidence_ids"] == list(ref.source_evidence_ids)
        checks["action"] += item["action"] == ep.action.value
        checks["eligible_date"] += item["eligible_at"] == ref.eligible_at.isoformat()
        checks["decision_date_present"] += "decision_time" in item
        checks["decision_date_exact"] += item.get("decision_time") == ref.decision_time.isoformat()
        checks["outcome_present"] += "historical_outcome" in item
        checks["outcome_exact"] += item.get("historical_outcome") == ep.outcome.model_dump(mode="json")
        checks["uncertainty"] += item.get("uncertainty") == ref.notes.uncertainty
        checks["structured_lesson"] += item.get("process_lesson") == ref.notes.lesson
        source = {**required_fields(ep, ref), **optional_fields(ep, ref)}
        source["potential_error"] = ref.notes.potential_error if "situation" in item else ep.reflection.potential_error
        checks["numeric_content_exact"] += all(
            value == source[key] for key, value in item.items()
            if key in source and re.search(r"\d", compact_json(value)) and key != "memory_id"
        )
    return checks
