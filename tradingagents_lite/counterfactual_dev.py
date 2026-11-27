"""Generate the ten synthetic Phase 2 development fixtures, offline."""

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
from pathlib import Path
from random import Random


DEV_SEED = 910237
CUTOFF = datetime(2024, 6, 30, 16, tzinfo=timezone.utc)
BEFORE = CUTOFF - timedelta(days=2)
AFTER = CUTOFF + timedelta(days=1)
RECIPES = (
    ("Vesnaro Relay Works", "free_cash_flow"),
    ("Quilvaro Alloy Systems", "cash_conversion"),
    ("Tervixi Process Labs", "funding_route"),
    ("Brevanto Signal Foundry", "contract_route"),
    ("Nalvexa Gauge Works", "as_of"),
)


def _id(rng, prefix):
    return f"{prefix}-{rng.getrandbits(80):020x}"


def _doc(rng, title, text, eligible_at=BEFORE, document_id=None):
    return {
        "document_id": document_id or _id(rng, "doc"),
        "version_id": _id(rng, "ver"),
        "title": title,
        "text": text,
        "eligible_at": eligible_at.isoformat(),
    }


def _amount(rng, lower, upper):
    return Decimal(rng.randint(lower, upper)) / 100


def _reference(result, unit, documents):
    return {
        "status": "answered" if result is not None else "abstained",
        "value": str(result.quantize(Decimal(".01"), rounding=ROUND_HALF_UP)) if result is not None else None,
        "unit": unit,
        "required_versions": [doc["version_id"] for doc in documents],
    }


def _financial(rng, company, family, variant):
    unit = "USD millions"
    cash = _amount(rng, 6200 + 1100 * variant, 7100 + 1100 * variant)
    if family == "free_cash_flow":
        capital = _amount(rng, 2000 + 9300 * variant, 3000 + 9300 * variant)
        reimbursements = _amount(rng, 120, 750)
        definition = "free cash flow, defined as operating cash flow minus gross capital purchases plus capital reimbursements"
        docs = [
            _doc(rng, "Operating cash statement", f"H1 operating cash flow: {cash:.2f} USD millions."),
            _doc(rng, "Capital movement note", f"H1 gross capital purchases: {capital:.2f} USD millions. "
                 f"H1 capital reimbursements: {reimbursements:.2f} USD millions."),
        ]
        result = cash - capital + reimbursements
    else:
        profit = _amount(rng, 12000, 13900)
        receipts = _amount(rng, 400, 950)
        unit = "percent"
        definition = "adjusted cash conversion, defined as 100 times (operating cash flow minus restructuring cash receipts) divided by operating profit"
        docs = [
            _doc(rng, "Cash and receipts note", f"H1 operating cash flow: {cash:.2f} USD millions. "
                 f"H1 restructuring cash receipts: {receipts:.2f} USD millions."),
            _doc(rng, "Operating results", f"H1 operating profit: {profit:.2f} USD millions."),
        ]
        result = 100 * (cash - receipts) / profit
    question = f"What was {company}'s H1 {definition}? Answer in {unit} to two decimal places."
    return question, docs, _reference(result, unit, docs)


def _routing(rng, company, family, schedules, active):
    references = [_id(rng, "REF") for _ in schedules]
    if family == "funding_route":
        notice, target = "H1 funding routing notice", "treasury schedule"
        left, right, metric = "gross funding requirement", "dedicated funds", "net funding requirement"
    else:
        notice, target = "H1 contract selection notice", "contract adjustment memo"
        left, right, metric = "gross contract settlement", "customer credits", "net contract settlement"
    route = _doc(rng, notice, f"Approved H1 {target}: {references[active]}. Unlisted references are scenario drafts.")
    docs = [_doc(rng, target, f"Reference {code}. {left.capitalize()}: {gross:.2f} USD millions. "
                 f"{right.capitalize()}: {deduction:.2f} USD millions.")
            for code, (gross, deduction) in zip(references, schedules, strict=True)]
    question = (f"Use {company}'s {notice} to identify the approved {target}. "
                f"What is the H1 {metric}, calculated as {left} less {right}? "
                "Answer in USD millions to two decimal places.")
    gross, deduction = schedules[active]
    answer = _reference(gross - deduction, "USD millions", [route, docs[active]])
    answer["planning"] = {"routing_version": route["version_id"], "approved_reference": references[active],
                          "target_version": docs[active]["version_id"]}
    return question, [route, *docs], answer


def _as_of(rng, company, amounts, variant):
    original, revision = amounts
    doc_id = _id(rng, "doc")
    first = _doc(rng, "Authorized credit facility notice",
                 f"Authorized undrawn H1 credit limit: {original:.2f} USD millions.",
                 CUTOFF if variant == 0 else AFTER, doc_id)
    later = _doc(rng, "Authorized credit facility notice",
                 f"Revision of June 30 H1 notice. Authorized undrawn H1 credit limit: {revision:.2f} USD millions.",
                 AFTER + timedelta(days=1), doc_id)
    question = (f"As of {CUTOFF.isoformat()}, what was {company}'s authorized undrawn H1 credit limit? "
                "Use only evidence available by that cutoff. Answer in USD millions to two decimal places, "
                "or abstain if no eligible authorization supports a value.")
    answer = (_reference(original, "USD millions", [first]) if variant == 0
              else _reference(None, "USD millions", []))
    answer["future_only_values"] = [f"{value:.2f}" for value in (amounts[1:] if variant == 0 else amounts)]
    return question, [first, later], answer


def _distractors(rng, family):
    if family == "funding_route":
        notes = (
            ("H1 funding assessment", "A separate routing notice identifies the approved H1 treasury schedule."),
            ("H1 funding minutes", "Gross funding and dedicated funds were discussed. No schedule was approved at this meeting."),
            ("Treasury procedures", "H1 net funding uses the approved schedule for the reporting period."),
            ("Funding archive", "An archive label alone does not establish a treasury schedule as approved for H1."),
        )
    elif family == "contract_route":
        notes = (
            ("Contract review notes", "A separate selection notice identifies the approved H1 adjustment memo."),
            ("Settlement committee minutes", "Gross contract settlement and credits were discussed. No memo was approved."),
            ("Credits policy", "Customer credits are deducted from the approved gross contract settlement."),
            ("Contract memo archive", "An archive label alone does not establish a memo as approved for H1."),
        )
    else:
        notes = (
            ("Reporting calendar", "H1 covers the six-month internal reporting period."),
            ("Operating update", "Production and staffing were reviewed. No authorized facility terms were issued."),
            ("Policy minutes", "Budget drafts have no approval status and no amounts are certified here."),
            ("Measurement note", "USD millions is the presentation unit for the internal finance pack."),
        )
    return [_doc(rng, title, text) for title, text in notes]


def _eligible_documents(documents):
    versions = {}
    availability_slots = set()
    for document in documents:
        when = datetime.fromisoformat(document["eligible_at"])
        if when > CUTOFF:
            continue
        document_id = document["document_id"]
        slot = document_id, when
        if slot in availability_slots:
            raise ValueError("Ambiguous versions at equal availability time")
        availability_slots.add(slot)
        previous = versions.get(document_id)
        if not previous or when > previous[0]:
            versions[document_id] = (when, document)
    return [document for document in documents
            if versions.get(document["document_id"], (None, None))[1] is document]


def _write(path, value, *, lines=False):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        if lines:
            for row in value:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        else:
            stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def generate_development(output):
    rng = Random(DEV_SEED)
    packets, answers = [], []
    categories = {"funding_route": "dynamic_planning", "contract_route": "dynamic_planning", "as_of": "time_conflict_or_missing"}
    for company, family in RECIPES:
        mother = _id(rng, "mother")
        if family in {"funding_route", "contract_route"}:
            schedules = [(_amount(rng, 10000 + branch * 20000, 12000 + branch * 20000),
                          _amount(rng, 1300, 3800)) for branch in range(2)]
        elif family == "as_of":
            amounts = (_amount(rng, 3000, 5900), _amount(rng, 7000, 9900))
        for variant in range(2):
            task_id = _id(rng, "cfdev")
            if family in {"funding_route", "contract_route"}:
                question, documents, answer = _routing(rng, company, family, schedules, variant)
            elif family == "as_of":
                question, documents, answer = _as_of(rng, company, amounts, variant)
            else:
                question, documents, answer = _financial(rng, company, family, variant)
            documents += _distractors(rng, family)
            rng.shuffle(documents)
            packets.append({"task_id": task_id, "company": company, "question": question,
                            "cutoff": CUTOFF.isoformat(), "documents": documents})
            answers.append({"task_id": task_id, "mother_id": mother, "variant": variant,
                            "category": categories.get(family, "financial_reasoning"), **answer})

    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for directory in ("authoring", "agent-inputs", "scoring"):
        (output / directory).mkdir()
    for packet in packets:
        filename = packet["task_id"] + ".json"
        _write(output / "authoring" / filename, packet)
        projection = {**packet, "documents": _eligible_documents(packet["documents"])}
        _write(output / "agent-inputs" / filename, projection)
    _write(output / "scoring" / "reference.jsonl", answers, lines=True)
    _write(output / "manifest.json", {
        "scope": "development", "synthetic": True, "seed": DEV_SEED,
        "tasks": len(packets), "mother_cases": len(RECIPES),
        "sha256": {path.relative_to(output).as_posix(): sha256(path.read_bytes()).hexdigest()
                   for path in sorted(output.rglob("*")) if path.is_file()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate_development(args.out)
    print("Generated 10 synthetic Development tasks from five mother cases.")


if __name__ == "__main__":
    main()
