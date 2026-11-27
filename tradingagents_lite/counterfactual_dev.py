"""Generate the ten synthetic Phase 2 development fixtures, offline."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
import os
from pathlib import Path
from random import Random
import secrets


DEV_SEED = 910237
CUTOFF = datetime(2024, 6, 30, 16, tzinfo=timezone.utc)
BEFORE = CUTOFF - timedelta(days=2)
AFTER = CUTOFF + timedelta(days=1)
PRIVATE_GENERATOR_VERSION = "counterfactual-private-test-v1"
PRIVATE_FREEZE_ID = "counterfactual-phase2.3-private-v1"
PRIVATE_TEST_PROFILE = {
    "schema_version": PRIVATE_GENERATOR_VERSION,
    "mother_cases": 10,
    "tasks": 20,
    "categories": {
        "financial_reasoning": {"mother_cases": 3, "tasks": 6},
        "dynamic_planning": {"mother_cases": 3, "tasks": 6},
        "time_conflict_or_missing": {"mother_cases": 4, "tasks": 8,
                                     "answered": 4, "abstained": 4},
    },
    "total_answered": 16,
    "total_abstained": 4,
    "variants_per_mother": 2,
    "arms_per_task": ["CF0", "CF1"],
}
PRIVATE_SYLLABLES = (
    "qev", "zarn", "vori", "kel", "drax", "nemi", "ulv", "tavi", "prax", "sori",
    "wex", "moru", "jex", "riva", "brel", "xani", "lorv", "cavi", "yure", "fex",
)
RECIPES = (
    ("Vesnaro Relay Works", "free_cash_flow"),
    ("Quilvaro Alloy Systems", "cash_conversion"),
    ("Tervixi Process Labs", "funding_route"),
    ("Brevanto Signal Foundry", "contract_route"),
    ("Nalvexa Gauge Works", "as_of"),
)


def _id(rng, prefix, namespace=None):
    identifier = f"{prefix}-{rng.getrandbits(80):020x}"
    return f"{namespace}-{identifier}" if namespace else identifier


def _doc(rng, title, text, eligible_at=BEFORE, document_id=None, namespace=None):
    return {
        "document_id": document_id or _id(rng, "doc", namespace),
        "version_id": _id(rng, "ver", namespace),
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


def _financial(rng, company, family, variant, namespace=None, cutoff=CUTOFF):
    unit = "USD millions"
    available_at = cutoff - timedelta(days=2)
    cash = _amount(rng, 6200 + 1100 * variant, 7100 + 1100 * variant)
    if family == "free_cash_flow":
        capital = _amount(rng, 2000 + 9300 * variant, 3000 + 9300 * variant)
        reimbursements = _amount(rng, 120, 750)
        definition = "free cash flow, defined as operating cash flow minus gross capital purchases plus capital reimbursements"
        docs = [
            _doc(rng, "Operating cash statement", f"H1 operating cash flow: {cash:.2f} USD millions.",
                 eligible_at=available_at, namespace=namespace),
            _doc(rng, "Capital movement note", f"H1 gross capital purchases: {capital:.2f} USD millions. "
                 f"H1 capital reimbursements: {reimbursements:.2f} USD millions.",
                 eligible_at=available_at, namespace=namespace),
        ]
        result = cash - capital + reimbursements
    else:
        profit = _amount(rng, 12000, 13900)
        receipts = _amount(rng, 400, 950)
        unit = "percent"
        definition = "adjusted cash conversion, defined as 100 times (operating cash flow minus restructuring cash receipts) divided by operating profit"
        docs = [
            _doc(rng, "Cash and receipts note", f"H1 operating cash flow: {cash:.2f} USD millions. "
                 f"H1 restructuring cash receipts: {receipts:.2f} USD millions.",
                 eligible_at=available_at, namespace=namespace),
            _doc(rng, "Operating results", f"H1 operating profit: {profit:.2f} USD millions.",
                 eligible_at=available_at, namespace=namespace),
        ]
        result = 100 * (cash - receipts) / profit
    question = f"What was {company}'s H1 {definition}? Answer in {unit} to two decimal places."
    return question, docs, _reference(result, unit, docs)


def _routing(rng, company, family, schedules, active, namespace=None, cutoff=CUTOFF):
    references = [_id(rng, "REF", namespace) for _ in schedules]
    if family == "funding_route":
        notice, target = "H1 funding routing notice", "treasury schedule"
        left, right, metric = "gross funding requirement", "dedicated funds", "net funding requirement"
    else:
        notice, target = "H1 contract selection notice", "contract adjustment memo"
        left, right, metric = "gross contract settlement", "customer credits", "net contract settlement"
    available_at = cutoff - timedelta(days=2)
    route = _doc(rng, notice, f"Approved H1 {target}: {references[active]}. Unlisted references are scenario drafts.",
                 eligible_at=available_at, namespace=namespace)
    docs = [_doc(rng, target, f"Reference {code}. {left.capitalize()}: {gross:.2f} USD millions. "
                 f"{right.capitalize()}: {deduction:.2f} USD millions.", eligible_at=available_at,
                 namespace=namespace)
            for code, (gross, deduction) in zip(references, schedules, strict=True)]
    question = (f"Use {company}'s {notice} to identify the approved {target}. "
                f"What is the H1 {metric}, calculated as {left} less {right}? "
                "Answer in USD millions to two decimal places.")
    gross, deduction = schedules[active]
    answer = _reference(gross - deduction, "USD millions", [route, docs[active]])
    answer["planning"] = {"routing_version": route["version_id"], "approved_reference": references[active],
                          "target_version": docs[active]["version_id"]}
    return question, [route, *docs], answer


def _as_of(rng, company, amounts, variant, namespace=None):
    original, revision = amounts
    doc_id = _id(rng, "doc", namespace)
    first = _doc(rng, "Authorized credit facility notice",
                 f"Authorized undrawn H1 credit limit: {original:.2f} USD millions.",
                 CUTOFF if variant == 0 else AFTER, doc_id, namespace)
    later = _doc(rng, "Authorized credit facility notice",
                 f"Revision of June 30 H1 notice. Authorized undrawn H1 credit limit: {revision:.2f} USD millions.",
                 AFTER + timedelta(days=1), doc_id, namespace)
    question = (f"As of {CUTOFF.isoformat()}, what was {company}'s authorized undrawn H1 credit limit? "
                "Use only evidence available by that cutoff. Answer in USD millions to two decimal places, "
                "or abstain if no eligible authorization supports a value.")
    answer = (_reference(original, "USD millions", [first]) if variant == 0
              else _reference(None, "USD millions", []))
    answer["future_only_values"] = [f"{value:.2f}" for value in (amounts[1:] if variant == 0 else amounts)]
    return question, [first, later], answer


def _distractors(rng, family, namespace=None, eligible_at=BEFORE):
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
    return [_doc(rng, title, text, eligible_at=eligible_at, namespace=namespace) for title, text in notes]


def _eligible_documents(documents, cutoff=CUTOFF):
    versions = {}
    availability_slots = set()
    for document in documents:
        when = datetime.fromisoformat(document["eligible_at"])
        if when > cutoff:
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


PRIVATE_CUTOFF = datetime(2025, 6, 30, 16, tzinfo=timezone.utc)


def _private_company(rng, index):
    return (f"{rng.choice(PRIVATE_SYLLABLES).capitalize()}{rng.choice(PRIVATE_SYLLABLES)}"
            f" Synthetic Systems {index + 1:02d}")


def _private_temporal_case(rng, company, mother, variant, scenario):
    cutoff = PRIVATE_CUTOFF
    namespace = "private"
    amount = _amount(rng, 1800, 9700)
    future_amount = amount + _amount(rng, 350, 2400)
    document_id = _id(rng, "doc", namespace)
    question = (f"As of {cutoff.isoformat()}, what was {company}'s approved undrawn H1 credit limit? "
                "Use only evidence available by the cutoff. Answer in USD millions to two decimal places, "
                "or abstain if no eligible authorization supports a value.")
    docs = []
    if variant == 0:
        available_at = (cutoff if scenario == 0 else
                        cutoff - timedelta(days=15 if scenario == 1 else 1 if scenario == 2 else 45))
        approved = _doc(rng, "Approved facility notice",
                        f"Authorized undrawn H1 credit limit: {amount:.2f} USD millions.",
                        eligible_at=available_at, document_id=document_id, namespace=namespace)
        revised = _doc(rng, "Revised facility notice",
                       f"Revised authorized undrawn H1 credit limit: {future_amount:.2f} USD millions.",
                       eligible_at=cutoff + timedelta(days=1), document_id=document_id, namespace=namespace)
        docs.extend((approved, revised))
        reference = _reference(amount, "USD millions", [approved])
        reference["future_only_values"] = [f"{future_amount:.2f}"]
    elif scenario < 3:
        first = _doc(rng, "Authorization notice",
                     f"Authorized undrawn H1 credit limit: {amount:.2f} USD millions.",
                     eligible_at=cutoff + timedelta(seconds=1 if scenario == 0 else 86400),
                     document_id=document_id, namespace=namespace)
        revision = _doc(rng, "Later authorization revision",
                        f"Revised authorized undrawn H1 credit limit: {future_amount:.2f} USD millions.",
                        eligible_at=cutoff + timedelta(days=3), document_id=document_id, namespace=namespace)
        docs.extend((first, revision))
        reference = _reference(None, "USD millions", [])
        reference["future_only_values"] = [f"{amount:.2f}", f"{future_amount:.2f}"]
    else:
        draft = _doc(rng, "Unsigned facility draft",
                     f"Draft undrawn H1 credit limit: {amount:.2f} USD millions. No approval is recorded.",
                     eligible_at=cutoff - timedelta(days=5), document_id=document_id, namespace=namespace)
        archive_note = _doc(rng, "Facility archive index",
                            "No signed authorization is present in the H1 facility archive as of the cutoff.",
                            eligible_at=cutoff - timedelta(days=2), namespace=namespace)
        docs.extend((draft, archive_note))
        reference = _reference(None, "USD millions", [])
        reference["future_only_values"] = []
    docs.extend(_distractors(rng, "as_of", namespace=namespace,
                             eligible_at=cutoff - timedelta(days=2)))
    packet = {"task_id": _id(rng, "task", namespace), "company": company,
              "question": question, "cutoff": cutoff.isoformat(), "documents": docs}
    answer = {"task_id": packet["task_id"], "mother_id": mother, "variant": variant,
              "category": "time_conflict_or_missing", **reference}
    return packet, answer


def _private_pairs(rng):
    namespace = "private"
    packets, answers, case_order = [], [], []
    company_index = 0

    def add_pair(company, mother, category, build):
        pair_packets, pair_answers = [], []
        for variant in (0, 1):
            packet, reference = build(variant)
            documents = packet["documents"]
            rng.shuffle(documents)
            pair_packets.append(packet)
            pair_answers.append(reference)
        if pair_answers[0]["value"] == pair_answers[1]["value"]:
            raise ValueError("private_counterfactual_pair_not_distinguishable")
        if category == "time_conflict_or_missing":
            if [row["status"] for row in pair_answers] != ["answered", "abstained"]:
                raise ValueError("private_time_pair_answerability_mismatch")
        if any(packet["company"] != company for packet in pair_packets):
            raise ValueError("private_pair_company_mismatch")
        packets.extend(pair_packets)
        answers.extend(pair_answers)
        case_order.append({"case_key": mother, "task_ids": [packet["task_id"] for packet in pair_packets]})

    for case_index, family in enumerate(("free_cash_flow", "cash_conversion", "free_cash_flow")):
        company = _private_company(rng, company_index)
        company_index += 1
        mother = _id(rng, "mother", namespace)
        cutoff = PRIVATE_CUTOFF
        if family == "free_cash_flow":
            base_cash = _amount(rng, 6500, 9200)
            base_capital = _amount(rng, 1900, 5300)
            base_reimbursements = _amount(rng, 170, 830)
            def build_financial(variant):
                cash = base_cash
                capital = base_capital + Decimal(variant) * Decimal("1250.00")
                reimbursements = base_reimbursements
                question, docs, reference = _financial(rng, company, family, variant,
                                                       namespace=namespace, cutoff=cutoff)
                # Preserve the existing two-step task format while varying one paired input.
                docs[0]["text"] = f"H1 operating cash flow: {cash:.2f} USD millions."
                docs[1]["text"] = (f"H1 gross capital purchases: {capital:.2f} USD millions. "
                                    f"H1 capital reimbursements: {reimbursements:.2f} USD millions.")
                value = cash - capital + reimbursements
                reference.update(_reference(value, "USD millions", docs))
                packet = {"task_id": _id(rng, "task", namespace), "company": company,
                          "question": question, "cutoff": cutoff.isoformat(), "documents": docs}
                reference["task_id"] = packet["task_id"]
                reference.update(mother_id=mother, variant=variant, category="financial_reasoning")
                return packet, reference
            build = build_financial
        else:
            profit = _amount(rng, 10400, 18600)
            cash = _amount(rng, 6500, 9200)
            receipts = _amount(rng, 250, 1100)
            delta = Decimal("137.00")
            def build_cash_conversion(variant):
                question, docs, reference = _financial(rng, company, family, variant,
                                                       namespace=namespace, cutoff=cutoff)
                adjusted_receipts = receipts + Decimal(variant) * delta
                docs[0]["text"] = (f"H1 operating cash flow: {cash:.2f} USD millions. "
                                    f"H1 restructuring cash receipts: {adjusted_receipts:.2f} USD millions.")
                docs[1]["text"] = f"H1 operating profit: {profit:.2f} USD millions."
                value = Decimal(100) * (cash - adjusted_receipts) / profit
                reference.update(_reference(value, "percent", docs))
                packet = {"task_id": _id(rng, "task", namespace), "company": company,
                          "question": question, "cutoff": cutoff.isoformat(), "documents": docs}
                reference["task_id"] = packet["task_id"]
                reference.update(mother_id=mother, variant=variant, category="financial_reasoning")
                return packet, reference
            build = build_cash_conversion
        add_pair(company, mother, "financial_reasoning", build)

    for case_index, family in enumerate(("funding_route", "contract_route", "funding_route")):
        company = _private_company(rng, company_index)
        company_index += 1
        mother = _id(rng, "mother", namespace)
        schedules = [(_amount(rng, 13000, 18000), _amount(rng, 1200, 3400)),
                     (_amount(rng, 26000, 31500), _amount(rng, 1200, 3400))]
        def build_routing(variant):
            question, docs, reference = _routing(rng, company, family, schedules, variant,
                                                 namespace=namespace, cutoff=PRIVATE_CUTOFF)
            docs.extend(_distractors(rng, family, namespace=namespace,
                                     eligible_at=PRIVATE_CUTOFF - timedelta(days=2)))
            packet = {"task_id": _id(rng, "task", namespace), "company": company,
                      "question": question, "cutoff": PRIVATE_CUTOFF.isoformat(), "documents": docs}
            reference["task_id"] = packet["task_id"]
            reference.update(mother_id=mother, variant=variant, category="dynamic_planning")
            return packet, reference
        add_pair(company, mother, "dynamic_planning", build_routing)

    for scenario in range(4):
        company = _private_company(rng, company_index)
        company_index += 1
        mother = _id(rng, "mother", namespace)
        add_pair(company, mother, "time_conflict_or_missing",
                 lambda variant, company=company, mother=mother, scenario=scenario:
                 _private_temporal_case(rng, company, mother, variant, scenario))

    return packets, answers, case_order


def _canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _private_write(path, value=None, *, raw=None, lines=False):
    data = raw if raw is not None else (
        b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode() for row in value)
        if lines else (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def validate_private_test_dataset(data_root, freeze_path=None):
    """Verify private dataset inventory and frozen config without returning content."""
    from . import counterfactual_runner as runner

    data_root = Path(data_root).resolve()
    freeze_path = Path(freeze_path or runner.PRIVATE_FREEZE_PATH).resolve()
    freeze = runner._verify_private_freeze(freeze_path)
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_profile = PRIVATE_TEST_PROFILE
    if (manifest.get("schema_version") != PRIVATE_GENERATOR_VERSION
            or manifest.get("scope") != "private_test" or manifest.get("synthetic") is not True
            or manifest.get("mother_cases") != expected_profile["mother_cases"]
            or manifest.get("tasks") != expected_profile["tasks"]
            or manifest.get("category_counts") != {
                key: value["tasks"] for key, value in expected_profile["categories"].items()}
            or manifest.get("time_answerability") != {"answered": 4, "abstained": 4}
            or manifest.get("total_answerability") != {"answered": 16, "abstained": 4}
            or manifest.get("private_profile_sha256") != _canonical_hash(expected_profile)):
        raise ValueError("private_test_manifest_profile_mismatch")
    if (manifest.get("freeze_id") != freeze["freeze_id"]
            or manifest.get("config_sha256") != freeze["config_sha256"]
            or manifest.get("freeze_file_sha256") != sha256(freeze_path.read_bytes()).hexdigest()):
        raise ValueError("private_test_config_freeze_mismatch")
    file_hashes = manifest.get("sha256")
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError("private_test_file_inventory_missing")
    actual_paths = set()
    for path in data_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("private_test_symlink_rejected")
        if path.is_file() and path != manifest_path:
            actual_paths.add(path.relative_to(data_root).as_posix())
    if actual_paths != set(file_hashes):
        raise ValueError("private_test_file_inventory_mismatch")
    for relative, digest in file_hashes.items():
        candidate = (data_root / relative).resolve()
        if not candidate.is_relative_to(data_root) or not candidate.is_file():
            raise ValueError("private_test_file_path_invalid")
        if sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ValueError("private_test_data_hash_mismatch")
    if manifest.get("data_fingerprint_sha256") != _canonical_hash(file_hashes):
        raise ValueError("private_test_data_fingerprint_mismatch")
    agent_paths = sorted((data_root / "agent-inputs").glob("*.json"))
    if len(agent_paths) != expected_profile["tasks"]:
        raise ValueError("private_test_task_count_mismatch")
    tasks_by_id = {}
    document_owner, seen_versions = {}, set()
    for path in agent_paths:
        packet = json.loads(path.read_text(encoding="utf-8"))
        if set(packet) != {"task_id", "company", "question", "cutoff", "documents"}:
            raise ValueError("private_test_agent_projection_schema_mismatch")
        if packet["task_id"] != path.stem or not packet["task_id"].startswith("private-task-"):
            raise ValueError("private_test_task_identity_mismatch")
        cutoff = datetime.fromisoformat(packet["cutoff"])
        if cutoff.utcoffset() is None:
            raise ValueError("private_test_cutoff_timezone_missing")
        local_document_ids = set()
        for doc in packet["documents"]:
            if set(doc) != {"document_id", "version_id", "title", "text", "eligible_at"}:
                raise ValueError("private_test_document_projection_schema_mismatch")
            eligible_at = datetime.fromisoformat(doc["eligible_at"])
            if eligible_at.utcoffset() is None or eligible_at > cutoff:
                raise ValueError("private_test_future_document_in_projection")
            if doc["version_id"] in seen_versions:
                raise ValueError("private_test_version_id_reused")
            seen_versions.add(doc["version_id"])
            local_document_ids.add(doc["document_id"])
        for document_id in local_document_ids:
            owner = document_owner.setdefault(document_id, packet["task_id"])
            if owner != packet["task_id"]:
                raise ValueError("private_test_document_id_reused_across_tasks")
        tasks_by_id[packet["task_id"]] = packet
    case_order = manifest.get("case_order")
    if not isinstance(case_order, list) or len(case_order) != expected_profile["mother_cases"]:
        raise ValueError("private_test_case_order_mismatch")
    seen_tasks, seen_cases, companies = set(), set(), set()
    for case in case_order:
        case_key, task_ids = case.get("case_key"), case.get("task_ids")
        if (not isinstance(case_key, str) or case_key in seen_cases or not isinstance(task_ids, list)
                or len(task_ids) != 2 or len(set(task_ids)) != 2):
            raise ValueError("private_test_case_pair_mismatch")
        if any(task_id not in tasks_by_id or task_id in seen_tasks for task_id in task_ids):
            raise ValueError("private_test_case_task_order_mismatch")
        pair_companies = {tasks_by_id[task_id]["company"] for task_id in task_ids}
        if len(pair_companies) != 1 or pair_companies & companies:
            raise ValueError("private_test_company_reuse")
        companies.update(pair_companies)
        seen_cases.add(case_key)
        seen_tasks.update(task_ids)
    if seen_tasks != set(tasks_by_id):
        raise ValueError("private_test_case_order_incomplete")
    seed_path = data_root / "authoring" / ".private-seed"
    if not seed_path.is_file() or seed_path.stat().st_size != 32:
        raise ValueError("private_test_os_seed_missing")
    return {
        "scope": "private_test", "mother_cases": expected_profile["mother_cases"],
        "tasks": expected_profile["tasks"], "category_counts": manifest["category_counts"],
        "time_answerability": manifest["time_answerability"],
        "total_answerability": manifest["total_answerability"],
        "config_sha256": manifest["config_sha256"],
        "data_fingerprint_sha256": manifest["data_fingerprint_sha256"],
        "file_inventory_valid": True, "case_order_valid": True,
    }


def generate_private_test(output, freeze_path=None):
    """Generate the formal private set from OS entropy and print only aggregate-safe metadata."""
    from . import counterfactual_runner as runner

    freeze_path = Path(freeze_path or runner.PRIVATE_FREEZE_PATH).resolve()
    freeze = runner._verify_private_freeze(freeze_path)
    output = Path(output).resolve()
    seed = secrets.token_bytes(32)
    if len(seed) != 32:
        raise ValueError("os_random_seed_length_invalid")
    rng = Random(int.from_bytes(seed, "big"))
    packets, answers, case_order = _private_pairs(rng)
    category_counts = Counter(answer["category"] for answer in answers)
    time_answers = [answer for answer in answers if answer["category"] == "time_conflict_or_missing"]
    if (len(packets) != 20 or len(answers) != 20 or len(case_order) != 10
            or dict(category_counts) != {key: value["tasks"] for key, value in PRIVATE_TEST_PROFILE["categories"].items()}
            or Counter(answer["status"] for answer in time_answers) != {"answered": 4, "abstained": 4}
            or Counter(answer["status"] for answer in answers) != {"answered": 16, "abstained": 4}):
        raise ValueError("private_test_generation_acceptance_failed")
    mothers = defaultdict(list)
    for answer in answers:
        mothers[answer["mother_id"]].append(answer)
    for pair in mothers.values():
        if len(pair) != 2 or sorted(answer["variant"] for answer in pair) != [0, 1]:
            raise ValueError("private_test_pair_generation_failed")
        if pair[0]["category"] != pair[1]["category"]:
            raise ValueError("private_test_pair_category_mismatch")
        if pair[0]["category"] != "time_conflict_or_missing" and pair[0]["value"] == pair[1]["value"]:
            raise ValueError("private_test_pair_gold_not_distinguishable")
        if pair[0]["category"] == "time_conflict_or_missing" and [a["status"] for a in sorted(pair, key=lambda a: a["variant"])] != ["answered", "abstained"]:
            raise ValueError("private_test_time_pair_mismatch")
    planning_codes = [answer["planning"]["approved_reference"] for answer in answers if "planning" in answer]
    if len(planning_codes) != 6 or len(set(planning_codes)) != 6:
        raise ValueError("private_test_planning_codes_not_unique")

    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    for directory in ("authoring", "agent-inputs", "scoring"):
        (output / directory).mkdir(mode=0o700)
    _private_write(output / "authoring" / ".private-seed", raw=seed)
    for packet in packets:
        filename = packet["task_id"] + ".json"
        _private_write(output / "authoring" / filename, packet)
        cutoff = datetime.fromisoformat(packet["cutoff"])
        projection = {**packet, "documents": _eligible_documents(packet["documents"], cutoff)}
        _private_write(output / "agent-inputs" / filename, projection)
    _private_write(output / "scoring" / "reference.jsonl", answers, lines=True)
    file_hashes = {path.relative_to(output).as_posix(): sha256(path.read_bytes()).hexdigest()
                   for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {
        "schema_version": PRIVATE_GENERATOR_VERSION, "scope": "private_test", "synthetic": True,
        "freeze_id": freeze["freeze_id"], "freeze_file_sha256": sha256(freeze_path.read_bytes()).hexdigest(),
        "config_sha256": freeze["config_sha256"],
        "private_profile_sha256": _canonical_hash(PRIVATE_TEST_PROFILE),
        "mother_cases": 10, "tasks": 20,
        "category_counts": {key: category_counts[key] for key in sorted(category_counts)},
        "time_answerability": {"answered": 4, "abstained": 4},
        "total_answerability": {"answered": 16, "abstained": 4},
        "case_order": case_order, "sha256": file_hashes,
        "data_fingerprint_sha256": _canonical_hash(file_hashes),
    }
    _private_write(output / "manifest.json", manifest)
    return validate_private_test_dataset(output, freeze_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("development", "private-test"), default="development")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.scope == "development":
        if args.freeze:
            parser.error("--freeze applies only to --scope private-test")
        generate_development(args.out)
        print("Generated 10 synthetic Development tasks from five mother cases.")
    else:
        result = generate_private_test(args.out, args.freeze)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
