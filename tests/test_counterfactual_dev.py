from collections import Counter, defaultdict
from datetime import datetime, timedelta
from hashlib import sha256
import json
import secrets
import stat

import pytest

from tradingagents_lite import counterfactual_runner
from tradingagents_lite.counterfactual_dev import (
    CUTOFF, _eligible_documents, generate_development,
    generate_private_test, validate_private_test_dataset,
)


def test_development_packets_are_reproducible_and_case_local(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    generate_development(first)
    generate_development(second)
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["scope"] == "development"
    assert manifest["mother_cases"] == 5 and manifest["tasks"] == 10
    assert len(manifest["sha256"]) == 21
    for relative_path, digest in manifest["sha256"].items():
        assert sha256((first / relative_path).read_bytes()).hexdigest() == digest

    references = [json.loads(line) for line in (first / "scoring/reference.jsonl").read_text().splitlines()]
    assert Counter(row["category"] for row in references) == {
        "financial_reasoning": 4, "dynamic_planning": 4, "time_conflict_or_missing": 2,
    }
    mothers = defaultdict(list)
    seen_versions = set()
    seen_documents = set()
    for row in references:
        mothers[row["mother_id"]].append(row)
        task = row["task_id"]
        packet = json.loads((first / "agent-inputs" / f"{task}.json").read_text())
        original = json.loads((first / "authoring" / f"{task}.json").read_text())
        assert set(packet) == {"task_id", "company", "question", "cutoff", "documents"}
        assert packet["task_id"] == task
        versions = {doc["version_id"] for doc in original["documents"]}
        documents = {doc["document_id"] for doc in original["documents"]}
        assert not seen_versions & versions
        assert not seen_documents & documents
        seen_versions.update(versions)
        seen_documents.update(documents)
        assert all(set(doc) == {"document_id", "version_id", "title", "text", "eligible_at"}
                   for doc in packet["documents"])
        assert set(row["required_versions"]) <= {doc["version_id"] for doc in packet["documents"]}
        if "planning" in row:
            plan = row["planning"]
            assert plan["approved_reference"] not in packet["question"]
            by_version = {doc["version_id"]: doc for doc in packet["documents"]}
            for version in (plan["routing_version"], plan["target_version"]):
                assert plan["approved_reference"] in by_version[version]["text"]
        if row["category"] == "time_conflict_or_missing":
            assert len(packet["documents"]) == (5 if row["status"] == "answered" else 4)
            assert all(value not in doc["text"] for value in row["future_only_values"] for doc in packet["documents"])

    for variants in mothers.values():
        assert sorted(row["variant"] for row in variants) == [0, 1]
        assert variants[0]["value"] != variants[1]["value"]
    assert sum(row["status"] == "abstained" for row in references) == 1
    with pytest.raises(FileExistsError):
        generate_development(first)


def test_projection_accepts_cutoff_equality_and_ignores_backdated_future_revision():
    def version(name, when):
        return {"document_id": "synthetic-notice", "version_id": name,
                "eligible_at": when.isoformat(), "text": "Report period June 30"}

    old = version("old", CUTOFF - timedelta(seconds=1))
    at_cutoff = version("at-cutoff", CUTOFF)
    future = version("future", CUTOFF + timedelta(seconds=1))
    assert _eligible_documents([future, old, at_cutoff]) == [at_cutoff]
    assert _eligible_documents([future]) == []
    conflicting = version("conflicting", CUTOFF)
    with pytest.raises(ValueError, match="Ambiguous"):
        _eligible_documents([at_cutoff, conflicting])


def test_private_test_generation_is_private_randomized_and_profile_valid(tmp_path, monkeypatch):
    freeze_path = tmp_path / "phase2.3-freeze.json"
    monkeypatch.setattr(counterfactual_runner, "PRIVATE_FREEZE_PATH", freeze_path)
    counterfactual_runner.write_private_freeze(freeze_path)
    monkeypatch.setattr(secrets, "token_bytes", lambda size: b"\xA7" * size)

    first, second = tmp_path / "private-one", tmp_path / "private-two"
    first_report = generate_private_test(first, freeze_path)
    second_report = generate_private_test(second, freeze_path)
    first_manifest = json.loads((first / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())

    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert first_report == second_report
    assert first_report["scope"] == "private_test"
    assert first_report["mother_cases"] == 10 and first_report["tasks"] == 20
    assert first_report["category_counts"] == {
        "dynamic_planning": 6, "financial_reasoning": 6, "time_conflict_or_missing": 8,
    }
    assert first_report["time_answerability"] == {"answered": 4, "abstained": 4}
    assert first_report["total_answerability"] == {"answered": 16, "abstained": 4}
    assert set(first_report) == {
        "scope", "mother_cases", "tasks", "category_counts", "time_answerability",
        "total_answerability", "config_sha256", "data_fingerprint_sha256",
        "file_inventory_valid", "case_order_valid",
    }
    assert first_manifest["scope"] == "private_test" and first_manifest["synthetic"] is True
    seed_path = first / "authoring" / ".private-seed"
    assert seed_path.stat().st_size == 32
    assert stat.S_IMODE(seed_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(first.stat().st_mode) == 0o700

    dev_root = tmp_path / "development"
    generate_development(dev_root)
    dev_packets = [json.loads(path.read_text()) for path in (dev_root / "agent-inputs").glob("*.json")]
    dev_companies = {packet["company"] for packet in dev_packets}
    dev_task_ids = {packet["task_id"] for packet in dev_packets}
    dev_document_ids = {doc["document_id"] for packet in dev_packets for doc in packet["documents"]}
    private_packets = [json.loads(path.read_text()) for path in (first / "agent-inputs").glob("*.json")]
    private_refs = [json.loads(line) for line in (first / "scoring/reference.jsonl").read_text().splitlines()]
    assert not dev_companies & {packet["company"] for packet in private_packets}
    assert not dev_task_ids & {packet["task_id"] for packet in private_packets}
    assert not dev_document_ids & {doc["document_id"] for packet in private_packets for doc in packet["documents"]}
    assert all(packet["task_id"].startswith("private-task-") for packet in private_packets)
    assert all(doc["document_id"].startswith("private-doc-")
               and doc["version_id"].startswith("private-ver-")
               for packet in private_packets for doc in packet["documents"])
    assert len({packet["task_id"] for packet in private_packets}) == 20
    assert len({doc["version_id"] for packet in private_packets for doc in packet["documents"]}) == sum(
        len(packet["documents"]) for packet in private_packets)

    by_task = {row["task_id"]: row for row in private_refs}
    mothers = defaultdict(list)
    route_codes = set()
    for reference in private_refs:
        mothers[reference["mother_id"]].append(reference)
        packet = json.loads((first / "agent-inputs" / f"{reference['task_id']}.json").read_text())
        cutoff = datetime.fromisoformat(packet["cutoff"])
        assert all(datetime.fromisoformat(doc["eligible_at"]) <= cutoff
                   for doc in packet["documents"])
        if reference["category"] == "dynamic_planning":
            plan = reference["planning"]
            assert plan["approved_reference"].startswith("private-REF-")
            assert plan["approved_reference"] not in packet["question"]
            route_codes.add(plan["approved_reference"])
            documents = {doc["version_id"]: doc for doc in packet["documents"]}
            assert plan["approved_reference"] in documents[plan["routing_version"]]["text"]
            assert plan["approved_reference"] in documents[plan["target_version"]]["text"]
        if reference["category"] == "time_conflict_or_missing":
            for value in reference["future_only_values"]:
                assert all(value not in doc["text"] for doc in packet["documents"])
    assert len(route_codes) == 6
    assert len(mothers) == 10
    for pair in mothers.values():
        pair = sorted(pair, key=lambda row: row["variant"])
        assert [row["variant"] for row in pair] == [0, 1]
        assert pair[0]["value"] != pair[1]["value"]
        if pair[0]["category"] == "time_conflict_or_missing":
            assert [row["status"] for row in pair] == ["answered", "abstained"]
    assert set(by_task) == {packet["task_id"] for packet in private_packets}
    assert validate_private_test_dataset(first, freeze_path)["file_inventory_valid"] is True

    with pytest.raises(FileExistsError):
        generate_private_test(first, freeze_path)


def test_private_test_manifest_detects_data_tampering(tmp_path, monkeypatch):
    freeze_path = tmp_path / "phase2.3-freeze.json"
    monkeypatch.setattr(counterfactual_runner, "PRIVATE_FREEZE_PATH", freeze_path)
    counterfactual_runner.write_private_freeze(freeze_path)
    monkeypatch.setattr(secrets, "token_bytes", lambda size: b"\x39" * size)
    data_root = tmp_path / "private"
    generate_private_test(data_root, freeze_path)
    target = next((data_root / "agent-inputs").glob("*.json"))
    packet = json.loads(target.read_text())
    packet["question"] += " tampered"
    target.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="private_test_data_hash_mismatch"):
        validate_private_test_dataset(data_root, freeze_path)
