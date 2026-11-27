from collections import Counter, defaultdict
from datetime import timedelta
from hashlib import sha256
import json

import pytest

from tradingagents_lite.counterfactual_dev import CUTOFF, _eligible_documents, generate_development


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
