import json
import platform
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tradingagents_lite import counterfactual_dev
from tradingagents_lite.counterfactual_dev import generate_development, generate_private_test
from tradingagents_lite import counterfactual_runner
from tradingagents_lite.counterfactual_runner import _hash, _sealed, _verify_sealed, run
from tradingagents_lite.counterfactual_score import _correct, parse_answer, score_run
from tradingagents_lite.data import sha256 as file_sha
from tradingagents_lite.provider import ModelCallError


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data/counterfactual-development-v1-001"


@pytest.mark.parametrize(("answer", "unit", "expected"), [
    ("12.34 percent", "percent", "12.34"),
    ("12.34%", "percent", "12.34"),
    ("$12.34 million", "USD millions", "12.34"),
    ("70.89 USD millions", "USD millions", "70.89"),
    ("70.89 million USD", "USD millions", "70.89"),
    ("USD 70.89 million", "USD millions", "70.89"),
    ("1.25 billion USD", "USD millions", "1250.00"),
    ("-2.50 USD millions", "USD millions", "-2.50"),
    ("70.895 USD millions", "USD millions", "70.90"),
    ("$1.25 billion", "USD millions", "1250.00"),
    ("$12,340,000", "USD millions", "12.34"),
    ("($2,500,000)", "USD millions", "-2.50"),
])
def test_counterfactual_deterministic_numeric_units(answer, unit, expected):
    assert str(parse_answer(answer, unit)) == expected


def test_counterfactual_numeric_rejects_ambiguous_or_wrong_dimension():
    assert parse_answer("12.34 or 13.00", "USD millions") is None
    assert parse_answer("70.89 million USD (100.86 - 29.97)", "USD millions") is None
    assert parse_answer("70.89 million EUR", "USD millions") is None
    assert parse_answer("about 70.89 million USD after credits", "USD millions") is None
    assert parse_answer("USD 70.89 USD million", "USD millions") is None
    assert parse_answer("$12.34 million", "percent") is None
    assert parse_answer("NaN", "USD millions") is None


def test_counterfactual_answer_and_abstention_scoring_are_deterministic():
    answer_ref = {"status": "answered", "value": "12.34", "unit": "USD millions"}
    answered = {"status": "ok", "answer_status": "answered", "answer": "$12,340,000"}
    assert _correct(answered, answer_ref)
    abstain_ref = {"status": "abstained", "value": None, "unit": "USD millions"}
    abstained = {"status": "ok", "answer_status": "abstained", "answer": ""}
    assert _correct(abstained, abstain_ref)
    assert not _correct({**answered, "status": "invalid_output"}, answer_ref)


def test_checkpoint_hash_detects_mutation():
    sealed = _sealed({"task_id": "CF0:synthetic", "status": "ok"})
    assert _verify_sealed(sealed)["status"] == "ok"
    with pytest.raises(ValueError, match="integrity"):
        _verify_sealed({**sealed, "status": "changed"})


def test_v2_config_freezes_new_protocol_and_shares_existing_experiment_budget():
    config = counterfactual_runner._config()

    assert config["protocol"] == "counterfactual-phase2-v2"
    assert counterfactual_runner.EXPERIMENT == "counterfactual-phase2-v1"
    assert config["experiment_token_limit"] == 300_000


def test_phase23_config_has_new_full_fingerprint_and_same_v2_core_strategy():
    config = counterfactual_runner._config()

    assert config["core_strategy_sha256"] == "8b57e1a1a0ec27da2825c45e5345b1ef27cc024225628b0bbfd86523d5d4ead1"
    assert config["protocol"] == "counterfactual-phase2-v2"
    assert config["model"] == "glm-5.3-flash"
    assert config["temperature"] == 0 and config["reasoning_effort"] == "low"
    assert config["thinking"] == {"type": "enabled"}
    assert config["max_output_tokens"] == 512 and config["max_prompt_bytes"] == 8000
    assert config["max_tool_rounds"] == 2 and config["max_tool_calls_per_round"] == 1
    assert config["rag"] == {"mode": "R3", "candidate_k": 24, "top_k": 4, "context_bytes": 3200}
    assert config["runtime"]["uv_lock_sha256"] and config["local_embedding_hashes"]
    assert config["private_test"]["acceptance"]["os_random_seed_bytes"] == 32
    assert config["private_test"]["acceptance"]["no_llm_generation"] is True


def _make_private_fixture(tmp_path, monkeypatch, seed=b"\xC3" * 32):
    freeze_path = tmp_path / "phase2.3-freeze.json"
    monkeypatch.setattr(counterfactual_runner, "PRIVATE_FREEZE_PATH", freeze_path)
    counterfactual_runner.write_private_freeze(freeze_path)
    monkeypatch.setattr(counterfactual_dev.secrets, "token_bytes", lambda size: seed)
    data_root = tmp_path / "private-data"
    generate_private_test(data_root, freeze_path)
    return data_root


def test_private_runner_requires_confirmation_and_rejects_subsets(tmp_path, monkeypatch):
    data_root = _make_private_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(counterfactual_runner, "_provider_for_cell",
                        lambda *_: pytest.fail("confirmation must be checked before provider initialization"))

    with pytest.raises(ValueError, match="private_test_requires_explicit_confirmation"):
        run(data_root, tmp_path / "unconfirmed", backend="llm")
    with pytest.raises(ValueError, match="private_test_does_not_allow_task_subsets"):
        run(data_root, tmp_path / "subset", backend="mock", limit=1, confirm_private_test=True)


def test_private_runner_rejects_configuration_freeze_mismatch_before_execution(tmp_path, monkeypatch):
    data_root = _make_private_fixture(tmp_path, monkeypatch, seed=b"\xC4" * 32)
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(counterfactual_runner, "_run_isolated",
                        lambda *_args, **_kwargs: pytest.fail("must reject before starting a worker"))

    with pytest.raises(ValueError, match="private_test_config_freeze_mismatch"):
        run(data_root, tmp_path / "bad-freeze", backend="mock", confirm_private_test=True)


def test_private_runner_orders_full_mother_cases_and_resumes_without_duplicates(tmp_path, monkeypatch):
    data_root = _make_private_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(counterfactual_runner, "_make_index", lambda *args, **kwargs: None)
    execution_order = []
    interrupt_after = {3}

    def fake_isolated(repo, dataset, stage, task_id, forbidden, arm, backend, broker_fd):
        if len(execution_order) + 1 in interrupt_after:
            raise KeyboardInterrupt
        execution_order.append((task_id, arm))
        result = {"id": task_id, "task_id": f"{arm}:{task_id}", "arm": arm,
                  "status": "ok", "answer": "", "answer_status": "abstained",
                  "citations": [], "valid_citations": [], "exposed": [], "retrieval_trace": [],
                  "tool_calls": 0, "latency_seconds": 0.01, "llm_calls": [], "budget_tokens": 0}
        path = stage / "output" / "result.json"
        path.write_text(json.dumps(result))
        return path

    monkeypatch.setattr(counterfactual_runner, "_run_isolated", fake_isolated)
    output = tmp_path / "private-run"
    with pytest.raises(KeyboardInterrupt):
        run(data_root, output, backend="mock", confirm_private_test=True)
    assert json.loads((output / "manifest.json").read_text())["recorded_cells"] == 2

    interrupt_after.clear()
    before_resume = list(execution_order)
    result = run(data_root, output, backend="mock", resume=True, confirm_private_test=True)
    assert result["status"] == "complete" and result["recorded_cells"] == 40
    assert len(execution_order) == 40
    assert before_resume == execution_order[:2]
    assert execution_order == [
        (task_id, arm)
        for case in json.loads((data_root / "manifest.json").read_text())["case_order"]
        for task_id in case["task_ids"] for arm in ("CF0", "CF1")
    ]
    plan = result["plan"]
    assert len(plan) == 40
    for start in range(0, 40, 4):
        group = plan[start:start + 4]
        assert [(cell["id"], cell["arm"]) for cell in group] == [
            (group[0]["id"], "CF0"), (group[0]["id"], "CF1"),
            (group[2]["id"], "CF0"), (group[2]["id"], "CF1"),
        ]

    scored = score_run(data_root, output, output / "scores.json")
    assert scored["scope"] == "private_test"
    assert scored["summary"]["CF0"]["answer_accuracy"]["total"] == 20
    assert scored["summary"]["CF1"]["answer_accuracy"]["total"] == 20


def test_private_runner_stops_at_mother_boundary_when_budget_preflight_fails(tmp_path, monkeypatch):
    data_root = _make_private_fixture(tmp_path, monkeypatch, seed=b"\xD4" * 32)
    monkeypatch.setattr(counterfactual_runner, "_make_index", lambda *args, **kwargs: None)
    calls, providers = [], []

    class SharedBudget:
        capacity_calls = 0

        def capacity(self, *, experiment_id, phase):
            self.capacity_calls += 1
            return {"available": 100_000 if self.capacity_calls == 1 else 0}

        def snapshot(self, *, experiment_id, phase):
            return {"experiment_id": experiment_id, "phase": phase, "charged_tokens": 0}

    budget = SharedBudget()

    class Provider:
        def __init__(self):
            self.client = SimpleNamespace(close=lambda: None)
            self.call_log = []

    def provider_for_cell(task_id):
        provider = Provider()
        providers.append((task_id, provider))
        return provider, budget, "b" * 64

    def fake_isolated(repo, dataset, stage, task_id, forbidden, arm, backend, broker_fd):
        calls.append((task_id, arm))
        path = stage / "output" / "result.json"
        path.write_text(json.dumps({"id": task_id, "task_id": f"{arm}:{task_id}", "arm": arm,
                                    "status": "ok", "answer": "", "answer_status": "abstained",
                                    "citations": [], "valid_citations": [], "exposed": [],
                                    "retrieval_trace": [], "tool_calls": 0, "latency_seconds": 0,
                                    "llm_calls": [], "budget_tokens": 0}))
        return path

    monkeypatch.setattr(counterfactual_runner, "_provider_for_cell", provider_for_cell)
    monkeypatch.setattr(counterfactual_runner, "_run_isolated", fake_isolated)
    result = run(data_root, tmp_path / "budget-stop", backend="llm", confirm_private_test=True)

    assert result["status"] == "partial"
    assert result["stop_reason"] == "mother_case_budget_preflight"
    assert result["recorded_cells"] == 4 and len(calls) == 4
    assert result["budget_preflight"][0]["required_worst_case_tokens"] == 73_216
    assert result["budget_preflight"][1]["available_tokens"] == 0
    assert all(not provider.call_log for _, provider in providers)
    with pytest.raises(ValueError, match="frozen stop condition"):
        run(data_root, tmp_path / "budget-stop", backend="llm", resume=True,
            confirm_private_test=True)


@pytest.mark.skipif(platform.system() != "Linux" or shutil.which("unshare") is None,
                    reason="requires Linux user/mount/network namespaces")
def test_private_worker_cannot_read_gold_authoring_or_seed(tmp_path, monkeypatch):
    data_root = _make_private_fixture(tmp_path, monkeypatch, seed=b"\xE5" * 32)
    manifest = json.loads((data_root / "manifest.json").read_text())
    packet_path = next((data_root / "agent-inputs").glob("*.json"))
    packet = json.loads(packet_path.read_text())
    stage = tmp_path / "worker-stage"
    (stage / "output").mkdir(parents=True)
    shutil.copyfile(packet_path, stage / "input.json")
    counterfactual_runner._make_index(packet, stage / "index", stage / "models")
    forbidden = [data_root / "manifest.json", *(data_root / path for path in manifest["sha256"])]
    result_path = counterfactual_runner._run_isolated(
        ROOT, data_root, stage, packet["task_id"], forbidden, "CF0", "mock", None)
    record = json.loads(result_path.read_text())
    probes = record["sandbox"]["host_paths"]

    assert record["status"] == "ok"
    assert probes[str((data_root / "scoring/reference.jsonl").resolve())] == "denied"
    assert probes[str((data_root / "authoring" / f"{packet['task_id']}.json").resolve())] == "denied"
    assert probes[str((data_root / "authoring" / ".private-seed").resolve())] == "denied"
    assert record["sandbox"]["api_key_in_environment"] is False
    assert record["sandbox"]["network_isolated"] is True


def test_frozen_scorer_reports_pairs_planning_abstention_and_future_acceptance(tmp_path):
    data_root = tmp_path / "data"
    generate_development(data_root)
    references = [json.loads(line) for line in (data_root / "scoring/reference.jsonl").read_text().splitlines()]
    references_by_id = {reference["task_id"]: reference for reference in references}
    plan = []
    records = []
    time_future = None
    for reference in references:
        packet = json.loads((data_root / "agent-inputs" / f"{reference['task_id']}.json").read_text())
        docs = {doc["version_id"]: doc for doc in packet["documents"]}
        evidence = ["cf:" + version for version in reference["required_versions"]]
        exposed = [{"evidence_id": "cf:" + doc["version_id"], "document_id": doc["document_id"],
                    "page": index} for index, doc in enumerate(packet["documents"], 1)]
        for arm in ("CF0", "CF1"):
            plan.append({"id": reference["task_id"], "arm": arm,
                         "input_sha256": file_sha(data_root / "agent-inputs" / f"{reference['task_id']}.json")})
            trace = []
            planning = reference.get("planning")
            if planning:
                trace.append({"query": packet["question"], "delivered_evidence": [
                    {"evidence_id": "cf:" + planning["routing_version"]}]})
                trace.append({"query": "approved " + planning["approved_reference"],
                              "delivered_evidence": [{"evidence_id": "cf:" + planning["target_version"]}]})
            future = []
            if reference["category"] == "time_conflict_or_missing":
                authoring = json.loads((data_root / "authoring" / f"{reference['task_id']}.json").read_text())
                cutoff = authoring["cutoff"]
                future = [doc for doc in authoring["documents"] if doc["eligible_at"] > cutoff]
            row = {"id": reference["task_id"], "task_id": f"{arm}:{reference['task_id']}", "arm": arm,
                   "status": "ok", "answer_status": reference["status"],
                   "answer": reference["value"] or "", "citations": evidence,
                   "valid_citations": evidence, "exposed": exposed, "retrieval_trace": trace,
                   "tool_calls": 2, "llm_calls": [], "budget_tokens": 0, "latency_seconds": 0,
                   "input_sha256": file_sha(data_root / "agent-inputs" / f"{reference['task_id']}.json")}
            if future and time_future is None and reference["status"] == "answered":
                time_future = (row, future[0], reference["future_only_values"][0])
            records.append(_sealed(row))

    run_root = tmp_path / "run"
    (run_root / "cells").mkdir(parents=True)
    predictions = run_root / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    manifest = {"backend": "mock", "status": "complete", "protocol": "counterfactual-phase2-v1",
                "plan": plan, "predictions_sha256": file_sha(predictions),
                "dataset_manifest_sha256": file_sha(data_root / "manifest.json")}
    (run_root / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "scores.json"
    scored = score_run(data_root, run_root, output)
    assert scored["summary"]["CF0"]["answer_accuracy"] == {"correct": 10, "total": 10}
    assert scored["summary"]["CF1"]["counterfactual_pair_accuracy"]["correct_mothers"] == 5
    assert scored["summary"]["CF1"]["adaptive_planning_success_rate"] == {"success": 4, "total": 4}
    assert scored["summary"]["CF0"]["abstention_accuracy"] == {"correct": 2, "total": 2}
    assert scored["summary"]["CF0"]["future_evidence_acceptance_rate"]["observed_acceptances"] == 0

    bad, future_doc, future_value = time_future
    changed = dict(bad)
    changed.pop("record_sha256", None)
    changed["answer"] = future_value
    changed["citations"] = ["cf:" + future_doc["version_id"]]
    changed = _sealed(changed)
    changed_key = changed["task_id"]
    records = [changed if row["task_id"] == changed_key else row for row in records]
    predictions.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    manifest["predictions_sha256"] = file_sha(predictions)
    (run_root / "manifest.json").write_text(json.dumps(manifest))
    detected = score_run(data_root, run_root, output)
    assert detected["summary"]["CF0"]["future_evidence_acceptance_rate"]["observed_acceptances"] == 1

    # A route reference guessed before the routing observation is not adaptive planning.
    planning_row = next(row for row in records
                        if row["arm"] == "CF1" and references_by_id[row["id"]].get("planning"))
    planning_ref = references_by_id[planning_row["id"]]["planning"]
    planning_row.pop("record_sha256", None)
    planning_row["retrieval_trace"][0]["query"] += " " + planning_ref["approved_reference"]
    planning_row.update(_sealed(planning_row))
    predictions.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    manifest["predictions_sha256"] = file_sha(predictions)
    (run_root / "manifest.json").write_text(json.dumps(manifest))
    preguessed = score_run(data_root, run_root, output)
    assert preguessed["summary"]["CF1"]["adaptive_planning_success_rate"] == {"success": 3, "total": 4}


def _tool_call(call_id, query, *, name="rag_search", limit=4):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps({"query": query, "limit": limit})}}


def _execute_cf1_with_replies(monkeypatch, replies):
    class Context:
        def __init__(self):
            self.trace, self.exposed = [], {}

    class Search:
        def __init__(self, case, retriever, partition, context):
            self.case, self.trace, self.exposed = case, context.trace, context.exposed

        def search(self, query, limit, *, round_index, tool_call_id, **kwargs):
            evidence_id = f"cf:version-{round_index}"
            item = {"evidence_id": evidence_id, "document_id": f"doc-{round_index}",
                    "page": round_index, "excerpt": f"Synthetic observation {round_index}.",
                    "chunk_id": f"chunk-{round_index}"}
            self.exposed[evidence_id] = {key: item[key] for key in ("evidence_id", "document_id", "page")}
            delivered = [{key: item[key] for key in ("evidence_id", "document_id", "page", "excerpt")}]
            self.trace.append({"query": query, "requested_limit": limit,
                               "delivered_evidence": delivered})
            return {"status": "available", "reason": None, "items": [item]}

    class Backend:
        def __init__(self):
            self.requests, self.call_log, self.observations = [], [], []

        def _call(self, messages, tools, **kwargs):
            self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools, **kwargs})
            reply = replies[len(self.requests) - 1]
            if isinstance(reply, Exception):
                raise reply
            return reply

    context = Context()
    backend = Backend()
    monkeypatch.setattr(counterfactual_runner.financebench, "CaseSearch",
                        lambda *args: context)
    monkeypatch.setattr(counterfactual_runner, "_Search", Search)
    packet = {"task_id": "offline-state-machine", "company": "Synthetic Co",
              "question": "Find the synthetic value using the supplied evidence.",
              "cutoff": "2025-01-01T00:00:00+00:00", "documents": []}
    record = counterfactual_runner._execute_case(packet, "CF1", None, "offline", backend)
    return record, backend


def _assert_tool_message_pairs(messages):
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        assert index + 1 < len(messages)
        tool_message = messages[index + 1]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_call_id"] == message["tool_calls"][0]["id"]


@pytest.mark.parametrize(("search_rounds", "stop_first", "expected_requests"), [
    (0, True, 2),
    (1, False, 3),
    (2, False, 3),
])
def test_cf1_normal_stops_use_final_json_mode_once(monkeypatch, search_rounds, stop_first,
                                                   expected_requests):
    replies = []
    if stop_first:
        replies.append({"content": "natural language intermediate", "finish_reason": "stop", "tool_calls": []})
    else:
        replies.extend({"content": "", "finish_reason": "tool_calls",
                        "tool_calls": [_tool_call(f"call-{turn}", f"query {turn}")]}
                       for turn in range(1, search_rounds + 1))
        if search_rounds < counterfactual_runner.MAX_ROUNDS:
            replies.append({"content": "natural language intermediate", "finish_reason": "stop",
                            "tool_calls": []})
    replies.append({"content": json.dumps({"status": "answered", "answer": "1.25",
                                             "evidence_ids": []}),
                    "finish_reason": "stop", "tool_calls": []})

    record, backend = _execute_cf1_with_replies(monkeypatch, replies)

    assert record["status"] == "ok"
    assert record["tool_calls"] == search_rounds
    assert len(backend.requests) == expected_requests <= 3
    assert len(record["retrieval_trace"]) == search_rounds <= 2
    final_request = backend.requests[-1]
    assert final_request["tools"] == []
    assert final_request["structured_stage"] == "counterfactual"
    assert final_request["messages"][-1] == {"role": "user", "content": counterfactual_runner.financebench.FINAL}
    for request in backend.requests:
        _assert_tool_message_pairs(request["messages"])


def test_cf1_rejects_illegal_tool_call_without_final_retry(monkeypatch):
    record, backend = _execute_cf1_with_replies(monkeypatch, [{
        "content": "", "finish_reason": "tool_calls",
        "tool_calls": [_tool_call("bad-call", "irrelevant", name="shell")],
    }])

    assert record["status"] == "invalid_output"
    assert len(backend.requests) == 1
    assert record["validation_error"]["phase"] == "tool_round_1"
    assert record["validation_error"]["exception_type"] == "ValueError"


def test_cf1_final_schema_failure_is_recorded_without_repair(monkeypatch):
    record, backend = _execute_cf1_with_replies(monkeypatch, [
        {"content": "", "finish_reason": "stop", "tool_calls": []},
        {"content": "not-json with private details", "finish_reason": "stop", "tool_calls": []},
    ])

    assert record["status"] == "invalid_output"
    assert len(backend.requests) == 2
    diagnostic = record["validation_error"]
    assert diagnostic["phase"] == "final_validation"
    assert diagnostic["exception_type"] == "ValidationError"
    assert diagnostic["error_count"] > 0
    assert all(set(error) == {"loc", "type"} for error in diagnostic["errors"])
    assert "private details" not in json.dumps(diagnostic)


def test_cf1_budget_refusal_stops_without_final_call_or_retry(monkeypatch):
    record, backend = _execute_cf1_with_replies(monkeypatch, [ModelCallError("token_budget_exhausted", 1,
                                                                              budget_exhausted=True)])

    assert record["status"] == "token_budget_exhausted"
    assert len(backend.requests) == 1
    assert record["validation_error"]["phase"] == "tool_round_1"
    assert record["validation_error"]["exception_type"] == "ModelCallError"


@pytest.mark.skipif(platform.system() != "Linux" or shutil.which("unshare") is None,
                    reason="requires Linux user/mount/network namespaces")
def test_actual_isolated_mock_runner_and_resume(tmp_path):
    output = tmp_path / "cf-run"
    manifest = run(DEV, output, backend="mock", limit=1)
    assert manifest["status"] == "complete"
    assert manifest["recorded_cells"] == 2
    lines = (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert {row["arm"] for row in rows} == {"CF0", "CF1"}
    for row in rows:
        sandbox = row["sandbox"]
        assert all(value == "denied" for value in sandbox["host_paths"].values())
        assert sandbox["host_paths"]["symlink_to_scoring_reference"] == "denied"
        assert sandbox["host_paths"]["relative_traversal_to_host_data"] == "denied"
        assert sandbox["network_isolated"] is True
        assert sandbox["network_devices"] == ["lo"]
        assert sandbox["fd_escape"] is False
        assert sandbox["api_key_in_environment"] is False
        assert row["model"] == "mock-counterfactual"
        assert row["budget_tokens"] == 0
        assert all(call["charged_tokens"] == 0 for call in row["llm_calls"])
        assert not any(key.casefold() in {"gold", "seed", "reference"} for key in row)
    cf1 = next(row for row in rows if row["arm"] == "CF1")
    cf0_cell = str(output / "cells" / f"{cf1['id']}-CF0.json")
    assert cf1["sandbox"]["host_paths"][cf0_cell] == "denied"
    before = (output / "predictions.jsonl").read_bytes()
    resumed = run(DEV, output, backend="mock", resume=True, limit=1)
    assert resumed["status"] == "complete"
    assert (output / "predictions.jsonl").read_bytes() == before


def test_llm_provider_initializer_returns_token_budget_object_without_http(monkeypatch):
    class FakeBudget:
        pass

    class FakeClient:
        def close(self):
            pass

    budget, opened_budget, client = FakeBudget(), object(), FakeClient()
    captured = {}

    def make_backend(*args, **kwargs):
        captured["args"], captured["kwargs"] = args, kwargs
        return SimpleNamespace(client=client, call_log=[], set_budget_context=lambda **values: None)

    monkeypatch.setenv("TALITE_MODEL", "glm-5.3-flash")
    monkeypatch.delenv("TALITE_BASE_URL", raising=False)
    monkeypatch.setenv("TALITE_API_KEY", "offline-test-placeholder")
    monkeypatch.setattr(counterfactual_runner.b1, "_open_b1_budget",
                        lambda path, scope, available: opened_budget)
    monkeypatch.setattr(counterfactual_runner, "TokenBudget", lambda **kwargs: budget)
    monkeypatch.setattr(counterfactual_runner.financebench, "client_from_env",
                        lambda **kwargs: (client, "glm-5.3-flash", "open.bigmodel.cn"))
    monkeypatch.setattr(counterfactual_runner.financebench, "ModelBackend", make_backend)

    provider, returned_budget, ledger_hash = counterfactual_runner._provider_for_cell("CF0:test")

    assert returned_budget is budget
    assert not hasattr(returned_budget, "token_budget")
    assert provider.client is client
    assert captured["kwargs"]["max_retries"] == 0
    assert len(ledger_hash) == 64


@pytest.mark.skipif(platform.system() != "Linux" or shutil.which("unshare") is None,
                    reason="requires Linux user/mount/network namespaces")
def test_llm_runner_budget_initialization_and_broker_are_offline(monkeypatch, tmp_path):
    class FakeBudget:
        def __init__(self):
            self.snapshots = 0

        def capacity(self, *, experiment_id, phase):
            return {"available": counterfactual_runner.EXPERIMENT_LIMIT}

        def snapshot(self, *, experiment_id, phase):
            self.snapshots += 1
            return {"experiment_id": experiment_id, "phase": phase, "charged_tokens": 0}

    class FakeClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProvider:
        def __init__(self):
            self.client = FakeClient()
            self.call_log = []
            self.request_shapes = []

        def set_budget_context(self, **kwargs):
            return None

        def _call(self, messages, tools, **kwargs):
            self.request_shapes.append({"tools": tools, "structured_stage": kwargs.get("structured_stage")})
            self.call_log.append({"actual_model": "offline-provider-mock", "charged_tokens": 0,
                                  "status": "ok"})
            if tools:
                return {"content": "Search complete.", "finish_reason": "stop", "tool_calls": []}
            return {"content": json.dumps({"status": "abstained", "answer": "",
                                             "evidence_ids": []}),
                    "finish_reason": "stop", "tool_calls": []}

    providers, budgets = [], []

    def fake_provider_for_cell(task_identity):
        provider, budget = FakeProvider(), FakeBudget()
        providers.append(provider)
        budgets.append(budget)
        return provider, budget, "a" * 64

    monkeypatch.delenv("TALITE_API_KEY", raising=False)
    monkeypatch.setattr(counterfactual_runner, "_provider_for_cell", fake_provider_for_cell)
    output = tmp_path / "llm-broker-offline"
    manifest = run(DEV, output, backend="llm", limit=1)

    assert manifest["status"] == "complete"
    assert manifest["recorded_cells"] == 2
    assert len(providers) == 3  # preflight plus one trusted provider per experiment arm
    assert all(provider.client.closed for provider in providers)
    assert all(budget.snapshots >= 1 for budget in budgets[1:])
    records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(row["status"] == "ok" and row["budget_tokens"] == 0 for row in records)
    assert [len(row["llm_calls"]) for row in records] == [1, 2]
    assert providers[1].request_shapes == [{"tools": [], "structured_stage": "counterfactual"}]
    assert providers[2].request_shapes == [
        {"tools": counterfactual_runner.TOOLS, "structured_stage": "counterfactual"},
        {"tools": [], "structured_stage": "counterfactual"},
    ]
    assert sum(len(provider.call_log) for provider in providers) == 3
    assert manifest["token_budget"]["charged_tokens"] == 0
