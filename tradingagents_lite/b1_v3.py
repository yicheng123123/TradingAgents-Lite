"""Run and resume Flash V3 Smoke using instance checkpoints and the existing ledger."""
from contextlib import contextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sqlite3

from . import b1, b1_v2, prompts
from .data import read_tasks, sha256
from .provider import DEFAULT_BASE_URL, ModelBackend, client_from_env
from .runner import run
from .schema import Budget
from .structured_output import schema_hashes


VERSION = "b1-flash-v3"
SMOKE_LIMIT = 50_000
VARIANTS = ("B1-M", "B1-E")
TASK_FAILURES = {
    "invalid_json", "missing_field", "extra_field", "string_too_long", "list_too_long",
    "string_too_short", "list_too_short", "invalid_field_type", "invalid_citation",
    "output_truncated", "unexpected_tool_calls", "unexpected_finish_reason",
    "tool_budget", "prompt_byte_limit",
}


def _frozen_protocol(tasks, market, evidence, b0, seed):
    files = {"tasks": tasks, "market_manifest": market / "manifest.json",
             "evidence_manifest": evidence / "source_manifest.json", "evidence": evidence / "evidence.jsonl",
             "b0_manifest": b0 / "run.json", "b0_trajectory": b0 / "trajectory.jsonl"}
    fingerprints = {key: sha256(path) for key, path in files.items()}
    expected = (b1.EXPECTED_TASKS_SHA256, b1.EXPECTED_MARKET_MANIFEST_SHA256,
                b1.EXPECTED_EVIDENCE_MANIFEST_SHA256, b1.EXPECTED_EVIDENCE_SHA256)
    if tuple(fingerprints[key] for key in ("tasks", "market_manifest", "evidence_manifest", "evidence")) != expected:
        raise ValueError("V3 requires the original frozen B1 inputs")
    reference = json.loads(files["b0_manifest"].read_text())
    if (reference.get("status") != "completed" or reference.get("ablation") != "B0"
            or reference.get("tasks_sha256") != expected[0] or reference.get("snapshot_sha256") != expected[1]
            or len(b1._load_jsonl(files["b0_trajectory"])) != 20
            or sha256(b0 / "tasks.jsonl") != fingerprints["b0_trajectory"]):
        raise ValueError("V3 requires the original completed B0 archive")
    protocol = b1_v2._protocol(seed, market_hash=expected[1], evidence_manifest_hash=expected[2],
                               evidence_hash=expected[3], b0_run_hash=fingerprints["b0_manifest"],
                               b0_trajectory_hash=fingerprints["b0_trajectory"])
    config = {**protocol["model_config"], "protocol_version": VERSION, "response_format": "json_object",
              "max_structured_repairs_per_instance": 1}
    # The tree identity lets a run resume after an archive-only commit.
    protocol.pop("code_git_sha")
    protocol.update(
        version=VERSION, source_tree=b1._git("rev-parse", "HEAD:tradingagents_lite"),
        protocol_document_sha256=sha256(Path("docs/b1-v3-protocol.md")),
        prompt_hashes={key: b1._json_hash(getattr(prompts, key)) for key in (
            "ANALYST_SYSTEM_PROMPT_V3", "TRADER_SYSTEM_PROMPT_V3", "FINAL_JSON_V3", "REPAIR_JSON_V3")},
        response_schema_hashes=schema_hashes(), model_config=config, smoke_token_limit=SMOKE_LIMIT,
        max_calls_per_task=5, max_structured_repairs_per_instance=1,
    )
    return protocol


@contextmanager
def _lock_archive(path):
    import fcntl

    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("V3 archive is already running") from None
        yield
    finally:
        os.close(fd)


def _save(path, value):
    temp = path.with_suffix(".tmp")
    b1._write_json(temp, value)
    temp.replace(path)


def _read_cell(path, task_id, variant, protocol_hash):
    run_meta = json.loads((path / "run.json").read_text())
    if (run_meta["ablation"] != variant
            or run_meta["protocol_metadata"].get("protocol_sha256") != protocol_hash):
        raise ValueError("V3 cell protocol or arm differs")
    try:
        with sqlite3.connect((path / "state.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as conn:
            records = conn.execute("SELECT task_id, record FROM outcomes").fetchall()
    except sqlite3.Error as exc:
        raise ValueError("V3 instance checkpoint is unavailable. Reconcile its ledger before resuming.") from exc
    if len(records) != 1 or records[0][0] != task_id:
        raise ValueError("V3 instance was interrupted before its outcome checkpoint")
    row = json.loads(records[0][1])
    trajectory = path / "trajectory.jsonl"
    if row["task_id"] != task_id or (trajectory.exists() and b1._load_jsonl(trajectory) != [row]):
        raise ValueError("V3 cell trajectory differs from its outcome checkpoint")
    return row


def _system_fault(row):
    if row["status"] == "ok":
        return None
    for error in row["errors"]:
        if error["code"] not in TASK_FAILURES:
            return error["code"]
    return None if row["errors"] else "failed_instance_without_diagnostics"


def _structured_metrics(events):
    first = [event for event in events if event["attempt"] == "initial"]
    repairs = [event for event in events if event["attempt"] == "repair"]
    passed = sum(event["valid"] for event in first)
    recovered = sum(event["valid"] for event in repairs)
    return {
        "initial_valid": passed, "initial_outputs": len(first),
        "initial_valid_rate": passed / len(first) if first else None,
        "repair_successes": recovered, "repair_attempts": len(repairs),
        "repair_success_rate": recovered / len(repairs) if repairs else None,
        "repair_tokens_charged": sum(event["charged_tokens"] for event in repairs),
        "final_valid": passed + recovered,
        "final_valid_rate": (passed + recovered) / len(first) if first else None,
    }


def _checkpoint(out, manifest, rows, ledger, ledger_path):
    experiment = str(out.resolve())
    budget = ledger.snapshot(experiment_id=experiment, phase="B1")
    manifest["token_budget"] = budget
    manifest["ledger_audit"] = b1._verify_ledger_records(
        [row for _, _, row in rows], ledger.attempts(experiment_id=experiment),
        manifest=manifest, ledger_path=ledger_path, expected_model=b1.B1_V2_MODEL,
    )
    cells = {}
    for task, variant, row in rows:
        key = f"{task}/{variant}"
        digest = b1._json_hash(row)
        previous = manifest["cells"].get(key)
        if previous and previous["checkpoint_sha256"] != digest:
            raise ValueError("V3 outcome checkpoint hash changed")
        cells[key] = {
            "status": row["status"], "errors": row["errors"], "checkpoint_sha256": digest,
            "structured_outputs": row["structured_output_trace"],
            "actual_models": sorted({call["actual_model"] for call in row["llm_calls"] if call.get("actual_model")}),
            "http_requests": row["usage"]["calls"], "charged_tokens": row["usage"]["total_tokens_charged"],
        }
    if manifest["cells"].keys() - cells.keys():
        raise ValueError("A V3 checkpoint was removed")
    manifest["cells"] = cells
    manifest["structured"] = {
        label: _structured_metrics([event for key, cell in cells.items()
                                    if label == "all" or key.endswith("/" + label)
                                    for event in cell["structured_outputs"]])
        for label in ("all", *VARIANTS)
    }
    manifest["completed_instances"] = sum(cell["status"] == "ok" for cell in cells.values())
    manifest["failed_instances"] = len(cells) - manifest["completed_instances"]
    manifest["unattempted_instances"] = len(manifest["plan"]) - len(cells)
    manifest["valid_pairs"] = sum(all(cells.get(f"{task}/{variant}", {}).get("status") == "ok"
                                      for variant in VARIANTS) for task in b1.SMOKE_TASK_IDS)
    _save(out / "manifest.json", manifest)


def run_b1_v3(tasks, market, evidence, b0, out: Path, *, seed=42, resume=False):
    protocol = _frozen_protocol(tasks, market, evidence, b0, seed)
    protocol_hash = b1._json_hash(protocol)
    selected = read_tasks(tasks)[:2]
    if tuple(task.task_id for task in selected) != tuple(b1.SMOKE_TASK_IDS):
        raise ValueError("V3 Smoke requires the frozen dev-01 and dev-02 tasks")
    plan = [[task.task_id, variant] for task in selected for variant in VARIANTS]
    ledger_path = b1._ledger_path()
    ledger = b1._open_b1_budget(ledger_path, "smoke", bool(os.getenv("TALITE_API_KEY", "").strip()),
                                smoke_limit=SMOKE_LIMIT)
    if ledger is None:
        raise ValueError("V3 Smoke requires TALITE_API_KEY")
    ledger_hash = b1._ledger_path_hash(ledger_path)
    experiment = str(out.resolve())
    if not resume:
        if ledger.attempts(experiment_id=experiment):
            raise ValueError("The ledger already contains attempts for this V3 archive. Use its checkpoint.")
        out.mkdir(parents=True, exist_ok=False)
    with _lock_archive(out):
        if resume:
            manifest = json.loads((out / "manifest.json").read_text())
            if (manifest["protocol_sha256"] != protocol_hash or manifest["protocol"] != protocol
                    or manifest["plan"] != plan or manifest["token_budget_ledger_path_sha256"] != ledger_hash):
                raise ValueError("V3 resume requires the same protocol, task plan and ledger")
        else:
            manifest = {
                "status": "preflight", "protocol": protocol, "protocol_sha256": protocol_hash,
                "code_git_sha": b1._git("rev-parse", "HEAD"), "plan": plan, "cells": {},
                "token_budget_ledger_path_sha256": ledger_hash, "started_at": datetime.now(UTC).isoformat(),
            }
            _save(out / "manifest.json", manifest)

        def cell_path(task, variant):
            return out / "cells" / task / variant

        rows = []
        for task, variant in plan:
            path = cell_path(task, variant)
            if path.exists():
                rows.append((task, variant, _read_cell(path, task, variant, protocol_hash)))
        _checkpoint(out, manifest, rows, ledger, ledger_path)
        seen = {(task, variant) for task, variant, _ in rows}
        if len(seen) == len(plan):
            if manifest["status"] == "running":
                fault = _system_fault(rows[-1][2])
                manifest.update(status="stopped_system_fault" if fault else
                                "completed_with_task_failures" if manifest["failed_instances"] else "completed",
                                system_fault=fault, finished_at=datetime.now(UTC).isoformat())
                _save(out / "manifest.json", manifest)
            return manifest

        budget = Budget.model_validate(protocol["budget"])
        client, model, host = client_from_env(default_model=b1.B1_V2_MODEL, trust_env=False, follow_redirects=False)
        try:
            if model != b1.B1_V2_MODEL or host != "open.bigmodel.cn" or str(client.base_url).rstrip("/") != DEFAULT_BASE_URL:
                raise ValueError("V3 pins the official glm-5.3-flash endpoint and model")
            manifest.update(status="running", system_fault=None)
            _save(out / "manifest.json", manifest)
            for task, variant in plan:
                if (task, variant) in seen:
                    continue
                task_slice = out / "task-slices" / f"{task}.jsonl"
                task_slice.parent.mkdir(exist_ok=True)
                task_slice.write_bytes(b1._selected_task_bytes(tasks, [task]))
                path = cell_path(task, variant)

                def backend():
                    return ModelBackend(client, model, budget, None, max_retries=0,
                                        max_prompt_bytes=protocol["max_prompt_bytes"],
                                        temperature=protocol["temperature"], reasoning_effort=protocol["reasoning_effort"],
                                        token_budget=ledger, phase="B1", require_model_identity=True)

                run(task_slice, market, path, backend, protocol["model_config"], budget, seed, True,
                    evidence_path=evidence if variant == "B1-E" else None, run_label=variant,
                    protocol_metadata={"protocol_sha256": protocol_hash},
                    token_budget=ledger, experiment_id=experiment, phase="B1")
                row = _read_cell(path, task, variant, protocol_hash)
                rows.append((task, variant, row))
                _checkpoint(out, manifest, rows, ledger, ledger_path)
                print(f"Flash V3 {task} {variant}: {row['status']}, "
                      f"{manifest['token_budget']['experiment_used']} tokens so far", flush=True)
                fault = _system_fault(row)
                if fault:
                    manifest.update(status="stopped_system_fault", system_fault=fault)
                    break
            else:
                manifest["status"] = "completed_with_task_failures" if manifest["failed_instances"] else "completed"
        finally:
            client.close()
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        _save(out / "manifest.json", manifest)
        return manifest
