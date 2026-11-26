"""Frozen B1 paired smoke/development protocol for GLM and A0.5 evidence."""
import csv
from datetime import UTC, datetime
from hashlib import sha256 as _sha256
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import subprocess

from .data import read_tasks, sha256
from .evidence import EvidenceStore
from .provider import (
    DEFAULT_BASE_URL, LEGACY_B1_MODEL as DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_TEMPERATURE,
    ModelBackend, client_from_env,
)
from .prompts import prompt_hashes
from .runner import clustered_bootstrap, run, summary
from .schema import Budget
from .token_budget import (
    DEFAULT_LEDGER_PATH, GLOBAL_TOKEN_LIMIT, PHASE_TOKEN_LIMITS,
    PROTOCOL_TOKEN_ALLOWANCE, SMOKE_TOKEN_LIMIT, TokenBudget,
)
from .workflow import MomentumBackend, specs
from .tool_registry import FINANCIAL_SPEC, SPECS


EXPECTED_TASKS_SHA256 = "2d0e4f92d583d64ceba729b4e3ffa0004b3bd29a5a00d990434dcdab5fceedae"
EXPECTED_MARKET_MANIFEST_SHA256 = "f5e4c5095953ce737b7b99bba71b2ad26a9f30d2f34b9fba8320027e46e0f3cb"
EXPECTED_EVIDENCE_MANIFEST_SHA256 = "b46f9ef2ebc1c08022500282147940985eb9d3ceabe8cdaf78eec6efa190e4dc"
EXPECTED_EVIDENCE_SHA256 = "b0ffce595575257231432650495d756681a9f4ac10f5deee972110c7a7b7a532"
SMOKE_TASK_IDS = ("dev-01", "dev-02")
MAX_TOOL_ROUNDS = 2
MAX_TOKENS = 512
MAX_PROMPT_BYTES = 5_000
MAX_RETRIES = 0
TEMPERATURE = DEFAULT_TEMPERATURE
REASONING_EFFORT = DEFAULT_REASONING_EFFORT
PRICE_CNY_PER_MILLION = {"input": 8.0, "output": 28.0}
PRICE_SOURCE = "https://docs.bigmodel.cn/cn/guide/start/pricing"
PRICE_AS_OF = "2026-11-24"
SETTLEMENT_PROTOCOL = "next shared session open to horizon-th shared session close; alpha vs SPY; no costs"
RESUME_PROTOCOL_VERSION = "b1-smoke-reuse-v1"
RESUME_PROTOCOL_DOCUMENT = Path(__file__).resolve().parents[1] / "docs" / "b1-resume-protocol.md"

# Independently frozen Flash protocol; leave the legacy GLM-5.3 constants above unchanged.
B1_V2_PROTOCOL_VERSION = "b1-flash-v2"
B1_V2_MODEL = "glm-5.3-flash"
B1_V2_MAX_TOOL_CALLS_PER_ROUND = 4
B1_V2_MAX_PROMPT_BYTES = 8_000
B1_V2_PRICE_CNY_PER_MILLION = {"input": 0.8, "cached_input": 0.23, "output": 2.8}
B1_V2_PRICE_SOURCE = "https://docs.bigmodel.cn/cn/guide/start/pricing"
B1_V2_PRICE_AS_OF = "2026-11-26"

def _json_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return _sha256(encoded).hexdigest()


def _b1_protocol(seed: int) -> dict:
    return {
        "model_requested": DEFAULT_MODEL, "base_url": DEFAULT_BASE_URL,
        "temperature": TEMPERATURE, "max_tokens_per_model_call": MAX_TOKENS,
        "max_tool_rounds": MAX_TOOL_ROUNDS, "max_tool_calls_per_round": 2,
        "tool_retries": 0, "model_max_retries": MAX_RETRIES,
        "model_timeout_seconds": 25, "max_prompt_bytes": MAX_PROMPT_BYTES,
        "provider_generation_seed": None, "statistics_seed": seed,
        "reasoning_effort": REASONING_EFFORT, "thinking": "enabled", "sdk_retries": 0,
        "reasoning_tokens_reserved_separately": MAX_TOKENS,
        "settlement": SETTLEMENT_PROTOCOL, "holdout_accessed": False,
        "rag": False, "memory": False, "debate": False, "mcp": False,
        "verifier_extension": False,
    }


def _tool_schema_hashes() -> dict[str, str]:
    market = specs(SPECS)
    evidence = specs({**SPECS, "financial_evidence": FINANCIAL_SPEC})
    return {"B1-M": _json_hash(market), "B1-E": _json_hash(evidence)}


def _ledger_path() -> Path:
    return Path(os.getenv("TALITE_TOKEN_BUDGET_DB") or DEFAULT_LEDGER_PATH).expanduser()


def _open_b1_budget(path: Path, scope: str, api_key_available: bool, *,
                    smoke_limit: int = SMOKE_TOKEN_LIMIT) -> TokenBudget | None:
    if not api_key_available:
        return None
    if not path.is_file():
        raise ValueError("The existing project token ledger is missing; refusing to start B1 with an empty ledger")
    required_columns = {
        "id", "created_at", "experiment_id", "task_id", "phase", "model", "logical_call",
        "attempt_number", "is_retry", "request_sha256", "request_bytes", "reservation_tokens",
        "charged_tokens", "status", "input_tokens", "output_tokens", "reasoning_tokens",
        "cached_tokens", "reasoning_in_output", "usage_known", "usage_missing", "error_code", "deny_reason",
    }
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(token_attempts)")}
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise ValueError("The existing project token ledger is unreadable; refusing to initialize it") from exc
    if not required_columns.issubset(columns):
        raise ValueError("The existing project token ledger has no complete attempt schema; refusing to initialize it")
    return TokenBudget(db_path=path, experiment_limit=smoke_limit if scope == "smoke" else None)


def _ledger_path_hash(path: Path) -> str:
    return _sha256(str(path.resolve()).encode("utf-8")).hexdigest()


def _resume_compatibility(*, seed: int, market_hash: str,
                          evidence_manifest_hash: str, evidence_hash: str,
                          b0_run_hash: str, b0_trajectory_hash: str) -> dict:
    if not RESUME_PROTOCOL_DOCUMENT.is_file():
        raise ValueError("B1 resume protocol document is missing; refusing an unfrozen run")
    tracked_changes = _git("status", "--porcelain", "--untracked-files=no")
    if tracked_changes:
        raise ValueError("Commit B1 protocol/code changes before starting a real B1 run")
    protocol = _b1_protocol(seed)
    return {
        "version": RESUME_PROTOCOL_VERSION,
        "code_git_sha": _git("rev-parse", "HEAD"),
        "tracked_worktree_clean": True,
        "lock_sha256": sha256(Path("uv.lock")),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("langgraph", "pydantic", "openai")},
        "protocol": protocol,
        "budget": Budget(max_tool_rounds=MAX_TOOL_ROUNDS, max_tokens=MAX_TOKENS).model_dump(),
        "react": True,
        "full_tasks_sha256": EXPECTED_TASKS_SHA256,
        "market_manifest_sha256": market_hash,
        "evidence_manifest_sha256": evidence_manifest_hash,
        "evidence_sha256": evidence_hash,
        "b0_run_metadata_sha256": b0_run_hash,
        "b0_trajectory_sha256": b0_trajectory_hash,
        "prompt_hashes": prompt_hashes(),
        "tool_schema_hashes": _tool_schema_hashes(),
        "resume_protocol_document_sha256": sha256(RESUME_PROTOCOL_DOCUMENT),
    }


def _read_ledger_attempts(path: Path, experiment_id: str) -> list[dict]:
    if not path.is_file():
        raise ValueError("The existing project token ledger is missing; refusing to assume zero prior usage")
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT * FROM token_attempts WHERE experiment_id = ? ORDER BY id", (experiment_id,)
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise ValueError("The B1 token ledger cannot be read for archive verification") from exc
    return [dict(row) for row in rows]


def _verify_ledger_records(rows: list[dict], attempts: list[dict], *,
                           manifest: dict, ledger_path: Path,
                           expected_model: str = DEFAULT_MODEL) -> dict:
    budget = manifest.get("token_budget") or {}
    experiment_id = budget.get("experiment_id")
    if not experiment_id:
        raise ValueError("B1 archive has no token ledger experiment ID")
    if manifest.get("token_budget_ledger_path_sha256") != _ledger_path_hash(ledger_path):
        raise ValueError("B1 archive points to a different token ledger")
    logged = {}
    for row in rows:
        for call in row.get("llm_calls", []):
            for field in ("request_sha256", "prompt_sha256", "tools_sha256"):
                value = call.get(field)
                try:
                    valid_hash = isinstance(value, str) and len(value) == 64 and len(bytes.fromhex(value)) == 32
                except ValueError:
                    valid_hash = False
                if not valid_hash:
                    raise ValueError("B1 request trajectory is missing a valid SHA-256")
            attempt_number = call.get("attempt")
            if not attempt_number:
                if call.get("status") != "blocked" or call.get("error_code") != "prompt_byte_limit":
                    raise ValueError("B1 trajectory contains an unaccounted request record")
                continue
            key = (row.get("task_id"), call.get("logical_call"), attempt_number,
                   call.get("request_sha256"), call.get("request_bytes"))
            if None in key:
                raise ValueError("B1 trajectory has an incomplete request identity")
            logged.setdefault(key, []).append(call)
    recorded = {}
    for item in attempts:
        key = (item.get("task_id"), item.get("logical_call"), item.get("attempt_number"),
               item.get("request_sha256"), item.get("request_bytes"))
        if None in key:
            raise ValueError("B1 token ledger has an incomplete attempt identity")
        recorded.setdefault(key, []).append(item)
    if logged.keys() != recorded.keys():
        raise ValueError("B1 trajectory and token ledger attempt records do not match")
    for key, group_calls in logged.items():
        group_attempts = recorded[key]
        if len(group_calls) != len(group_attempts):
            raise ValueError("B1 trajectory and token ledger request multiplicities do not match")
        for call, item in zip(group_calls, sorted(group_attempts, key=lambda record: record["id"])):
            if item.get("model") != expected_model:
                raise ValueError("B1 request model differs from the token ledger")
            status = call.get("status")
            ledger_status = item.get("status")
            allowed = ({"succeeded", "usage_missing"} if status == "ok" else
                       {"failed", "usage_missing"} if status in {"error", "retrying"} else
                       {"denied"} if status == "blocked" and call.get("error_code") == "token_budget_exhausted" else set())
            if ledger_status not in allowed:
                raise ValueError("B1 request status differs from its token ledger record")
            reserved = call.get("reserved_tokens", call.get("required_tokens"))
            if reserved != item.get("reservation_tokens"):
                raise ValueError("B1 request reservation differs from the token ledger")
            if call.get("charged_tokens") is not None and call["charged_tokens"] != item.get("charged_tokens"):
                raise ValueError("B1 request charge differs from the token ledger")
            for call_key, ledger_key in (("prompt_tokens", "input_tokens"),
                                         ("completion_tokens", "output_tokens"),
                                         ("reasoning_tokens", "reasoning_tokens"),
                                         ("cached_tokens", "cached_tokens")):
                if call.get(call_key) is not None and call.get(call_key) != item.get(ledger_key):
                    raise ValueError("B1 usage detail differs from the token ledger")
    if budget.get("attempts_recorded") != len(attempts):
        raise ValueError("B1 archived budget snapshot does not match the ledger")
    charged = sum(item.get("charged_tokens") or 0 for item in attempts)
    http_attempts = sum(item.get("status") != "denied" for item in attempts)
    if budget.get("conservatively_charged_tokens") != charged or budget.get("http_attempts") != http_attempts:
        raise ValueError("B1 archived token totals do not match the ledger")
    if budget.get("experiment_used") != charged:
        raise ValueError("B1 experiment cumulative charge differs from the token ledger")
    return {
        "experiment_id": experiment_id,
        "attempt_count": len(attempts),
        "http_attempt_count": http_attempts,
        "failed_attempt_count": sum(item.get("status") in {"failed", "usage_missing"} for item in attempts),
        "conservatively_charged_tokens": charged,
        "attempt_ids": [item["id"] for item in attempts],
    }


def _selected_task_bytes(tasks_path: Path, task_ids: list[str]) -> bytes:
    lines = _read_original_lines(tasks_path)
    if set(task_ids) - lines.keys():
        raise ValueError("Requested B1 task is absent from the frozen task set")
    return "".join(lines[task_id] + "\n" for task_id in task_ids).encode("utf-8")


def _read_variant_archive(root: Path, variant: str, selected_sha256: str,
                          compatibility: dict, manifest: dict) -> dict:
    run_dir = root / variant
    run_path, tasks_path, trajectory_path = run_dir / "run.json", run_dir / "tasks.jsonl", run_dir / "trajectory.jsonl"
    if not all(path.is_file() for path in (run_path, tasks_path, trajectory_path)):
        raise ValueError(f"{variant} archive is missing run metadata or trajectory")
    run_meta = json.loads(run_path.read_text(encoding="utf-8"))
    if sha256(root / "tasks.jsonl") != selected_sha256 or run_meta.get("tasks_sha256") != selected_sha256:
        raise ValueError(f"{variant} input task hash differs from the frozen archive")
    if tasks_path.read_bytes() != trajectory_path.read_bytes():
        raise ValueError(f"{variant} trajectory is incomplete or changed")
    recorded_trajectory_hash = (manifest.get("trajectory_sha256") or {}).get(variant)
    if recorded_trajectory_hash is None or recorded_trajectory_hash != sha256(trajectory_path):
        raise ValueError(f"{variant} trajectory hash is missing or invalid")
    rows = _load_jsonl(trajectory_path)
    ids = [row.get("task_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{variant} trajectory repeats a task instance")
    if run_meta.get("processed_tasks") != len(rows) or run_meta.get("requested_tasks") != len(ids):
        raise ValueError(f"{variant} trajectory task count does not match run metadata")
    if run_meta.get("ablation") != variant or run_meta.get("status") != "completed":
        raise ValueError(f"{variant} run is not a completed archived run")
    config = run_meta.get("config") or {}
    frozen_config = {
        "backend": "llm", "model": DEFAULT_MODEL, "api_host": "open.bigmodel.cn",
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "thinking": "enabled",
        "reasoning_effort": REASONING_EFFORT, "provider_seed": None, "phase": "B1",
        "sdk_retries": 0, "application_retries": MAX_RETRIES, "max_prompt_bytes": MAX_PROMPT_BYTES,
        "pricing_cny_per_million": PRICE_CNY_PER_MILLION, "pricing_source": PRICE_SOURCE,
        "pricing_as_of": PRICE_AS_OF,
    }
    if config != frozen_config:
        raise ValueError(f"{variant} model or request configuration differs from frozen B1")
    metadata = run_meta.get("protocol_metadata") or {}
    expected_schema_hash = compatibility["tool_schema_hashes"][variant]
    if (metadata.get("prompt_hashes") != compatibility["prompt_hashes"] or
            metadata.get("tool_schema_hashes", {}).get(variant) != expected_schema_hash or
            metadata.get("b1_compatibility_sha256") != manifest.get("resume_compatibility_sha256") or
            metadata.get("resume_protocol_document_sha256") != compatibility["resume_protocol_document_sha256"] or
            metadata.get("max_model_retries") != MAX_RETRIES or
            metadata.get("max_prompt_bytes") != MAX_PROMPT_BYTES or
            metadata.get("model_timeout_seconds") != 25 or
            metadata.get("provider_generation_seed") is not None or
            metadata.get("holdout_accessed") is not False):
        raise ValueError(f"{variant} protocol metadata differs from frozen B1")
    if (run_meta.get("snapshot_sha256") != compatibility["market_manifest_sha256"] or
            run_meta.get("execution") != "next shared session open to horizon-th shared session close, OHLC price return" or
            run_meta.get("alpha") != "position direction × (asset price return − SPY price return); no costs" or
            run_meta.get("budget") != compatibility["budget"] or
            run_meta.get("react") is not compatibility["react"] or
            run_meta.get("lock_sha256") != compatibility["lock_sha256"] or
            run_meta.get("dependencies") != compatibility["dependencies"] or
            run_meta.get("token_budget_phase") != "B1"):
        raise ValueError(f"{variant} data, settlement, runtime or budget protocol differs")
    if variant == "B1-E":
        if (run_meta.get("evidence_manifest_sha256") != compatibility["evidence_manifest_sha256"] or
                run_meta.get("evidence_sha256") != compatibility["evidence_sha256"]):
            raise ValueError("B1-E financial evidence hash differs from the frozen A0.5 corpus")
    if any(row.get("llm_calls") is None for row in rows):
        raise ValueError(f"{variant} trajectory is missing its original model-call log")
    actual_models = sorted({call.get("actual_model") for row in rows for call in row.get("llm_calls", [])
                            if call.get("actual_model")})
    if run_meta.get("actual_models_observed") != actual_models:
        raise ValueError(f"{variant} actual model identities differ from archived trajectories")
    summary_key = "b1_m_summary" if variant == "B1-M" else "b1_e_summary"
    archived_summary = manifest.get(summary_key)
    if not isinstance(archived_summary, dict):
        raise ValueError(f"{variant} summary is missing from its source manifest")
    recomputed = summary(rows, compatibility["protocol"]["statistics_seed"], config)
    for key, value in recomputed.items():
        if archived_summary.get(key) != value:
            raise ValueError(f"{variant} archived summary does not match its trajectory")
    return {"run": run_meta, "rows": rows, "actual_models": actual_models}


def _validate_b1_archive(root: Path, *, expected_scope: str, expected_ids: list[str],
                         expected_compatibility: dict | None = None,
                         require_success: bool = False,
                         frozen_tasks_path: Path | None = None) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("B1 source archive has no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compatibility = manifest.get("resume_compatibility")
    if not isinstance(compatibility, dict):
        raise ValueError("B1 source archive predates the auditable resume protocol")
    if _json_hash(compatibility) != manifest.get("resume_compatibility_sha256"):
        raise ValueError("B1 source compatibility manifest hash is invalid")
    if expected_compatibility is not None and compatibility != expected_compatibility:
        raise ValueError("B1 source does not match the frozen task, model, prompt, tool, data or settlement hashes")
    if (manifest.get("scope") != expected_scope or manifest.get("selected_task_ids") != expected_ids or
            manifest.get("full_tasks_sha256") != EXPECTED_TASKS_SHA256 or
            manifest.get("protocol") != compatibility.get("protocol") or
            manifest.get("prompt_hashes") != compatibility.get("prompt_hashes") or
            manifest.get("tool_schema_hashes") != compatibility.get("tool_schema_hashes") or
            manifest.get("code_git_sha") != compatibility.get("code_git_sha")):
        raise ValueError("B1 source scope, task selection or protocol metadata is inconsistent")
    tasks_path = root / "tasks.jsonl"
    if not tasks_path.is_file() or sha256(tasks_path) != manifest.get("selected_tasks_sha256"):
        raise ValueError("B1 source task definition is missing or changed")
    expected_task_bytes = _selected_task_bytes_from_source_lines(tasks_path, expected_ids)
    if expected_task_bytes != tasks_path.read_bytes():
        raise ValueError("B1 source task definitions do not match the frozen task order")
    if frozen_tasks_path is not None and _selected_task_bytes(frozen_tasks_path, expected_ids) != tasks_path.read_bytes():
        raise ValueError("B1 source task definitions differ from the original frozen task file")
    runs = {}
    for variant in ("B1-M", "B1-E"):
        runs[variant] = _read_variant_archive(root, variant, manifest["selected_tasks_sha256"],
                                              compatibility, manifest)
        if [row.get("task_id") for row in runs[variant]["rows"]] != expected_ids:
            raise ValueError(f"{variant} trajectory task order differs from the frozen selection")
    all_rows = runs["B1-M"]["rows"] + runs["B1-E"]["rows"]
    archive_actual_models = sorted({call.get("actual_model") for row in all_rows
                                    for call in row.get("llm_calls", []) if call.get("actual_model")})
    if manifest.get("actual_models_observed") != archive_actual_models:
        raise ValueError("B1 manifest actual model identities differ from the trajectories")
    experiment_id = (manifest.get("token_budget") or {}).get("experiment_id", "")
    ledger_attempts = _read_ledger_attempts(_ledger_path(), experiment_id)
    ledger_summary = _verify_ledger_records(
        all_rows, ledger_attempts, manifest=manifest, ledger_path=_ledger_path())
    runs["token_budget"] = ledger_summary
    if require_success:
        check = manifest.get("completion_check") or {}
        if manifest.get("status") != "completed" or not check.get("complete"):
            raise ValueError("Smoke did not complete successfully; no continuation or task retry is allowed")
        for variant in ("B1-M", "B1-E"):
            if runs[variant]["actual_models"] != [DEFAULT_MODEL] or any(
                    row.get("status") != "ok" or row.get("alpha") is None
                    for row in runs[variant]["rows"]):
                raise ValueError("Smoke contains a failed task or unconfirmed model identity")
    return {"manifest": manifest, "compatibility": compatibility, "runs": runs}


def _selected_task_bytes_from_source_lines(path: Path, task_ids: list[str]) -> bytes:
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_ids = [json.loads(line).get("task_id") for line in raw_lines]
    lines = _read_original_lines(path)
    if (len(source_ids) != len(set(source_ids)) or set(lines) != set(task_ids) or
            len(lines) != len(task_ids)):
        raise ValueError("B1 archive task file does not contain exactly the declared task IDs")
    return "".join(lines[task_id] + "\n" for task_id in task_ids).encode("utf-8")


def _merge_rows(smoke_rows: list[dict], continuation_rows: list[dict], expected_ids: list[str]) -> list[dict]:
    combined = {}
    for row in [*smoke_rows, *continuation_rows]:
        task_id = row.get("task_id")
        if not task_id or task_id in combined:
            raise ValueError("Cannot merge duplicate or unidentified B1 task instances")
        combined[task_id] = row
    if set(combined) != set(expected_ids):
        raise ValueError("Cannot merge B1 sources with missing or unexpected task trajectories")
    return [combined[task_id] for task_id in expected_ids]


def request_cost_ceiling(task_count: int) -> dict:
    """Conservative token and standard-price ceiling for the frozen B1 protocol."""
    workflows = task_count * 2
    logical_calls_per_task = MAX_TOOL_ROUNDS + 2  # final analyst response and Trader call
    attempts = workflows * logical_calls_per_task * (MAX_RETRIES + 1)
    protocol_token_allowance = PROTOCOL_TOKEN_ALLOWANCE
    input_token_upper = attempts * (MAX_PROMPT_BYTES + protocol_token_allowance)
    output_token_upper = attempts * MAX_TOKENS
    separate_reasoning_upper = attempts * MAX_TOKENS
    max_conservative_tokens = input_token_upper + output_token_upper + separate_reasoning_upper
    cost = (input_token_upper * PRICE_CNY_PER_MILLION["input"] +
            (output_token_upper + separate_reasoning_upper) * PRICE_CNY_PER_MILLION["output"]) / 1_000_000
    return {
        "tasks_per_condition": task_count,
        "conditions": 2,
        "max_logical_calls": workflows * logical_calls_per_task,
        "max_http_attempts_including_retries": attempts,
        "max_prompt_bytes_per_attempt": MAX_PROMPT_BYTES,
        "protocol_token_allowance_per_attempt": protocol_token_allowance,
        "input_token_upper_bound": input_token_upper,
        "max_output_tokens_per_attempt": MAX_TOKENS,
        "output_token_upper_bound": output_token_upper,
        "separate_reasoning_token_upper_bound": separate_reasoning_upper,
        "max_conservative_tokens": max_conservative_tokens,
        "list_price_cost_ceiling_cny": round(cost, 6),
        "price_cny_per_million": PRICE_CNY_PER_MILLION,
        "price_as_of": PRICE_AS_OF,
        "price_source": PRICE_SOURCE,
        "note": "Conservative byte/token envelope; provider invoice and cached-token rates may differ.",
    }


def _completion_check(requested_tasks: int, m_stats: dict, e_stats: dict, paired_n: int) -> dict:
    conditions = {}
    for name, stats in (("B1-M", m_stats), ("B1-E", e_stats)):
        conditions[name] = {
            "requested_tasks": requested_tasks,
            "processed_tasks": stats.get("tasks", 0),
            "completed_tasks": stats.get("completed_tasks", 0),
            "failed_tasks": stats.get("failed_tasks", 0),
            "complete": (
                stats.get("tasks", 0) == requested_tasks
                and stats.get("completed_tasks", 0) == requested_tasks
                and stats.get("failed_tasks", 0) == 0
            ),
        }
    return {
        "conditions": conditions,
        "requested_pairs": requested_tasks,
        "valid_pairs": paired_n,
        "complete": all(item["complete"] for item in conditions.values()) and paired_n == requested_tasks,
    }


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_original_lines(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            result[item["task_id"]] = line
    return result


def _summary_rows(variant: str, scope: str, metrics: dict, status: str = "completed") -> list[dict]:
    columns = (
        "tasks", "requested_tasks", "completed_tasks", "failed_tasks", "task_completion_rate", "structured_output_valid_rate",
        "structured_output_valid_tasks", "failure_status_counts", "directional_n", "directional_hits", "non_hold_coverage",
        "directional_hit_rate", "hit_rate_wilson_95", "mean_alpha", "alpha_date_bootstrap_95",
        "tool_success_rate", "tool_failure_events", "tool_recovery_events", "tool_recovery_rate",
        "tool_recovered_tasks", "evidence_citation_valid_rate", "invalid_citation_rate",
        "valid_citation_references", "invalid_citation_references",
        "financial_citation_references", "eligible_financial_citation_task_rate",
        "input_tokens", "output_tokens", "reasoning_tokens", "separate_reasoning_tokens",
        "cached_tokens", "total_tokens_charged", "missing_usage_calls",
        "reasoning_detail_missing_calls", "cached_detail_missing_calls",
        "token_budget_denials", "llm_calls", "llm_retries", "api_cost_cny_estimate",
        "input_tokens_per_task", "output_tokens_per_task", "reasoning_tokens_per_task",
        "total_tokens_charged_per_task", "completed_tasks_per_100k_tokens",
        "directional_hits_per_100k_tokens", "latency_mean_seconds", "latency_p50_seconds",
        "latency_p95_seconds", "llm_call_latency_mean_seconds", "llm_call_latency_p50_seconds",
        "llm_call_latency_p95_seconds",
    )
    output = []
    for ticker in [*sorted(metrics.get("by_ticker", {})), "ALL"]:
        item = metrics.get("by_ticker", {}).get(ticker, metrics) if ticker != "ALL" else metrics
        output.append({"variant": variant, "scope": scope, "ticker": ticker, "status": status,
                       **{key: item.get(key) for key in columns}})
    return output


def _write_summary_csv(path: Path, rows: list[dict]):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["variant", "scope", "ticker", "status"]
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def _paired(ms_rows: list[dict], evidence_rows: list[dict], seed: int) -> dict:
    m = {row["task_id"]: row for row in ms_rows}
    e = {row["task_id"]: row for row in evidence_rows}
    paired = []
    for task_id in sorted(m.keys() & e.keys()):
        left, right = m[task_id], e[task_id]
        if (left.get("status") == right.get("status") == "ok"
                and left.get("alpha") is not None and right.get("alpha") is not None):
            paired.append({
                "task_id": task_id, "symbol": left["symbol"], "decision_date": left["decision_date"],
                "alpha_b1_m": left["alpha"], "alpha_b1_e": right["alpha"],
                "alpha_delta_e_minus_m": right["alpha"] - left["alpha"],
                "action_b1_m": left["decision"]["action"], "action_b1_e": right["decision"]["action"],
            })
    deltas = [{"decision_date": row["decision_date"], "alpha": row["alpha_delta_e_minus_m"]}
              for row in paired]
    by_ticker = {}
    for symbol in sorted({row["symbol"] for row in paired}):
        group = [row for row in paired if row["symbol"] == symbol]
        by_ticker[symbol] = {
            "paired_n": len(group),
            "mean_alpha_delta_e_minus_m": (
                sum(row["alpha_delta_e_minus_m"] for row in group) / len(group) if group else None
            ),
            "action_changed_n": sum(row["action_b1_m"] != row["action_b1_e"] for row in group),
        }
    both_directional = [row for row in paired
                        if m[row["task_id"]].get("directional_hit") is not None
                        and e[row["task_id"]].get("directional_hit") is not None]
    wrong_to_right = sum(
        not m[row["task_id"]]["directional_hit"] and e[row["task_id"]]["directional_hit"]
        for row in both_directional
    )
    right_to_wrong = sum(
        m[row["task_id"]]["directional_hit"] and not e[row["task_id"]]["directional_hit"]
        for row in both_directional
    )
    return {
        "paired_n": len(paired),
        "mean_alpha_delta_e_minus_m": (
            sum(row["alpha_delta_e_minus_m"] for row in paired) / len(paired) if paired else None
        ),
        "alpha_delta_date_bootstrap_95": clustered_bootstrap(deltas, seed),
        "date_clusters": len({row["decision_date"] for row in paired}),
        "action_changed_n": sum(row["action_b1_m"] != row["action_b1_e"] for row in paired),
        "directional_transition_n": len(both_directional),
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "by_ticker": by_ticker,
        "tasks": paired,
        "interval_note": "No cluster interval is emitted with fewer than two paired decision dates.",
    }


def _report(manifest: dict, rows: list[dict], paired: dict | None, b0_full: dict,
            b0_scope: dict, blocked_reason: str | None = None) -> str:
    lines = [
        "# B1 Real LLM Development Baseline",
        "",
        f"Status: {manifest['status']}. Scope: {manifest['scope']}.",
        "",
        "B0 is the deterministic five-session momentum rule. B1-M adds the real model with market-only tools; "
        "B1-E uses the same model and core prompts with the cutoff-bound A0.5 financial-evidence tool. "
        "No B1 task is scored unless the workflow, structured outputs, citations, and settlement all complete.",
        "",
        f"Frozen Development task SHA-256: {manifest['full_tasks_sha256']}",
        f"Selected task IDs: {', '.join(manifest['selected_task_ids'])}",
        f"Market manifest SHA-256: {manifest['market_manifest_sha256']}",
        f"Evidence manifest SHA-256: {manifest['evidence_manifest_sha256']}",
        f"Evidence JSONL SHA-256: {manifest['evidence_sha256']}",
        f"Core prompt hashes: {json.dumps(manifest['prompt_hashes'], ensure_ascii=False, sort_keys=True)}",
        "",
        "## Results",
        "",
        "| Variant | Scope | Ticker | Completed / requested | Failed | Completion | Structured valid | Non-Hold coverage | Hits / Non-Hold N | Hit rate | Mean alpha / 95% date CI | Tool success | Citation valid / invalid | LLM calls | Input / output / reasoning / charged / average tokens per task | Tasks per 100k Tokens / hits per 100k Tokens | Task latency mean/P50/P95; call latency mean/P50/P95 (seconds) | API cost (CNY) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        completed = row.get("completed_tasks")
        requested = row.get("requested_tasks", row.get("tasks"))
        rate = row.get("directional_hit_rate")
        alpha = row.get("mean_alpha")
        valid_citation = row.get("evidence_citation_valid_rate")
        invalid_citation = row.get("invalid_citation_rate")
        tool_success = row.get("tool_success_rate")
        recovery = f"{row.get('tool_recovery_events', 0)} / {row.get('tool_recovery_rate')}"
        citation = f"{valid_citation if valid_citation is not None else '-'} / {invalid_citation if invalid_citation is not None else '-'}"
        lines.append(
            f"| {row['variant']} | {row['scope']} | {row['ticker']} | "
            f"{completed if completed is not None else '-'} / {requested if requested is not None else '-'} | "
            f"{row.get('failed_tasks') if row.get('failed_tasks') is not None else '-'} | "
            f"{row.get('task_completion_rate') if row.get('task_completion_rate') is not None else '-'} | "
            f"{row.get('structured_output_valid_rate') if row.get('structured_output_valid_rate') is not None else '-'} | "
            f"{row.get('non_hold_coverage') if row.get('non_hold_coverage') is not None else '-'} | "
            f"{row.get('directional_hits') if row.get('directional_hits') is not None else '-'} / "
            f"{row.get('directional_n') if row.get('directional_n') is not None else '-'} | "
            f"{rate if rate is not None else '-'} | {alpha if alpha is not None else '-'} / "
            f"{row.get('alpha_date_bootstrap_95') if row.get('alpha_date_bootstrap_95') is not None else '-'} | "
            f"{tool_success if tool_success is not None else '-'} | {citation} | "
            f"{row.get('llm_calls') if row.get('llm_calls') is not None else '-'} | "
            f"{('/'.join(str(row.get(key)) for key in ('input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens_charged', 'total_tokens_charged_per_task')) if any(row.get(key) is not None for key in ('input_tokens', 'total_tokens_charged')) else '-')} | "
            f"{('/'.join(str(row.get(key)) for key in ('completed_tasks_per_100k_tokens', 'directional_hits_per_100k_tokens')) if row.get('completed_tasks_per_100k_tokens') is not None else '-')} | "
            f"{('/'.join(str(row.get(key)) for key in ('latency_mean_seconds', 'latency_p50_seconds', 'latency_p95_seconds', 'llm_call_latency_mean_seconds', 'llm_call_latency_p50_seconds', 'llm_call_latency_p95_seconds')) if row.get('latency_mean_seconds') is not None else '-')} | "
            f"{row.get('api_cost_cny_estimate') if row.get('api_cost_cny_estimate') is not None else '-'} |"
        )
    lines.extend([
        "",
        "B0 full Development metrics:",
        "",
        f"- Tasks: {b0_full['tasks']}; completed: {b0_full['completed_tasks']}; failures: {b0_full['failed_tasks']}.",
        f"- Non-Hold: {b0_full['directional_n']}; coverage: {b0_full['non_hold_coverage']}; "
        f"directional hit rate: {b0_full['directional_hit_rate']}; Wilson 95%: {b0_full['hit_rate_wilson_95']}.",
        f"- Mean alpha: {b0_full['mean_alpha']}; date-cluster interval: {b0_full['alpha_date_bootstrap_95']}.",
        f"- B0 paired-scope subset contains {b0_scope['tasks']} tasks.",
        "",
        "## Cost and execution limits",
        "",
        f"The conservative standard-list-price ceiling for this scope is "
        f"{manifest['cost_ceiling']['list_price_cost_ceiling_cny']} CNY. It assumes every request reaches "
        f"the {manifest['cost_ceiling']['max_prompt_bytes_per_attempt']}-byte prompt cap plus "
        f"{manifest['cost_ceiling']['protocol_token_allowance_per_attempt']} protocol-token allowance, each request "
        f"uses the full {MAX_TOKENS}-token output cap, and reasoning is conservatively allowed another "
        f"{MAX_TOKENS} tokens. The worst-case token reservation is "
        f"{manifest['cost_ceiling']['max_conservative_tokens']}; this is a ceiling, not observed usage.",
        "No observed API spend is available until a real run completes. The ceiling uses the GLM-5.3 standard "
        "list-price snapshot; account discounts, cache treatment and provider billing may differ.",
        "",
        "## Interpretation limits",
        "",
        ("The smoke pair uses two Development tasks (one AAPL and one MSFT) on one decision date. It is an "
         "integration check, not a performance estimate; no meaningful confidence interval can be computed from "
         "one date cluster. " if manifest["scope"] == "smoke" else
         "The Development sample has 20 tasks across two tickers and 10 decision dates. Its intervals remain "
         "descriptive and do not establish out-of-sample performance. ")
        + "Directional accuracy uses the non-Hold denominator; failures are reported separately. "
        "Alpha is per-task price-return excess over SPY, without fees, dividends, borrow or portfolio execution. "
        "The underlying Yahoo snapshot and A0.5 issuer evidence are historical proxies, not certified point-in-time data.",
        "",
        "This is not an official TradingAgents benchmark and B0, B1-M and B1-E are Lite internal comparisons.",
        "",
    ])
    lines.extend(["Failure status counts by group:", ""])
    for row in rows:
        if row.get("ticker") == "ALL":
            lines.append(f"- {row['variant']}: {json.dumps(row.get('failure_status_counts') or {}, ensure_ascii=False, sort_keys=True)}")
    lines.append("")
    if blocked_reason:
        lines.extend(["## Blocker", "", blocked_reason, "",
                      "No real model request was sent. No B1 accuracy, token usage, latency or cost is claimed.", ""])
    if paired is not None:
        lines.extend([
            "## Paired B1-E minus B1-M",
            "",
            f"- Scored pairs: {paired['paired_n']}; mean alpha difference: "
            f"{paired['mean_alpha_delta_e_minus_m']}; date-cluster 95% interval: "
            f"{paired['alpha_delta_date_bootstrap_95']}.",
            f"- Action changed on {paired['action_changed_n']} paired tasks.",
            f"- Among {paired['directional_transition_n']} tasks where both groups chose a direction: "
            f"wrong-to-right={paired['wrong_to_right']}; right-to-wrong={paired['right_to_wrong']}.",
            f"- Per ticker: {json.dumps(paired['by_ticker'], ensure_ascii=False, sort_keys=True)}",
            "",
        ])
    return "\n".join(lines)


def _ledger_totals(path: Path) -> dict:
    if not path.is_file():
        raise ValueError("The existing project token ledger is missing; cannot merge B1 budgets")
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            global_used = connection.execute(
                "SELECT COALESCE(SUM(charged_tokens), 0) FROM token_attempts"
            ).fetchone()[0]
            phase_rows = connection.execute(
                "SELECT phase, COALESCE(SUM(charged_tokens), 0) FROM token_attempts GROUP BY phase"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise ValueError("The project token ledger cannot be read") from exc
    by_phase = {phase: int(used or 0) for phase, used in phase_rows}
    global_used = int(global_used or 0)
    phase_used = by_phase.get("B1", 0)
    phase_limit = PHASE_TOKEN_LIMITS["B1"]
    return {
        "ledger_path_sha256": _ledger_path_hash(path),
        "global_limit": GLOBAL_TOKEN_LIMIT,
        "global_used": global_used,
        "global_remaining": max(0, GLOBAL_TOKEN_LIMIT - global_used),
        "b1_limit": phase_limit,
        "b1_used": phase_used,
        "b1_remaining": max(0, phase_limit - phase_used),
    }


def _combine_token_budget_sources(*sources: dict) -> dict:
    experiment_ids = [source.get("experiment_id") for source in sources]
    attempt_ids = [attempt_id for source in sources for attempt_id in source.get("attempt_ids", [])]
    if (any(not value for value in experiment_ids) or len(experiment_ids) != len(set(experiment_ids)) or
            len(attempt_ids) != len(set(attempt_ids))):
        raise ValueError("B1 source budget records overlap or lack experiment identity")
    return {
        "experiment_ids": experiment_ids,
        "attempt_count": sum(source.get("attempt_count", 0) for source in sources),
        "http_attempt_count": sum(source.get("http_attempt_count", 0) for source in sources),
        "failed_attempt_count": sum(source.get("failed_attempt_count", 0) for source in sources),
        "conservatively_charged_tokens": sum(
            source.get("conservatively_charged_tokens", 0) for source in sources),
        "attempt_ids": attempt_ids,
        "counted_once": True,
    }


def _write_paired_csv(path: Path, paired: dict) -> None:
    fields = ["task_id", "symbol", "decision_date", "alpha_b1_m", "alpha_b1_e",
              "alpha_delta_e_minus_m", "action_b1_m", "action_b1_e"]
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired.get("tasks", []))


def merge_b1(smoke_path: Path, continuation_path: Path, tasks_path: Path, data_path: Path,
             evidence_path: Path, b0_path: Path, out: Path, seed: int = 42) -> dict:
    """Merge validated Smoke and 18-task continuation archives without recounting calls."""
    if out.exists():
        raise ValueError("B1 merged output path already exists; archives are never overwritten")
    if sha256(tasks_path) != EXPECTED_TASKS_SHA256:
        raise ValueError("Development task file hash changed; B1 uses the frozen 20-task set only")
    if sha256(data_path / "manifest.json") != EXPECTED_MARKET_MANIFEST_SHA256:
        raise ValueError("Market snapshot manifest hash differs from frozen B1 data")
    evidence_manifest_path = evidence_path / "source_manifest.json"
    evidence_file = evidence_path / "evidence.jsonl"
    evidence_manifest_hash, evidence_hash = sha256(evidence_manifest_path), sha256(evidence_file)
    if (evidence_manifest_hash != EXPECTED_EVIDENCE_MANIFEST_SHA256 or
            evidence_hash != EXPECTED_EVIDENCE_SHA256):
        raise ValueError("Financial evidence snapshot differs from the frozen A0.5 corpus")
    all_tasks = read_tasks(tasks_path)
    all_ids = [task.task_id for task in all_tasks]
    if len(all_ids) != 20 or all_ids[:2] != list(SMOKE_TASK_IDS):
        raise ValueError("The frozen Development task order is invalid")
    smoke = _validate_b1_archive(smoke_path, expected_scope="smoke",
                                 expected_ids=list(SMOKE_TASK_IDS), require_success=True,
                                 frozen_tasks_path=tasks_path)
    continuation = _validate_b1_archive(
        continuation_path, expected_scope="development", expected_ids=all_ids[2:],
        expected_compatibility=smoke["compatibility"], require_success=False,
        frozen_tasks_path=tasks_path,
    )
    if sha256(RESUME_PROTOCOL_DOCUMENT) != smoke["compatibility"]["resume_protocol_document_sha256"]:
        raise ValueError("The B1 resume protocol document changed after Smoke was frozen")
    if seed != smoke["compatibility"]["protocol"]["statistics_seed"]:
        raise ValueError("Merge statistics seed differs from the frozen B1 protocol")
    resume_source = continuation["manifest"].get("resume_source") or {}
    smoke_manifest_sha = sha256(smoke_path / "manifest.json")
    smoke_budget = smoke["runs"]["token_budget"]
    if (resume_source.get("manifest_sha256") != smoke_manifest_sha or
            resume_source.get("source_experiment_id") != smoke_budget["experiment_id"] or
            resume_source.get("reused_task_ids") != list(SMOKE_TASK_IDS) or
            resume_source.get("reused_task_instances") != 4 or
            resume_source.get("reused_http_attempts") != smoke_budget["http_attempt_count"] or
            resume_source.get("reused_charged_tokens") != smoke_budget["conservatively_charged_tokens"] or
            resume_source.get("source_attempt_ids") != smoke_budget["attempt_ids"] or
            resume_source.get("counted_again") is not False):
        raise ValueError("Continuation did not preserve a verifiable Smoke reuse audit")
    smoke_experiment = smoke_budget["experiment_id"]
    continuation_budget = continuation["runs"]["token_budget"]
    continuation_experiment = continuation_budget["experiment_id"]
    combined_budget_sources = _combine_token_budget_sources(smoke_budget, continuation_budget)

    b0_run = json.loads((b0_path / "run.json").read_text(encoding="utf-8"))
    if (b0_run.get("status") != "completed" or b0_run.get("ablation") != "B0" or
            b0_run.get("tasks_sha256") != EXPECTED_TASKS_SHA256 or
            b0_run.get("snapshot_sha256") != EXPECTED_MARKET_MANIFEST_SHA256):
        raise ValueError("B0 archive must be the original completed 20-task baseline")
    if (not (b0_path / "trajectory.jsonl").is_file() or
            (b0_path / "tasks.jsonl").read_bytes() != (b0_path / "trajectory.jsonl").read_bytes()):
        raise ValueError("B0 task and trajectory archives must be byte-identical")
    if (sha256(b0_path / "run.json") != smoke["compatibility"]["b0_run_metadata_sha256"] or
            sha256(b0_path / "trajectory.jsonl") != smoke["compatibility"]["b0_trajectory_sha256"]):
        raise ValueError("The archived B0 reference differs from the B0 frozen before Smoke")
    b0_rows = _load_jsonl(b0_path / "tasks.jsonl")
    if len(b0_rows) != 20 or any(row.get("status") != "ok" for row in b0_rows):
        raise ValueError("B0 archive must contain 20 successful frozen-task trajectories")
    b0_full = summary(b0_rows, seed, b0_run["config"])

    merged = {}
    model_configs = []
    for variant in ("B1-M", "B1-E"):
        smoke_run = smoke["runs"][variant]
        continuation_run = continuation["runs"][variant]
        if smoke_run["run"]["config"] != continuation_run["run"]["config"]:
            raise ValueError(f"{variant} model settings changed between Smoke and continuation")
        model_configs.append(smoke_run["run"]["config"])
        merged[variant] = _merge_rows(smoke_run["rows"], continuation_run["rows"], all_ids)
    if model_configs[0] != model_configs[1]:
        raise ValueError("B1-M and B1-E model settings differ")

    m_stats = summary(merged["B1-M"], seed, model_configs[0])
    e_stats = summary(merged["B1-E"], seed, model_configs[1])
    for stats in (m_stats, e_stats):
        stats.update(
            requested_tasks=20,
            run_status="completed" if stats["completed_tasks"] == 20 and stats["failed_tasks"] == 0 else "incomplete",
            budget_exhausted=False,
        )
    paired = _paired(merged["B1-M"], merged["B1-E"], seed)
    completion_check = _completion_check(20, m_stats, e_stats, paired["paired_n"])
    actual_models = sorted({model for variant in ("B1-M", "B1-E")
                            for model in smoke["runs"][variant]["actual_models"] +
                            continuation["runs"][variant]["actual_models"]})
    model_mismatch = not actual_models or any(model != DEFAULT_MODEL for model in actual_models)
    final_status = ("model_identity_unconfirmed" if model_mismatch else
                    "completed" if completion_check["complete"] else "incomplete")

    ledger_path = _ledger_path()
    budget_snapshot = _ledger_totals(ledger_path)
    budget_snapshot.update({
        "smoke_experiment_id": smoke_experiment,
        "continuation_experiment_id": continuation_experiment,
        "merged_source_attempts": combined_budget_sources["attempt_count"],
        "merged_source_http_attempts": combined_budget_sources["http_attempt_count"],
        "merged_source_charged_tokens_counted_once": combined_budget_sources["conservatively_charged_tokens"],
        "merged_budget_attempt_ids": combined_budget_sources["attempt_ids"],
        "source_attempt_ids_disjoint": True,
        "merge_wrote_ledger": False,
    })

    b1_rows = []
    m_group_status = "completed" if m_stats["completed_tasks"] == 20 else "incomplete"
    e_group_status = "completed" if e_stats["completed_tasks"] == 20 else "incomplete"
    b1_rows.extend(_summary_rows("B0", "development-20", b0_full))
    b1_rows.extend(_summary_rows("B1-M", "development-20", m_stats, m_group_status))
    b1_rows.extend(_summary_rows("B1-E", "development-20", e_stats, e_group_status))
    full_cost_ceiling = request_cost_ceiling(20)
    manifest = {
        "experiment": "B1 Real LLM Development Baseline",
        "scope": "development-20",
        "status": final_status,
        "merged_archive": True,
        "created_at": datetime.now(UTC).isoformat(),
        "code_git_sha": _git("rev-parse", "HEAD"),
        "full_tasks_sha256": EXPECTED_TASKS_SHA256,
        "selected_task_ids": all_ids,
        "selected_tasks_sha256": sha256(tasks_path),
        "market_manifest_sha256": EXPECTED_MARKET_MANIFEST_SHA256,
        "evidence_manifest_sha256": EXPECTED_EVIDENCE_MANIFEST_SHA256,
        "evidence_sha256": EXPECTED_EVIDENCE_SHA256,
        "prompt_hashes": smoke["manifest"]["prompt_hashes"],
        "tool_schema_hashes": smoke["manifest"]["tool_schema_hashes"],
        "protocol": smoke["manifest"]["protocol"],
        "resume_compatibility": smoke["compatibility"],
        "resume_compatibility_sha256": smoke["manifest"]["resume_compatibility_sha256"],
        "source_archives": {
            "smoke": {"path": str(smoke_path), "manifest_sha256": smoke_manifest_sha,
                      "trajectory_sha256": {v: sha256(smoke_path / v / "trajectory.jsonl")
                                             for v in ("B1-M", "B1-E")}},
            "continuation": {"path": str(continuation_path),
                             "manifest_sha256": sha256(continuation_path / "manifest.json"),
                             "trajectory_sha256": {v: sha256(continuation_path / v / "trajectory.jsonl")
                                                    for v in ("B1-M", "B1-E")}},
        },
        "resume_audit": resume_source,
        "cost_ceiling": full_cost_ceiling,
        "b0_reference": str(b0_path),
        "b0_summary": b0_full,
        "b1_m_summary": m_stats,
        "b1_e_summary": e_stats,
        "paired_comparison": paired,
        "completion_check": completion_check,
        "actual_models_observed": actual_models,
        "token_budget": budget_snapshot,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    if model_mismatch:
        manifest["blocker"] = "actual_model_identity_mismatch"
    elif not completion_check["complete"]:
        manifest["blocker"] = "one_or_more_tasks_failed_or_unpaired"

    out.mkdir(parents=True, exist_ok=False)
    (out / "tasks.jsonl").write_bytes(_selected_task_bytes(tasks_path, all_ids))
    for variant in ("B1-M", "B1-E"):
        run_dir = out / variant
        run_dir.mkdir()
        trajectory = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged[variant])
        (run_dir / "tasks.jsonl").write_text(trajectory, encoding="utf-8")
        (run_dir / "trajectory.jsonl").write_text(trajectory, encoding="utf-8")
        run_meta = {
            "status": "completed" if (m_group_status if variant == "B1-M" else e_group_status) == "completed" else "incomplete",
            "ablation": variant, "merged_archive": True, "requested_tasks": 20,
            "processed_tasks": len(merged[variant]), "config": model_configs[0 if variant == "B1-M" else 1],
            "actual_models_observed": sorted({call.get("actual_model") for row in merged[variant]
                                               for call in row.get("llm_calls", []) if call.get("actual_model")}),
            "source_experiment_ids": [smoke_experiment, continuation_experiment],
            "token_budget_counted_once": True,
        }
        _write_json(run_dir / "run.json", run_meta)
        _write_summary_csv(run_dir / "summary.csv", _summary_rows(
            variant, "development-20", m_stats if variant == "B1-M" else e_stats,
            m_group_status if variant == "B1-M" else e_group_status,
        ))
    _write_summary_csv(out / "summary.csv", b1_rows)
    _write_json(out / "paired_summary.json", paired)
    _write_paired_csv(out / "paired.csv", paired)
    _write_json(out / "token-budget.json", budget_snapshot)
    (out / "REPORT.md").write_text(
        _report(manifest, b1_rows, paired, b0_full, b0_full), encoding="utf-8")
    _write_json(out / "manifest.json", manifest)
    return {"status": final_status, "output": str(out), "selected_task_ids": all_ids,
            "paired_tasks": paired["paired_n"], "token_budget": budget_snapshot}


def run_b1(scope: str, tasks_path: Path, data_path: Path, evidence_path: Path,
           b0_path: Path, out: Path, seed: int = 42, budget_cap_cny: float | None = None,
           resume_from: Path | None = None) -> dict:
    if scope not in {"smoke", "development"}:
        raise ValueError("B1 scope must be smoke or development")
    if scope == "smoke" and resume_from is not None:
        raise ValueError("Smoke cannot resume from another archive")
    if scope == "development" and resume_from is None:
        raise ValueError("Development must continue from a validated successful Smoke archive")
    if sha256(tasks_path) != EXPECTED_TASKS_SHA256:
        raise ValueError("Development task file hash changed; B1 uses the frozen 20-task set only")
    all_tasks = read_tasks(tasks_path)
    if len(all_tasks) != 20:
        raise ValueError("The frozen Development task set must contain 20 tasks")
    market_hash = sha256(data_path / "manifest.json")
    if market_hash != EXPECTED_MARKET_MANIFEST_SHA256:
        raise ValueError("Market snapshot manifest hash differs from the frozen A0.5 snapshot")
    evidence_manifest_path = evidence_path / "source_manifest.json"
    evidence_file = evidence_path / "evidence.jsonl"
    evidence_manifest_hash, evidence_hash = sha256(evidence_manifest_path), sha256(evidence_file)
    if (evidence_manifest_hash != EXPECTED_EVIDENCE_MANIFEST_SHA256 or
            evidence_hash != EXPECTED_EVIDENCE_SHA256):
        raise ValueError("Financial evidence snapshot differs from the frozen A0.5 corpus")
    evidence = EvidenceStore.load(evidence_path)
    if json.loads(evidence_manifest_path.read_text())["tasks_sha256"] != EXPECTED_TASKS_SHA256:
        raise ValueError("A0.5 evidence was not collected for this frozen Development set")

    selected = all_tasks[:2] if scope == "smoke" else all_tasks[2:]
    if scope == "smoke" and tuple(task.task_id for task in selected) != SMOKE_TASK_IDS:
        raise ValueError("Smoke task selection must be the frozen dev-01/dev-02 pair")
    selected_ids = [task.task_id for task in selected]
    selected_bytes = _selected_task_bytes(tasks_path, selected_ids)
    cost_ceiling = request_cost_ceiling(len(selected))
    if scope == "development":
        if budget_cap_cny is None:
            raise ValueError(
                f"Development run requires --budget-cap-cny >= {cost_ceiling['list_price_cost_ceiling_cny']}"
            )
        if budget_cap_cny < cost_ceiling["list_price_cost_ceiling_cny"]:
            raise ValueError("Budget cap is below the conservative protocol cost ceiling")
    elif budget_cap_cny is not None and budget_cap_cny < cost_ceiling["list_price_cost_ceiling_cny"]:
        raise ValueError("Budget cap is below the conservative smoke cost ceiling")

    b0_run_path = b0_path / "run.json"
    b0_tasks_path = b0_path / "tasks.jsonl"
    b0_trajectory_path = b0_path / "trajectory.jsonl"
    b0_run = json.loads(b0_run_path.read_text(encoding="utf-8"))
    if (b0_run.get("status") != "completed" or b0_run.get("ablation") != "B0" or
            b0_run.get("tasks_sha256") != EXPECTED_TASKS_SHA256 or
            b0_run.get("snapshot_sha256") != EXPECTED_MARKET_MANIFEST_SHA256):
        raise ValueError("B0 archive must be a completed 20-task run on the frozen Development snapshot")
    if (not b0_trajectory_path.is_file() or b0_tasks_path.read_bytes() != b0_trajectory_path.read_bytes()):
        raise ValueError("B0 task and trajectory archives must be byte-identical")
    b0_rows = _load_jsonl(b0_tasks_path)
    if len(b0_rows) != 20 or any(row.get("status") != "ok" for row in b0_rows):
        raise ValueError("B0 archive must contain 20 successfully completed Development trajectories")
    b0_run_hash = sha256(b0_run_path)
    b0_trajectory_hash = sha256(b0_trajectory_path)
    b0_full = summary(b0_rows, seed, b0_run["config"])
    b0_scope_rows = [row for row in b0_rows if row["task_id"] in selected_ids]
    b0_scope = summary(b0_scope_rows, seed, b0_run["config"])

    model = os.getenv("TALITE_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("TALITE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if model != DEFAULT_MODEL or base_url != DEFAULT_BASE_URL:
        raise ValueError("Frozen B1 protocol requires TALITE_MODEL=glm-5.3 and the official OpenAI-compatible GLM endpoint")
    compatibility = _resume_compatibility(
        seed=seed, market_hash=market_hash,
        evidence_manifest_hash=evidence_manifest_hash, evidence_hash=evidence_hash,
        b0_run_hash=b0_run_hash, b0_trajectory_hash=b0_trajectory_hash,
    )
    resume_source = None
    if resume_from is not None:
        if len(selected_ids) != 18 or selected_ids[0] in SMOKE_TASK_IDS:
            raise ValueError("B1 continuation must contain exactly the 18 tasks after dev-01/dev-02")
        resume_source = _validate_b1_archive(
            resume_from, expected_scope="smoke", expected_ids=list(SMOKE_TASK_IDS),
            expected_compatibility=compatibility, require_success=True, frozen_tasks_path=tasks_path,
        )

    code_worktree_dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    experiment_id = str(out.resolve())
    ledger_path = _ledger_path()
    api_key_available = bool(os.getenv("TALITE_API_KEY"))
    token_budget = _open_b1_budget(ledger_path, scope, api_key_available)
    token_budget_preflight = (token_budget.can_reserve(
        cost_ceiling["max_conservative_tokens"], experiment_id=experiment_id, phase="B1")
        if token_budget is not None else None)
    out.mkdir(parents=True, exist_ok=False)
    (out / "tasks.jsonl").write_bytes(selected_bytes)

    manifest = {
        "experiment": "B1 Real LLM Development Baseline",
        "scope": scope,
        "status": "preflight",
        "started_at": datetime.now(UTC).isoformat(),
        "code_git_sha": _git("rev-parse", "HEAD"),
        "code_worktree_dirty": code_worktree_dirty,
        "full_tasks_sha256": EXPECTED_TASKS_SHA256,
        "selected_tasks_sha256": sha256(out / "tasks.jsonl"),
        "selected_task_ids": selected_ids,
        "market_manifest_sha256": market_hash,
        "evidence_manifest_sha256": evidence_manifest_hash,
        "evidence_sha256": evidence_hash,
        "prompt_hashes": prompt_hashes(),
        "tool_schema_hashes": compatibility["tool_schema_hashes"],
        "protocol": compatibility["protocol"],
        "resume_compatibility": compatibility,
        "resume_compatibility_sha256": _json_hash(compatibility),
        "token_budget_ledger_path_sha256": _ledger_path_hash(ledger_path),
        "cost_ceiling": cost_ceiling,
        "token_budget_preflight": token_budget_preflight,
        "token_budget": (token_budget.snapshot(experiment_id=experiment_id, phase="B1")
                         if token_budget is not None else {
                             "status": "unavailable_without_api_key",
                             "existing_ledger_present": ledger_path.is_file(),
                             "experiment_id": experiment_id,
                         }),
        "budget_cap_cny": budget_cap_cny,
        "b0_reference": str(b0_path),
    }
    if resume_source is not None:
        smoke_manifest = resume_source["manifest"]
        smoke_rows = resume_source["runs"]["B1-M"]["rows"] + resume_source["runs"]["B1-E"]["rows"]
        manifest["resume_source"] = {
            "path": str(resume_from),
            "manifest_sha256": sha256(resume_from / "manifest.json"),
            "reused_task_ids": list(SMOKE_TASK_IDS),
            "reused_task_instances": len(smoke_rows),
            "reused_http_attempts": resume_source["runs"]["token_budget"]["http_attempt_count"],
            "reused_charged_tokens": resume_source["runs"]["token_budget"]["conservatively_charged_tokens"],
            "source_experiment_id": smoke_manifest["token_budget"]["experiment_id"],
            "source_attempt_ids": resume_source["runs"]["token_budget"]["attempt_ids"],
            "counted_again": False,
        }
    manifest_path = out / "manifest.json"
    _write_json(manifest_path, manifest)

    summary_rows = []
    summary_rows.extend(_summary_rows("B0", "development-20", b0_full))
    if scope == "smoke":
        summary_rows.extend(_summary_rows("B0", "paired-smoke-2", b0_scope))
    blocked_reason = None

    if not os.getenv("TALITE_API_KEY"):
        blocked_reason = (
            "TALITE_API_KEY is absent in the current WSL environment. Set it in the shell without printing "
            "or committing it, then rerun with a new --output directory. No API request was sent. "
            + ("The existing project token ledger is also absent, so restore its prior contents before any real run."
               if not ledger_path.is_file() else "")
        )
        manifest.update(status="blocked_missing_api_key", blocker="missing_api_key",
                        finished_at=datetime.now(UTC).isoformat())
        requested_tickers = sorted({task.symbol for task in selected})
        for variant in ("B1-M", "B1-E"):
            for ticker in [*requested_tickers, "ALL"]:
                summary_rows.append({"variant": variant, "scope": scope, "ticker": ticker,
                    "status": "not_run_missing_api_key", "requested_tasks": (len(selected) if ticker == "ALL" else sum(task.symbol == ticker for task in selected)),
                    "tasks": 0, "completed_tasks": None, "directional_n": None,
                    "directional_hit_rate": None, "mean_alpha": None,
                    "evidence_citation_valid_rate": None, "llm_calls": None,
                    "input_tokens": None, "output_tokens": None, "api_cost_cny_estimate": None})
        paired = None
        manifest["b1_m_and_b1_e"] = {"status": "not_run", "reason": "missing_api_key"}
    elif not token_budget_preflight["fits"]:
        blocked_reason = (
            f"The conservative B1 reservation needs {cost_ceiling['max_conservative_tokens']} tokens, "
            f"but only {token_budget_preflight['available']} remain across the project, B1 phase and smoke limits. "
            "No model request was sent."
        )
        manifest.update(status="blocked_token_budget", blocker="insufficient_token_budget",
                        finished_at=datetime.now(UTC).isoformat(),
                        token_budget=token_budget.snapshot(experiment_id=experiment_id, phase="B1"))
        requested_tickers = sorted({task.symbol for task in selected})
        for variant in ("B1-M", "B1-E"):
            for ticker in [*requested_tickers, "ALL"]:
                summary_rows.append({"variant": variant, "scope": scope, "ticker": ticker,
                    "status": "not_run_token_budget",
                    "requested_tasks": (len(selected) if ticker == "ALL" else sum(task.symbol == ticker for task in selected)),
                    "tasks": 0, "completed_tasks": 0, "failed_tasks": 0,
                    "directional_n": 0, "llm_calls": 0, "input_tokens": 0,
                    "output_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0,
                    "separate_reasoning_tokens": 0,
                    "total_tokens_charged": 0, "missing_usage_calls": 0,
                    "reasoning_detail_missing_calls": 0, "cached_detail_missing_calls": 0,
                    "api_cost_cny_estimate": None})
        paired = None
        manifest["b1_m_and_b1_e"] = {"status": "not_run", "reason": "insufficient_token_budget"}
    else:
        client, configured_model, api_host = client_from_env(default_model=DEFAULT_MODEL)
        if configured_model != DEFAULT_MODEL or api_host != "open.bigmodel.cn":
            client.close()
            raise ValueError("Resolved GLM endpoint/model differs from the frozen B1 protocol")
        budget = Budget(max_tool_rounds=MAX_TOOL_ROUNDS, max_tokens=MAX_TOKENS)
        model_config = {
            "backend": "llm", "model": configured_model, "api_host": api_host,
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "thinking": "enabled", "reasoning_effort": REASONING_EFFORT,
            "provider_seed": None, "phase": "B1", "sdk_retries": 0,
            "application_retries": MAX_RETRIES,
            "max_prompt_bytes": MAX_PROMPT_BYTES,
            "pricing_cny_per_million": PRICE_CNY_PER_MILLION,
            "pricing_source": PRICE_SOURCE, "pricing_as_of": PRICE_AS_OF,
        }
        protocol_metadata = {"prompt_hashes": prompt_hashes(), "max_model_retries": MAX_RETRIES,
            "max_prompt_bytes": MAX_PROMPT_BYTES, "model_timeout_seconds": 25,
            "provider_generation_seed": None, "holdout_accessed": False,
            "tool_schema_hashes": compatibility["tool_schema_hashes"],
            "b1_compatibility_sha256": manifest["resume_compatibility_sha256"],
            "resume_protocol_document_sha256": compatibility["resume_protocol_document_sha256"]}
        manifest.update(status="running", api_host=api_host, model_requested=configured_model)
        _write_json(manifest_path, manifest)
        m_rows, e_rows = [], []
        empty_stats = {"tasks": 0, "requested_tasks": len(selected), "completed_tasks": 0,
                       "failed_tasks": 0, "run_status": "not_run", "budget_exhausted": False}
        m_stats = e_stats = empty_stats
        try:
            for variant, financial_path in (("B1-M", None), ("B1-E", evidence_path)):
                factory = lambda: ModelBackend(
                    client, configured_model, budget, None, max_retries=MAX_RETRIES,
                    max_prompt_bytes=MAX_PROMPT_BYTES, temperature=TEMPERATURE,
                    reasoning_effort=REASONING_EFFORT, token_budget=token_budget, phase="B1",
                )
                stats = run(out / "tasks.jsonl", data_path, out / variant, factory, model_config,
                    budget, seed, True, evidence_path=financial_path, run_label=variant,
                    protocol_metadata=protocol_metadata, token_budget=token_budget,
                    experiment_id=experiment_id, phase="B1")
                rows_for_variant = _load_jsonl(out / variant / "tasks.jsonl")
                group_status = (
                    stats["run_status"] if stats["budget_exhausted"] else
                    "completed" if stats["tasks"] == len(selected) and stats["failed_tasks"] == 0 else
                    "incomplete"
                )
                summary_rows.extend(_summary_rows(variant, scope, stats, status=group_status))
                if variant == "B1-M":
                    m_rows = rows_for_variant
                    m_stats = stats
                else:
                    e_rows = rows_for_variant
                    e_stats = stats
                if stats["budget_exhausted"]:
                    break
            paired = _paired(m_rows, e_rows, seed)
            actual = sorted({call.get("actual_model") for rows in (m_rows, e_rows) for row in rows
                             for call in row.get("llm_calls", []) if call.get("actual_model")})
            mismatch = not actual or any(name != DEFAULT_MODEL for name in actual)
            completion_check = _completion_check(len(selected), m_stats, e_stats, paired["paired_n"])
            any_budget_exhausted = m_stats.get("budget_exhausted", False) or e_stats.get("budget_exhausted", False)
            final_status = (
                "budget_exhausted" if any_budget_exhausted else
                "model_identity_unconfirmed" if mismatch else
                "completed" if completion_check["complete"] else
                "incomplete"
            )
            manifest.update(
                status=final_status,
                actual_models_observed=actual,
                b1_m_summary=m_stats,
                b1_e_summary=e_stats,
                paired_comparison=paired,
                completion_check=completion_check,
                token_budget=token_budget.snapshot(experiment_id=experiment_id, phase="B1"),
                trajectory_sha256={variant: sha256(out / variant / "trajectory.jsonl")
                                   for variant in ("B1-M", "B1-E")
                                   if (out / variant / "trajectory.jsonl").is_file()},
                finished_at=datetime.now(UTC).isoformat(),
            )
            if any_budget_exhausted:
                blocked_reason = "The B1 token reservation limit was reached; the completed task records were archived."
            elif mismatch and actual:
                blocked_reason = "The API returned a model identity different from the frozen glm-5.3 request."
            elif mismatch:
                blocked_reason = "No actual model identity was observed, so the frozen glm-5.3 identity is unconfirmed."
            elif not completion_check["complete"]:
                blocked_reason = (
                    "B1 is incomplete: both groups must process and successfully complete every requested task, "
                    "with zero failed tasks and one valid paired result per task."
                )
            if blocked_reason:
                manifest["blocker"] = final_status
                manifest["blocker_reason"] = blocked_reason
            _write_json(out / "paired_summary.json", paired)
        finally:
            client.close()

    _write_summary_csv(out / "summary.csv", summary_rows)
    (out / "REPORT.md").write_text(
        _report(manifest, summary_rows, paired, b0_full, b0_scope, blocked_reason), encoding="utf-8"
    )
    _write_json(manifest_path, manifest)
    return {"status": manifest["status"], "output": str(out),
            "selected_task_ids": selected_ids, "cost_ceiling_cny": cost_ceiling["list_price_cost_ceiling_cny"],
            "b0_completed": b0_full["completed_tasks"], "b0_tasks": b0_full["tasks"],
            "blocker": manifest.get("blocker")}
