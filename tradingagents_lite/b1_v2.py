"""Frozen, pair-by-pair GLM-5.3-Flash B1 protocol, isolated from legacy B1."""
from datetime import UTC, datetime
import csv
from io import StringIO
import importlib.metadata
import json
import os
from pathlib import Path

from . import b1
from .data import read_tasks, sha256
from .evidence import EvidenceStore
from .prompts import prompt_hashes_v2
from .provider import DEFAULT_BASE_URL, DEFAULT_TEMPERATURE, ModelBackend, client_from_env
from .runner import run, summary
from .schema import Budget
from .token_budget import PROTOCOL_TOKEN_ALLOWANCE, SMOKE_TOKEN_LIMIT, TokenBudget
from .workflow import specs
from .tool_registry import FINANCIAL_SPEC, SPECS


MAX_TOOL_ROUNDS = 2
MAX_TOOL_CALLS_PER_ROUND = b1.B1_V2_MAX_TOOL_CALLS_PER_ROUND
MAX_TOKENS = 512
MAX_PROMPT_BYTES = b1.B1_V2_MAX_PROMPT_BYTES
MAX_RETRIES = 0
REASONING_EFFORT = "low"
THINKING = "enabled"  # Flash Thinking is always enabled by the official API.
SETTLEMENT_PROTOCOL = b1.SETTLEMENT_PROTOCOL
PAIR_SAFETY_MULTIPLIER = 2


def _model_config() -> dict:
    return {
        "backend": "llm", "protocol_version": b1.B1_V2_PROTOCOL_VERSION,
        "model": b1.B1_V2_MODEL, "api_host": "open.bigmodel.cn",
        "temperature": DEFAULT_TEMPERATURE, "max_tokens": MAX_TOKENS,
        "thinking": THINKING, "reasoning_effort": REASONING_EFFORT,
        "provider_seed": None, "phase": "B1", "sdk_retries": 0,
        "application_retries": MAX_RETRIES, "max_prompt_bytes": MAX_PROMPT_BYTES,
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "max_tool_calls_per_round": MAX_TOOL_CALLS_PER_ROUND,
        "compact_tool_results": True,
        "pricing_cny_per_million": b1.B1_V2_PRICE_CNY_PER_MILLION,
        "pricing_source": b1.B1_V2_PRICE_SOURCE, "pricing_as_of": b1.B1_V2_PRICE_AS_OF,
    }


def _tool_schema_hashes() -> dict[str, str]:
    market = specs(SPECS)
    evidence = specs({**SPECS, "financial_evidence": FINANCIAL_SPEC})
    return {"B1-M": b1._json_hash(market), "B1-E": b1._json_hash(evidence)}


def _protocol(seed: int, *, market_hash: str, evidence_manifest_hash: str,
              evidence_hash: str, b0_run_hash: str, b0_trajectory_hash: str) -> dict:
    tracked_changes = b1._git("status", "--porcelain", "--untracked-files=no")
    if tracked_changes:
        raise ValueError("Commit and freeze B1 V2 code and protocol before any real Flash call")
    return {
        "version": b1.B1_V2_PROTOCOL_VERSION,
        "code_git_sha": b1._git("rev-parse", "HEAD"),
        "tracked_worktree_clean": True,
        "lock_sha256": sha256(Path("uv.lock")),
        "protocol_document_sha256": sha256(Path("docs/b1-v2-protocol.md")),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("langgraph", "pydantic", "openai")},
        "model": b1.B1_V2_MODEL,
        "base_url": DEFAULT_BASE_URL,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
        "thinking": THINKING,
        "max_prompt_bytes": MAX_PROMPT_BYTES,
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "max_tool_calls_per_round": MAX_TOOL_CALLS_PER_ROUND,
        "sdk_retries": 0,
        "application_retries": MAX_RETRIES,
        "provider_generation_seed": None,
        "statistics_seed": seed,
        "reasoning_tokens_reserved_separately": MAX_TOKENS,
        "task_file_sha256": b1.EXPECTED_TASKS_SHA256,
        "market_manifest_sha256": market_hash,
        "evidence_manifest_sha256": evidence_manifest_hash,
        "evidence_sha256": evidence_hash,
        "b0_run_metadata_sha256": b0_run_hash,
        "b0_trajectory_sha256": b0_trajectory_hash,
        "prompt_hashes": prompt_hashes_v2(),
        "tool_schema_hashes": _tool_schema_hashes(),
        "settlement": SETTLEMENT_PROTOCOL,
        "holdout_accessed": False,
        "rag": False,
        "memory": False,
        "debate": False,
        "mcp": False,
        "verifier_extension": False,
        "automatic_fallback": False,
        "max_calls_per_task": MAX_TOOL_ROUNDS + 2,
        "pricing_cny_per_million": b1.B1_V2_PRICE_CNY_PER_MILLION,
        "pricing_source": b1.B1_V2_PRICE_SOURCE,
        "pricing_as_of": b1.B1_V2_PRICE_AS_OF,
        "model_config": _model_config(),
        "budget": Budget(max_tool_rounds=MAX_TOOL_ROUNDS, max_tokens=MAX_TOKENS).model_dump(),
        "react": True,
        "compact_tool_results": True,
    }


def _assert_flash_rows(rows: list[dict], *, require_calls: bool = True) -> list[str]:
    observed = sorted({call.get("actual_model") for row in rows for call in row.get("llm_calls", [])
                       if call.get("actual_model")})
    if any(model != b1.B1_V2_MODEL for model in observed):
        raise ValueError("Cannot reuse or merge a trajectory from a different model")
    if require_calls and not observed:
        raise ValueError("Flash model identity is absent from the archived trajectories")
    for row in rows:
        if require_calls and row.get("status") == "ok" and not any(
                call.get("actual_model") == b1.B1_V2_MODEL for call in row.get("llm_calls", [])):
            raise ValueError("A successful Flash task is missing its model identity record")
        for call in row.get("llm_calls", []):
            if call.get("status") == "ok" and call.get("actual_model") != b1.B1_V2_MODEL:
                raise ValueError("A successful call does not have the frozen Flash model identity")
    return observed


def _cell_quality_issues(row: dict) -> list[str]:
    """Fail a Flash smoke cell if its actual tool/JSON/settlement path was not healthy."""
    issues = []
    if row.get("status") != "ok":
        issues.append("task_not_completed")
    structured = row.get("structured_output_trace", [])
    if len(structured) != 2 or any(item.get("valid") is not True for item in structured):
        issues.append("structured_output_invalid")
    tools = row.get("tool_trace", [])
    if not any(item.get("dispatch") == "react" and item.get("status") == "ok"
               and item.get("name") in SPECS for item in tools):
        issues.append("no_successful_function_call")
    if any(item.get("dispatch") == "react" and item.get("status") == "error" for item in tools):
        issues.append("tool_execution_error")
    if any(event.get("invalid_ids") for event in row.get("citation_trace", [])):
        issues.append("invalid_evidence_reference")
    usage = row.get("usage") or {}
    if usage.get("retries", 0):
        issues.append("unexpected_retry")
    if any(call.get("error_code") == "output_truncated" for call in row.get("llm_calls", [])):
        issues.append("output_truncated")
    calls = row.get("llm_calls", [])
    if len(calls) > MAX_TOOL_ROUNDS + 2 or any(
            len(call.get("tool_names", [])) > MAX_TOOL_CALLS_PER_ROUND for call in calls):
        issues.append("tool_budget_exceeded")
    if any(call.get("request_bytes", 0) > MAX_PROMPT_BYTES for call in calls):
        issues.append("prompt_byte_limit_exceeded")
    return issues


def _validated_smoke(path: Path, protocol: dict, tasks_path: Path, ledger_path: Path) -> dict:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("A completed B1 V2 Smoke archive is required before Development")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("scope") != "smoke" or manifest.get("status") != "completed" or
            manifest.get("protocol_sha256") != b1._json_hash(protocol) or
            manifest.get("protocol") != protocol or
            manifest.get("selected_task_ids") != list(b1.SMOKE_TASK_IDS) or
            manifest.get("full_tasks_sha256") != b1.EXPECTED_TASKS_SHA256 or
            manifest.get("token_budget_ledger_path_sha256") != b1._ledger_path_hash(ledger_path)):
        raise ValueError("Smoke archive is incomplete or differs from the frozen Flash V2 protocol")
    if manifest.get("model_config") != _model_config():
        raise ValueError("Smoke model, sampling, price or request configuration differs")
    if sha256(path / "tasks.jsonl") != manifest.get("selected_tasks_sha256"):
        raise ValueError("Flash Smoke task file hash does not match its manifest")
    if (b1._selected_task_bytes(tasks_path, list(b1.SMOKE_TASK_IDS)) != (path / "tasks.jsonl").read_bytes()):
        raise ValueError("Flash Smoke task inputs differ from the frozen Development task set")
    rows_by_variant = {}
    for variant in ("B1-M", "B1-E"):
        trajectory_path = path / variant / "trajectory.jsonl"
        task_copy = path / variant / "tasks.jsonl"
        if not trajectory_path.is_file() or task_copy.read_bytes() != trajectory_path.read_bytes():
            raise ValueError(f"{variant} Smoke trajectory is missing or changed")
        if manifest.get("trajectory_sha256", {}).get(variant) != sha256(trajectory_path):
            raise ValueError(f"{variant} Smoke trajectory hash is invalid")
        rows = b1._load_jsonl(trajectory_path)
        if ([row.get("task_id") for row in rows] != list(b1.SMOKE_TASK_IDS) or
                any(row.get("status") != "ok" or row.get("alpha") is None for row in rows)):
            raise ValueError("Smoke must have four successful, paired task instances")
        _assert_flash_rows(rows)
        run_meta_path = path / variant / "run.json"
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        expected_actual = [b1.B1_V2_MODEL]
        if (run_meta.get("status") != "completed" or run_meta.get("ablation") != variant or
                run_meta.get("requested_tasks") != 2 or run_meta.get("processed_tasks") != 2 or
                run_meta.get("tasks_sha256") != manifest.get("selected_tasks_sha256") or
                run_meta.get("config") != _model_config() or
                run_meta.get("actual_models_observed") != expected_actual or
                run_meta.get("snapshot_sha256") != protocol["market_manifest_sha256"] or
                run_meta.get("evidence_manifest_sha256") != protocol["evidence_manifest_sha256"] or
                run_meta.get("evidence_sha256") != protocol["evidence_sha256"] or
                run_meta.get("prompt_hashes") != protocol["prompt_hashes"] or
                run_meta.get("tool_schema_hashes") != protocol["tool_schema_hashes"] or
                run_meta.get("budget") != protocol["budget"] or run_meta.get("react") is not True):
            raise ValueError(f"{variant} run metadata does not match the frozen V2 protocol")
        if (run_meta.get("settlement") != protocol["settlement"] or
                run_meta.get("execution") != "next shared session open to horizon-th shared session close, OHLC price return" or
                run_meta.get("alpha") != "position direction × (asset price return − SPY price return); no costs"):
            raise ValueError(f"{variant} settlement metadata differs from the frozen protocol")
        if _stats(rows, protocol["statistics_seed"], _model_config(), 2) != manifest.get(
                "b1_m_summary" if variant == "B1-M" else "b1_e_summary"):
            raise ValueError(f"{variant} archived Smoke statistics do not match its trajectory")
        rows_by_variant[variant] = rows
    check = manifest.get("completion_check") or {}
    if not check.get("complete") or check.get("valid_pairs") != 2:
        raise ValueError("Smoke does not contain two valid Flash pairs")
    if not (manifest.get("quality_check") or {}).get("complete"):
        raise ValueError("Smoke quality checks did not pass")
    token_budget = manifest.get("token_budget") or {}
    attempts = b1._read_ledger_attempts(ledger_path, token_budget.get("experiment_id", ""))
    rows = rows_by_variant["B1-M"] + rows_by_variant["B1-E"]
    ledger_audit = b1._verify_ledger_records(
        rows, attempts, manifest=manifest, ledger_path=ledger_path, expected_model=b1.B1_V2_MODEL)
    return {"manifest": manifest, "rows": rows_by_variant, "ledger_audit": ledger_audit}


def _write_rows(path: Path, rows: list[dict]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    columns = fields or []
    if not columns:
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns or ["variant", "scope", "ticker", "status"])
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _write_paired_csv(path: Path, paired: dict) -> None:
    fields = ["task_id", "symbol", "decision_date", "alpha_b1_m", "alpha_b1_e",
              "alpha_delta_e_minus_m", "action_b1_m", "action_b1_e"]
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(paired.get("tasks", []))
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _stats(rows: list[dict], seed: int, config: dict, requested: int) -> dict:
    metrics = summary(rows, seed, config)
    metrics["requested_tasks"] = requested
    metrics["unattempted_tasks"] = max(0, requested - len(rows))
    metrics["task_completion_rate"] = metrics["completed_tasks"] / requested if requested else None
    metrics["run_status"] = "completed" if len(rows) == requested and metrics["failed_tasks"] == 0 else "incomplete"
    metrics["budget_exhausted"] = any(call.get("error_code") == "token_budget_exhausted"
                                       for row in rows for call in row.get("llm_calls", []))
    return metrics


def _summary_rows(variant: str, scope: str, metrics: dict, status: str = "completed") -> list[dict]:
    rows = b1._summary_rows(variant, scope, metrics, status=status)
    for row in rows:
        if row.get("ticker") == "ALL":
            row["unattempted_tasks"] = metrics.get("unattempted_tasks", 0)
    return rows


def _task_charge(tokens: TokenBudget, experiment_id: str, task_id: str) -> int:
    return sum(int(row["charged_tokens"] or 0) for row in tokens.attempts(experiment_id=experiment_id)
               if row["task_id"] == task_id)


def _completed_instances(scope: str, cell_results: list[dict]) -> int:
    reused = 4 if scope == "development" else 0
    return reused + sum(item.get("status") == "ok" for item in cell_results)


def _report(manifest: dict, b0: dict, m_stats: dict, e_stats: dict, paired: dict) -> str:
    def show(value):
        return "-" if value is None else value

    def row(label: str, stats: dict) -> str:
        requested = stats.get("requested_tasks", stats.get("tasks"))
        citation = (f"{stats.get('valid_citation_references', 0)}/"
                    f"{stats.get('citation_references', 0)}; "
                    f"rate {show(stats.get('evidence_citation_valid_rate'))}")
        latency = "/".join(str(show(stats.get(key))) for key in (
            "latency_p50_seconds", "latency_p95_seconds",
            "llm_call_latency_p50_seconds", "llm_call_latency_p95_seconds"))
        unattempted = stats.get("unattempted_tasks", max(0, requested - stats.get("tasks", 0)))
        return (f"| {label} | {requested} | {stats.get('completed_tasks')} | {stats.get('failed_tasks')} | "
                f"{unattempted} | "
                f"{show(stats.get('task_completion_rate'))} | "
                f"{stats.get('structured_output_valid_count', 0)}/{stats.get('structured_output_count', 0)}; "
                f"rate {show(stats.get('structured_output_valid_rate'))} | "
                f"{stats.get('tool_successes', 0)}/{stats.get('tool_calls', 0)}; "
                f"rate {show(stats.get('tool_success_rate'))} | "
                f"{show(stats.get('non_hold_coverage'))} | "
                f"{stats.get('directional_hits', 0)}/{stats.get('directional_n', 0)}; "
                f"rate {show(stats.get('directional_hit_rate'))} | "
                f"{show(stats.get('mean_alpha'))}; CI {show(stats.get('alpha_date_bootstrap_95'))} | "
                f"{citation} | {show(stats.get('llm_calls'))} | "
                f"{show(stats.get('input_tokens'))}/{show(stats.get('output_tokens'))}/"
                f"{show(stats.get('reasoning_tokens'))}/{show(stats.get('total_tokens_charged'))} | "
                f"{show(stats.get('total_tokens_charged_per_task'))} | "
                f"{show(stats.get('completed_tasks_per_100k_tokens'))}/"
                f"{show(stats.get('directional_hits_per_100k_tokens'))} | {latency} | "
                f"{show(stats.get('api_cost_cny_estimate'))} |")

    table = [
        "| Group | Requested | Complete | Failed | Unattempted | Completion | Structured valid | Tool calls successful | Non-Hold coverage | Direction hits / N | Mean alpha; date-cluster 95% CI | Evidence refs valid / total | LLM calls | Input / output / reasoning / charged tokens | Avg charged / attempted task | Complete / hits per 100k tokens | Task P50/P95; call P50/P95 latency (s) | List-price estimate (CNY) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        row("B0", b0), row("B1-M Flash", m_stats), row("B1-E Flash", e_stats),
    ]
    failures = {
        "B0": b0.get("failure_status_counts", {}),
        "B1-M Flash": m_stats.get("failure_status_counts", {}),
        "B1-E Flash": e_stats.get("failure_status_counts", {}),
    }
    return "\n".join([
        "# B1 GLM-5.3-Flash Baseline V2",
        "",
        f"Status: **{manifest.get('status')}**. These results use only the frozen `glm-5.3-flash` model. "
        f"Observed model identities: {', '.join(manifest.get('actual_models_observed', [])) or 'none' }.",
        "",
        *table,
        "",
        f"Paired Flash tasks: {paired.get('paired_n')}; mean paired alpha delta E−M: "
        f"{paired.get('mean_alpha_delta_e_minus_m')}; date-cluster 95% interval: "
        f"{paired.get('alpha_delta_date_bootstrap_95')}.",
        f"Directional transition denominator: {paired.get('directional_transition_n')}; "
        f"wrong-to-right: {paired.get('wrong_to_right')}; right-to-wrong: {paired.get('right_to_wrong')}; "
        f"action changed: {paired.get('action_changed_n')}.",
        "",
        f"Failure status counts by group: `{json.dumps(failures, ensure_ascii=False, sort_keys=True)}`.",
        f"B1-M calls: {m_stats.get('llm_calls')}; B1-E calls: {e_stats.get('llm_calls')}; "
        f"retry count: {m_stats.get('llm_retries', 0) + e_stats.get('llm_retries', 0)}.",
        "",
        f"Smoke source: `{manifest.get('smoke_source', {}).get('path', 'this run')}`. "
        f"Reused task instances are counted once: {manifest.get('smoke_source', {}).get('counted_again') is False}.",
        f"Quality check: `{json.dumps(manifest.get('quality_check', {}), ensure_ascii=False, sort_keys=True)}`.",
        f"Project remaining: {manifest.get('budget_capacity', {}).get('global_remaining')} Tokens; "
        f"B1 remaining: {manifest.get('budget_capacity', {}).get('phase_remaining')} Tokens.",
        "",
        "Standard price estimates use the published Flash input/output rates, with cached input "
        "priced separately when usage detail is available. They are not provider invoices. The API "
        "does not expose an experiment-specific cash debit or package debit, so those amounts remain unavailable.",
        "",
        "Flash Thinking is enabled with `reasoning_effort=low`; it cannot be disabled for this model. "
        "Reasoning is reported as a usage detail and is not double-counted when included in output tokens.",
        "",
        "No RAG, Memory, Debate, MCP, or verifier extension was enabled. Holdout tasks and labels were not accessed.",
        "",
    ])


def _archive_state(out: Path, manifest: dict, selected_ids: list[str], b0_rows: list[dict],
                   rows: dict[str, list[dict]], seed: int, config: dict,
                   token_budget: TokenBudget | None, experiment_id: str) -> tuple[dict, dict, dict]:
    for variant in ("B1-M", "B1-E"):
        folder = out / variant
        folder.mkdir(exist_ok=True)
        _write_rows(folder / "tasks.jsonl", rows[variant])
        _write_rows(folder / "trajectory.jsonl", rows[variant])
        actual = sorted({call.get("actual_model") for item in rows[variant] for call in item.get("llm_calls", [])
                         if call.get("actual_model")})
        meta = {"status": "completed" if len(rows[variant]) == len(selected_ids) and
                all(item.get("status") == "ok" for item in rows[variant]) else "incomplete",
                "ablation": variant, "requested_tasks": len(selected_ids), "processed_tasks": len(rows[variant]),
                "tasks_sha256": manifest["selected_tasks_sha256"], "config": config,
                "actual_models_observed": actual, "protocol_version": b1.B1_V2_PROTOCOL_VERSION,
                "source_experiment_ids": manifest.get("token_experiment_ids", []),
                "snapshot_sha256": manifest["protocol"]["market_manifest_sha256"],
                "evidence_manifest_sha256": manifest["protocol"]["evidence_manifest_sha256"],
                "evidence_sha256": manifest["protocol"]["evidence_sha256"],
                "prompt_hashes": manifest["protocol"]["prompt_hashes"],
                "tool_schema_hashes": manifest["protocol"]["tool_schema_hashes"],
                "budget": manifest["protocol"]["budget"], "react": True,
                "settlement": manifest["protocol"]["settlement"],
                "execution": "next shared session open to horizon-th shared session close, OHLC price return",
                "alpha": "position direction × (asset price return − SPY price return); no costs",
                "token_budget_counted_once": True}
        b1._write_json(folder / "run.json", meta)
        stats = _stats(rows[variant], seed, config, len(selected_ids))
        _write_csv(folder / "summary.csv", _summary_rows(
            variant, manifest["scope"], stats, status=meta["status"]))
    m_stats = _stats(rows["B1-M"], seed, config, len(selected_ids))
    e_stats = _stats(rows["B1-E"], seed, config, len(selected_ids))
    b0_stats = summary(b0_rows, seed, {"backend": "offline", "model": None})
    b0_stats["requested_tasks"] = len(b0_rows)
    b0_stats["unattempted_tasks"] = 0
    paired = b1._paired(rows["B1-M"], rows["B1-E"], seed)
    completion = b1._completion_check(len(selected_ids), m_stats, e_stats, paired["paired_n"])
    manifest.update(
        b1_m_summary=m_stats, b1_e_summary=e_stats, b0_summary=b0_stats,
        paired_comparison=paired, completion_check=completion,
        trajectory_sha256={variant: sha256(out / variant / "trajectory.jsonl")
                           for variant in ("B1-M", "B1-E")},
    )
    summary_rows = _summary_rows("B0", manifest["scope"], b0_stats)
    summary_rows.extend(_summary_rows("B1-M Flash", manifest["scope"], m_stats,
                                      status=manifest.get("status", "incomplete")))
    summary_rows.extend(_summary_rows("B1-E Flash", manifest["scope"], e_stats,
                                      status=manifest.get("status", "incomplete")))
    _write_csv(out / "summary.csv", summary_rows)
    b1._write_json(out / "paired_summary.json", paired)
    _write_paired_csv(out / "paired.csv", paired)
    if token_budget is not None:
        manifest["token_budget"] = token_budget.snapshot(experiment_id=experiment_id, phase="B1")
        b1._write_json(out / "token-budget.json", manifest["token_budget"])
        manifest["budget_capacity"] = token_budget.capacity(experiment_id=experiment_id, phase="B1")
    manifest["flash_baseline_total_tokens"] = sum(
        int((row.get("usage") or {}).get("total_tokens_charged") or 0)
        for group in rows.values() for row in group)
    if manifest.get("smoke_source", {}).get("source_charged_tokens") is not None:
        manifest["flash_token_sources_total"] = (
            manifest["smoke_source"]["source_charged_tokens"] +
            (manifest.get("token_budget") or {}).get("conservatively_charged_tokens", 0))
    b1._write_json(out / "manifest.json", manifest)
    (out / "REPORT.md").write_text(_report(manifest, b0_stats, m_stats, e_stats, paired), encoding="utf-8")
    return m_stats, e_stats, paired


def run_b1_v2(scope: str, tasks_path: Path, data_path: Path, evidence_path: Path,
              b0_path: Path, out: Path, *, seed: int = 42, smoke_path: Path | None = None) -> dict:
    """Run Smoke or continue from a fully verified Flash Smoke, one task pair at a time."""
    if scope not in {"smoke", "development"}:
        raise ValueError("B1 V2 scope must be smoke or development")
    if (scope == "smoke" and smoke_path is not None) or (scope == "development" and smoke_path is None):
        raise ValueError("Development requires a successful Flash Smoke archive; Smoke cannot import results")
    if out.exists():
        raise ValueError("Output path already exists; refusing to overwrite or repeat archived task instances")
    if sha256(tasks_path) != b1.EXPECTED_TASKS_SHA256:
        raise ValueError("B1 V2 uses the original frozen 20-task Development file")
    all_tasks = read_tasks(tasks_path)
    if len(all_tasks) != 20:
        raise ValueError("The frozen Development task set must contain exactly 20 tasks")
    market_hash = sha256(data_path / "manifest.json")
    evidence_manifest_path, evidence_file = evidence_path / "source_manifest.json", evidence_path / "evidence.jsonl"
    evidence_manifest_hash, evidence_hash = sha256(evidence_manifest_path), sha256(evidence_file)
    if (market_hash != b1.EXPECTED_MARKET_MANIFEST_SHA256 or
            evidence_manifest_hash != b1.EXPECTED_EVIDENCE_MANIFEST_SHA256 or
            evidence_hash != b1.EXPECTED_EVIDENCE_SHA256):
        raise ValueError("B1 V2 requires the original frozen A0.5 market and evidence snapshots")
    evidence = EvidenceStore.load(evidence_path)
    if json.loads(evidence_manifest_path.read_text())['tasks_sha256'] != b1.EXPECTED_TASKS_SHA256:
        raise ValueError("A0.5 evidence was not collected for the original Development task set")

    b0_run_path, b0_tasks_path = b0_path / "run.json", b0_path / "tasks.jsonl"
    b0_trajectory_path = b0_path / "trajectory.jsonl"
    b0_run = json.loads(b0_run_path.read_text(encoding="utf-8"))
    if (b0_run.get("status") != "completed" or b0_run.get("ablation") != "B0" or
            b0_run.get("tasks_sha256") != b1.EXPECTED_TASKS_SHA256 or
            b0_run.get("snapshot_sha256") != b1.EXPECTED_MARKET_MANIFEST_SHA256 or
            not b0_trajectory_path.is_file() or b0_tasks_path.read_bytes() != b0_trajectory_path.read_bytes()):
        raise ValueError("B0 must be the original completed 20-task deterministic baseline")
    b0_all = b1._load_jsonl(b0_tasks_path)
    if len(b0_all) != 20 or any(row.get("status") != "ok" for row in b0_all):
        raise ValueError("B0 archive must contain all 20 successful baseline tasks")
    b0_run_hash, b0_trajectory_hash = sha256(b0_run_path), sha256(b0_trajectory_path)
    protocol = _protocol(seed, market_hash=market_hash, evidence_manifest_hash=evidence_manifest_hash,
                         evidence_hash=evidence_hash, b0_run_hash=b0_run_hash,
                         b0_trajectory_hash=b0_trajectory_hash)
    protocol_hash = b1._json_hash(protocol)
    ledger_path = b1._ledger_path()
    smoke = _validated_smoke(smoke_path, protocol, tasks_path, ledger_path) if smoke_path else None
    if scope == "smoke":
        selected, selected_ids = all_tasks[:2], list(b1.SMOKE_TASK_IDS)
    else:
        selected, selected_ids = all_tasks, [task.task_id for task in all_tasks]
    if scope == "smoke" and [task.task_id for task in selected] != selected_ids:
        raise ValueError("Smoke must use the frozen dev-01/dev-02 task pair")
    out.mkdir(parents=True, exist_ok=False)
    selected_path = out / "tasks.jsonl"
    selected_path.write_bytes(b1._selected_task_bytes(tasks_path, selected_ids))
    experiment_id = str(out.resolve())
    key_available = bool(os.getenv("TALITE_API_KEY", "").strip())
    token_budget = b1._open_b1_budget(ledger_path, scope, key_available)
    config = _model_config()
    manifest = {
        "experiment": "B1 Real GLM-5.3-Flash Baseline V2",
        "scope": scope, "status": "preflight", "started_at": datetime.now(UTC).isoformat(),
        "code_git_sha": protocol["code_git_sha"], "code_worktree_dirty": bool(b1._git("status", "--porcelain")),
        "full_tasks_sha256": b1.EXPECTED_TASKS_SHA256,
        "selected_tasks_sha256": sha256(selected_path), "selected_task_ids": selected_ids,
        "market_manifest_sha256": market_hash, "evidence_manifest_sha256": evidence_manifest_hash,
        "evidence_sha256": evidence_hash,
        "b0_run_metadata_sha256": b0_run_hash, "b0_trajectory_sha256": b0_trajectory_hash,
        "prompt_hashes": protocol["prompt_hashes"], "tool_schema_hashes": protocol["tool_schema_hashes"],
        "protocol": protocol, "protocol_sha256": protocol_hash,
        "token_budget_ledger_path_sha256": b1._ledger_path_hash(ledger_path),
        "token_experiment_ids": ([smoke["manifest"]["token_budget"]["experiment_id"], experiment_id]
                                  if smoke else [experiment_id]),
        "smoke_source": ({"path": str(smoke_path), "manifest_sha256": sha256(smoke_path / "manifest.json"),
                          "reused_task_ids": list(b1.SMOKE_TASK_IDS), "reused_task_instances": 4,
                          "source_attempt_count": smoke["ledger_audit"]["attempt_count"],
                          "source_charged_tokens": smoke["ledger_audit"]["conservatively_charged_tokens"],
                          "counted_again": False} if smoke else {"path": str(out), "counted_again": False}),
        "progress": {"completed_instances": 4 if smoke else 0, "requested_instances": 4 if scope == "smoke" else 40,
                     "failed_instances": 0, "remaining_task_pairs": 2 if scope == "smoke" else 18},
        "model_config": config,
        "thinking_note": "Flash Thinking is enabled by the provider and cannot be disabled; reasoning_effort=low.",
        "actual_cash_invoice": "unavailable",
        "flash_package_debit": "unavailable; console does not report experiment-specific debit",
        "standard_price_is_invoice": False,
        "holdout_accessed": False,
        "budget_cap": {"project": b1.GLOBAL_TOKEN_LIMIT, "B1": b1.PHASE_TOKEN_LIMITS["B1"],
                       "smoke": SMOKE_TOKEN_LIMIT if scope == "smoke" else None},
        "token_budget_preflight": None,
        "token_budget": (token_budget.snapshot(experiment_id=experiment_id, phase="B1")
                         if token_budget is not None else {"status": "missing_api_key"}),
    }
    b1._write_json(out / "protocol.json", protocol)
    b1._write_json(out / "manifest.json", manifest)

    rows = {"B1-M": [], "B1-E": []}
    if smoke:
        rows["B1-M"].extend(smoke["rows"]["B1-M"])
        rows["B1-E"].extend(smoke["rows"]["B1-E"])
    b0_rows = [row for row in b0_all if row.get("task_id") in selected_ids]
    client = None
    if not key_available:
        manifest.update(status="blocked_missing_api_key", blocker="missing_api_key",
                        finished_at=datetime.now(UTC).isoformat())
        b1._write_json(out / "manifest.json", manifest)
        return {"status": manifest["status"], "output": str(out), "api_requests": 0}

    try:
        client, model, api_host = client_from_env(default_model=b1.B1_V2_MODEL,
                                                  trust_env=False, follow_redirects=False)
        if (model != b1.B1_V2_MODEL or api_host != "open.bigmodel.cn" or
                str(client.base_url).rstrip("/") != DEFAULT_BASE_URL):
            raise ValueError("Resolved model or endpoint is not the frozen GLM-5.3-Flash API")
        if not ledger_path.is_file() or token_budget is None:
            raise ValueError("The existing project token ledger is required; refusing to initialize a replacement")
        manifest.update(status="running", model_requested=model, api_host=api_host)
        if scope == "smoke":
            pair_worst_case = (2 * (MAX_TOOL_ROUNDS + 2) * TokenBudget.reservation_size(
                MAX_PROMPT_BYTES, MAX_TOKENS, protocol_allowance=PROTOCOL_TOKEN_ALLOWANCE,
                separate_reasoning_tokens=MAX_TOKENS))
            first_preflight = token_budget.can_reserve(pair_worst_case, experiment_id=experiment_id, phase="B1")
            manifest["token_budget_preflight"] = {
                **first_preflight, "first_pair_worst_case_tokens": pair_worst_case,
                "full_smoke_static_upper_bound": 2 * pair_worst_case,
                "smoke_static_upper_bound_exceeds_cap": 2 * pair_worst_case > SMOKE_TOKEN_LIMIT,
                "admission": "pair-by-pair forecast plus atomic per-HTTP reservation",
            }
            if not first_preflight["fits"]:
                manifest.update(status="blocked_token_budget", blocker="pair_preflight_denied",
                                finished_at=datetime.now(UTC).isoformat())
                b1._write_json(out / "manifest.json", manifest)
                return {"status": manifest["status"], "output": str(out), "api_requests": 0}
        else:
            smoke_totals = [_task_charge(
                token_budget, smoke["manifest"]["token_budget"]["experiment_id"], task_id)
                for task_id in b1.SMOKE_TASK_IDS]
            pair_estimate = max(1, max(smoke_totals, default=1) * PAIR_SAFETY_MULTIPLIER)
            remaining_pairs = 18
            capacity = token_budget.capacity(experiment_id=experiment_id, phase="B1")
            forecast = {"smoke_task_charged_tokens": smoke_totals,
                        "safety_multiplier": PAIR_SAFETY_MULTIPLIER,
                        "estimated_tokens_per_task_pair": pair_estimate,
                        "remaining_task_pairs": remaining_pairs,
                        "forecast_remaining_tokens": pair_estimate * remaining_pairs,
                        "available_before_development": capacity["available"],
                        "fits": pair_estimate * remaining_pairs <= capacity["available"]}
            manifest["token_budget_preflight"] = forecast
            if not forecast["fits"]:
                manifest.update(status="blocked_token_budget_forecast", blocker="development_forecast_exceeds_budget",
                                finished_at=datetime.now(UTC).isoformat())
                rows = {"B1-M": list(smoke["rows"]["B1-M"]), "B1-E": list(smoke["rows"]["B1-E"])}
                _archive_state(out, manifest, selected_ids, b0_all, rows, seed, config, token_budget, experiment_id)
                return {"status": manifest["status"], "output": str(out), "api_requests": 0,
                        "preflight": forecast}
        b1._write_json(out / "manifest.json", manifest)

        total_new_pairs = 2 if scope == "smoke" else 18
        new_tasks = selected if scope == "smoke" else selected[2:]
        processed_new_pairs = 0
        failed = False
        for task_index, task in enumerate(new_tasks):
            if scope == "smoke" and task_index > 0:
                prior_pair_charge = _task_charge(token_budget, experiment_id, selected[task_index-1].task_id)
                pair_estimate = max(1, prior_pair_charge * PAIR_SAFETY_MULTIPLIER)
                batch_preflight = token_budget.can_reserve(pair_estimate, experiment_id=experiment_id, phase="B1")
                if not batch_preflight["fits"]:
                    manifest.update(status="blocked_token_budget", blocker="next_pair_forecast_exceeds_budget",
                                    next_pair_preflight={**batch_preflight,
                                                         "pair_safety_multiplier": PAIR_SAFETY_MULTIPLIER})
                    break
            elif scope == "development":
                smoke_totals = [_task_charge(
                    token_budget, smoke["manifest"]["token_budget"]["experiment_id"], task_id)
                    for task_id in b1.SMOKE_TASK_IDS]
                pair_estimate = max(1, max(smoke_totals, default=1) * PAIR_SAFETY_MULTIPLIER)
                batch_preflight = token_budget.can_reserve(pair_estimate, experiment_id=experiment_id, phase="B1")
                if not batch_preflight["fits"]:
                    manifest.update(status="blocked_token_budget", blocker="next_pair_forecast_exceeds_budget",
                                    next_pair_preflight={**batch_preflight,
                                                         "pair_safety_multiplier": PAIR_SAFETY_MULTIPLIER})
                    break
            task_slice = out / "task-slices" / f"{task.task_id}.jsonl"
            task_slice.parent.mkdir(parents=True, exist_ok=True)
            task_slice.write_bytes(b1._selected_task_bytes(tasks_path, [task.task_id]))
            for variant, condition in (("B1-M", None), ("B1-E", evidence_path)):
                cell_out = out / "cells" / task.task_id / variant.replace("-", "_")
                budget = Budget(max_tool_rounds=MAX_TOOL_ROUNDS, max_tokens=MAX_TOKENS)
                factory = lambda: ModelBackend(
                    client, model, budget, None, max_retries=MAX_RETRIES,
                    max_prompt_bytes=MAX_PROMPT_BYTES, temperature=DEFAULT_TEMPERATURE,
                    reasoning_effort=REASONING_EFFORT, token_budget=token_budget, phase="B1",
                    require_model_identity=True, compact_context=True,
                )
                metadata = {"protocol_version": b1.B1_V2_PROTOCOL_VERSION,
                            "protocol_sha256": protocol_hash, "prompt_hashes": protocol["prompt_hashes"],
                            "tool_schema_hashes": protocol["tool_schema_hashes"],
                            "max_tool_calls_per_round": MAX_TOOL_CALLS_PER_ROUND,
                            "holdout_accessed": False}
                stats = run(task_slice, data_path, cell_out, factory, config, budget, seed, True,
                            evidence_path=condition, run_label=variant, protocol_metadata=metadata,
                            token_budget=token_budget, experiment_id=experiment_id, phase="B1",
                            )
                cell_rows = b1._load_jsonl(cell_out / "trajectory.jsonl")
                if len(cell_rows) != 1:
                    raise ValueError("A B1 V2 task cell did not produce exactly one archived task row")
                rows[variant].extend(cell_rows)
                _assert_flash_rows(cell_rows, require_calls=False)
                status = cell_rows[0].get("status")
                calls = cell_rows[0].get("llm_calls", [])
                actual = sorted({call.get("actual_model") for call in calls if call.get("actual_model")})
                manifest.setdefault("cell_results", []).append({
                    "task_id": task.task_id, "variant": variant, "status": status,
                    "relative_path": str(cell_out.relative_to(out)),
                    "request_count": sum(call.get("attempt", 0) > 0 for call in calls),
                    "actual_models": actual, "charged_tokens": sum(call.get("charged_tokens") or
                        call.get("reserved_tokens") or 0 for call in calls),
                    "quality_issues": _cell_quality_issues(cell_rows[0]),
                })
                completed_instances = _completed_instances(scope, manifest.get("cell_results", []))
                manifest["progress"] = {
                    "completed_instances": completed_instances,
                    "requested_instances": 4 if scope == "smoke" else 40,
                    "failed_instances": sum(1 for item in manifest.get("cell_results", [])
                                             if item.get("status") != "ok"),
                    "remaining_task_pairs": max(0, total_new_pairs - processed_new_pairs),
                    "current_task_id": task.task_id,
                    "current_variant": variant,
                }
                if token_budget is not None:
                    manifest["token_budget"] = token_budget.snapshot(experiment_id=experiment_id, phase="B1")
                    manifest["budget_capacity"] = token_budget.capacity(experiment_id=experiment_id, phase="B1")
                _archive_state(out, manifest, selected_ids, b0_rows, rows, seed, config, token_budget, experiment_id)
                print(f"[B1-V2] {scope} {task.task_id} {variant}: {status}; "
                      f"instances {completed_instances}/{'4' if scope == 'smoke' else '40'}; "
                      f"Flash tokens in this run {manifest.get('token_budget', {}).get('conservatively_charged_tokens', 0)}; "
                      f"B1 remaining {manifest.get('budget_capacity', {}).get('phase_remaining', 'unknown')}; "
                      f"requests {sum(call.get('attempt', 0) > 0 for call in calls)}; model "
                      f"{','.join(actual) if actual else 'unconfirmed'}", flush=True)
                if status != "ok" or stats.get("budget_exhausted") or any(
                        call.get("error_code") in {"model_identity_mismatch", "output_truncated"}
                        for call in calls) or _cell_quality_issues(cell_rows[0]):
                    failed = True
                    manifest.update(status="failed_smoke" if scope == "smoke" else "incomplete",
                                    blocker="task_failure_or_budget_stop",
                                    blocker_reason="Stopped immediately after a failed task instance; no automatic retry.")
                    break
            if not failed:
                processed_new_pairs += 1
            manifest["progress"]["remaining_task_pairs"] = max(0, total_new_pairs-processed_new_pairs)
            if failed or manifest.get("blocker") == "next_pair_forecast_exceeds_budget":
                break

        if not failed and not manifest.get("blocker"):
            manifest["status"] = "completed"
        elif manifest.get("blocker") == "next_pair_forecast_exceeds_budget":
            manifest["status"] = "blocked_token_budget"
        manifest["actual_models_observed"] = sorted({call.get("actual_model")
            for group in rows.values() for row in group for call in row.get("llm_calls", [])
            if call.get("actual_model")})
        manifest["flash_baseline_total_tokens"] = sum(
            int((row.get("usage") or {}).get("total_tokens_charged") or 0)
            for group in rows.values() for row in group)
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        if scope == "smoke":
            manifest["smoke_budget_remaining"] = (token_budget.capacity(
                experiment_id=experiment_id, phase="B1")["experiment_remaining"] if token_budget else None)
        quality_issues = [issue for item in manifest.get("cell_results", [])
                          for issue in item.get("quality_issues", [])]
        if smoke:
            quality_issues.extend(issue for group in smoke["rows"].values() for row in group
                                   for issue in _cell_quality_issues(row))
        quality_complete = (manifest.get("completion_check", {}).get("complete") is True and
                            not quality_issues)
        manifest["quality_check"] = {
            "complete": quality_complete,
            "failed_checks": sorted(set(quality_issues)),
            "requires_two_valid_json_outputs_per_task": True,
            "requires_successful_market_tool_call_per_task": True,
            "requires_valid_evidence_references": True,
            "requires_no_unexpected_retry_or_truncation": True,
        }
        if manifest.get("status") == "completed" and not quality_complete:
            manifest.update(status="failed_smoke" if scope == "smoke" else "incomplete",
                            blocker="smoke_quality_check_failed",
                            blocker_reason="One or more archived task/tool/output quality checks did not pass.")
        _archive_state(out, manifest, selected_ids, b0_rows, rows, seed, config, token_budget, experiment_id)
        if manifest["status"] == "completed":
            _assert_flash_rows(rows["B1-M"] + rows["B1-E"])
        return {"status": manifest["status"], "output": str(out),
                "completed_instances": manifest["progress"]["completed_instances"],
                "paired_tasks": manifest.get("completion_check", {}).get("valid_pairs", 0),
                "token_budget": manifest.get("token_budget"),
                "preflight": manifest.get("token_budget_preflight")}
    finally:
        if client is not None:
            client.close()
