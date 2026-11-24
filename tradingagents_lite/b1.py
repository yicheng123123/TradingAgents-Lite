"""Frozen B1 paired smoke/development protocol for GLM and A0.5 evidence."""
import csv
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess

from .data import read_tasks, sha256
from .evidence import EvidenceStore
from .provider import (
    DEFAULT_BASE_URL, DEFAULT_MAX_PROMPT_BYTES, DEFAULT_MAX_RETRIES, DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT, DEFAULT_TEMPERATURE, ModelBackend, client_from_env,
)
from .prompts import prompt_hashes
from .runner import clustered_bootstrap, run, summary
from .schema import Budget
from .workflow import MomentumBackend


EXPECTED_TASKS_SHA256 = "2d0e4f92d583d64ceba729b4e3ffa0004b3bd29a5a00d990434dcdab5fceedae"
EXPECTED_MARKET_MANIFEST_SHA256 = "f5e4c5095953ce737b7b99bba71b2ad26a9f30d2f34b9fba8320027e46e0f3cb"
EXPECTED_EVIDENCE_MANIFEST_SHA256 = "b46f9ef2ebc1c08022500282147940985eb9d3ceabe8cdaf78eec6efa190e4dc"
EXPECTED_EVIDENCE_SHA256 = "b0ffce595575257231432650495d756681a9f4ac10f5deee972110c7a7b7a532"
SMOKE_TASK_IDS = ("dev-01", "dev-02")
MAX_TOOL_ROUNDS = 2
MAX_TOKENS = 1200
MAX_PROMPT_BYTES = DEFAULT_MAX_PROMPT_BYTES
MAX_RETRIES = DEFAULT_MAX_RETRIES
TEMPERATURE = DEFAULT_TEMPERATURE
REASONING_EFFORT = DEFAULT_REASONING_EFFORT
PRICE_CNY_PER_MILLION = {"input": 8.0, "output": 28.0}
PRICE_SOURCE = "https://docs.bigmodel.cn/cn/guide/start/pricing"
PRICE_AS_OF = "2026-11-24"


def request_cost_ceiling(task_count: int) -> dict:
    """Conservative standard-price bound from capped prompt bytes and output tokens."""
    workflows = task_count * 2
    logical_calls_per_task = MAX_TOOL_ROUNDS + 2  # final analyst response and Trader call
    attempts = workflows * logical_calls_per_task * (MAX_RETRIES + 1)
    protocol_token_allowance = 512
    input_token_upper = attempts * (MAX_PROMPT_BYTES + protocol_token_allowance)
    output_token_upper = attempts * MAX_TOKENS
    cost = (input_token_upper * PRICE_CNY_PER_MILLION["input"] +
            output_token_upper * PRICE_CNY_PER_MILLION["output"]) / 1_000_000
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
        "list_price_cost_ceiling_cny": round(cost, 6),
        "price_cny_per_million": PRICE_CNY_PER_MILLION,
        "price_as_of": PRICE_AS_OF,
        "price_source": PRICE_SOURCE,
        "note": "Conservative byte/token envelope; provider invoice and cached-token rates may differ.",
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
        "tasks", "completed_tasks", "task_completion_rate", "structured_output_valid_rate",
        "structured_output_valid_tasks", "directional_n", "non_hold_coverage",
        "directional_hit_rate", "hit_rate_wilson_95", "mean_alpha", "alpha_date_bootstrap_95",
        "tool_success_rate", "tool_failure_events", "tool_recovery_events", "tool_recovery_rate",
        "tool_recovered_tasks", "evidence_citation_valid_rate", "invalid_citation_rate",
        "valid_citation_references", "invalid_citation_references",
        "financial_citation_references", "eligible_financial_citation_task_rate",
        "input_tokens", "output_tokens", "llm_calls", "llm_retries", "api_cost_cny_estimate",
        "latency_mean_seconds",
    )
    output = []
    for ticker in [*sorted(metrics.get("by_ticker", {})), "ALL"]:
        item = metrics.get("by_ticker", {}).get(ticker, metrics) if ticker != "ALL" else metrics
        output.append({"variant": variant, "scope": scope, "ticker": ticker, "status": status,
                       **{key: item.get(key) for key in columns}})
    return output


def _write_summary_csv(path: Path, rows: list[dict]):
    fields = list(rows[0]) if rows else ["variant", "scope", "ticker", "status"]
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
        if left.get("status") == right.get("status") == "ok" and left.get("alpha") is not None:
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
    return {
        "paired_n": len(paired),
        "mean_alpha_delta_e_minus_m": (
            sum(row["alpha_delta_e_minus_m"] for row in paired) / len(paired) if paired else None
        ),
        "alpha_delta_date_bootstrap_95": clustered_bootstrap(deltas, seed),
        "date_clusters": len({row["decision_date"] for row in paired}),
        "action_changed_n": sum(row["action_b1_m"] != row["action_b1_e"] for row in paired),
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
        "| Variant | Scope | Ticker | Completed / requested | Task completion | Structured valid | Non-Hold coverage | Hit rate (non-Hold) | Mean alpha | Tool success | Recovery events / rate | Citation valid / invalid | LLM calls | Input / output tokens | API cost (CNY) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"{row.get('task_completion_rate') if row.get('task_completion_rate') is not None else '-'} | "
            f"{row.get('structured_output_valid_rate') if row.get('structured_output_valid_rate') is not None else '-'} | "
            f"{row.get('non_hold_coverage') if row.get('non_hold_coverage') is not None else '-'} | "
            f"{rate if rate is not None else '-'} | {alpha if alpha is not None else '-'} | "
            f"{tool_success if tool_success is not None else '-'} | {recovery} | {citation} | "
            f"{row.get('llm_calls') if row.get('llm_calls') is not None else '-'} | "
            f"{(str(row.get('input_tokens')) + ' / ' + str(row.get('output_tokens'))) if row.get('input_tokens') is not None else '-'} | "
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
        f"{manifest['cost_ceiling']['protocol_token_allowance_per_attempt']} protocol-token allowance, all bounded "
        f"retries are charged, and each response uses the full {MAX_TOKENS}-token cap. This is a ceiling, not an observed bill.",
        "The rough smoke expectation before API usage is approximately 0.2–0.5 CNY, assuming 12 model requests, "
        "1–3k input tokens and 200–500 output tokens per request. This is an estimate only. GLM-5.3 standard "
        "list prices were captured from the official pricing page; account discounts, cache treatment and billing "
        "may differ.",
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
            f"- Per ticker: {json.dumps(paired['by_ticker'], ensure_ascii=False, sort_keys=True)}",
            "",
        ])
    return "\n".join(lines)


def run_b1(scope: str, tasks_path: Path, data_path: Path, evidence_path: Path,
           b0_path: Path, out: Path, seed: int = 42, budget_cap_cny: float | None = None) -> dict:
    if scope not in {"smoke", "development"}:
        raise ValueError("B1 scope must be smoke or development")
    raw_tasks = tasks_path.read_bytes()
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

    selected = all_tasks[:2] if scope == "smoke" else all_tasks
    if scope == "smoke" and tuple(task.task_id for task in selected) != SMOKE_TASK_IDS:
        raise ValueError("Smoke task selection must be the frozen dev-01/dev-02 pair")
    selected_ids = [task.task_id for task in selected]
    selected_lines = _read_original_lines(tasks_path)
    selected_bytes = "".join(selected_lines[task_id] + "\n" for task_id in selected_ids).encode("utf-8")
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

    b0_run = json.loads((b0_path / "run.json").read_text(encoding="utf-8"))
    if (b0_run.get("status") != "completed" or b0_run.get("ablation") != "B0" or
            b0_run.get("tasks_sha256") != EXPECTED_TASKS_SHA256 or
            b0_run.get("snapshot_sha256") != EXPECTED_MARKET_MANIFEST_SHA256):
        raise ValueError("B0 archive must be a completed 20-task run on the frozen Development snapshot")
    b0_rows = _load_jsonl(b0_path / "tasks.jsonl")
    if len(b0_rows) != 20 or any(row.get("status") != "ok" for row in b0_rows):
        raise ValueError("B0 archive must contain 20 successfully completed Development trajectories")
    b0_full = summary(b0_rows, seed, b0_run["config"])
    b0_scope_rows = [row for row in b0_rows if row["task_id"] in selected_ids]
    b0_scope = summary(b0_scope_rows, seed, b0_run["config"])

    model = os.getenv("TALITE_MODEL", DEFAULT_MODEL)
    base_url = os.getenv("TALITE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if model != DEFAULT_MODEL or base_url != DEFAULT_BASE_URL:
        raise ValueError("Frozen B1 protocol requires TALITE_MODEL=glm-5.3 and the official OpenAI-compatible GLM endpoint")
    code_worktree_dirty = bool(_git("status", "--porcelain"))
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
        "protocol": {
            "model_requested": model, "base_url": base_url, "temperature": TEMPERATURE,
            "max_tokens_per_model_call": MAX_TOKENS, "max_tool_rounds": MAX_TOOL_ROUNDS,
            "max_tool_calls_per_round": 2, "tool_retries": 0, "model_max_retries": MAX_RETRIES,
            "model_timeout_seconds": 25, "max_prompt_bytes": MAX_PROMPT_BYTES,
            "provider_generation_seed": None, "statistics_seed": seed,
            "reasoning_effort": REASONING_EFFORT, "thinking": "enabled",
            "settlement": "next shared session open to horizon-th shared session close; alpha vs SPY; no costs",
            "holdout_accessed": False,
        },
        "cost_ceiling": cost_ceiling,
        "budget_cap_cny": budget_cap_cny,
        "b0_reference": str(b0_path),
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
            "or committing it, then rerun with a new --output directory. The configured model and endpoint "
            "were taken from the checked official GLM documentation, but account-level authentication and "
            "actual response model identity remain untested."
        )
        manifest.update(status="blocked_missing_api_key", blocker="missing_api_key",
                        finished_at=datetime.now(UTC).isoformat())
        requested_tickers = sorted({task.symbol for task in selected})
        for variant in ("B1-M", "B1-E"):
            for ticker in [*requested_tickers, "ALL"]:
                summary_rows.append({"variant": variant, "scope": scope, "ticker": ticker,
                    "status": "not_run_missing_api_key", "requested_tasks": len(selected),
                    "tasks": 0, "completed_tasks": None, "directional_n": None,
                    "directional_hit_rate": None, "mean_alpha": None,
                    "evidence_citation_valid_rate": None, "llm_calls": None,
                    "input_tokens": None, "output_tokens": None, "api_cost_cny_estimate": None})
        paired = None
        manifest["b1_m_and_b1_e"] = {"status": "not_run", "reason": "missing_api_key"}
    else:
        client, configured_model, api_host = client_from_env()
        if configured_model != DEFAULT_MODEL or api_host != "open.bigmodel.cn":
            client.close()
            raise ValueError("Resolved GLM endpoint/model differs from the frozen B1 protocol")
        budget = Budget(max_tool_rounds=MAX_TOOL_ROUNDS, max_tokens=MAX_TOKENS)
        model_config = {
            "backend": "llm", "model": configured_model, "api_host": api_host,
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "thinking": "enabled", "reasoning_effort": REASONING_EFFORT,
            "provider_seed": None, "sdk_retries": 0, "application_retries": MAX_RETRIES,
            "max_prompt_bytes": MAX_PROMPT_BYTES,
            "pricing_cny_per_million": PRICE_CNY_PER_MILLION,
            "pricing_source": PRICE_SOURCE, "pricing_as_of": PRICE_AS_OF,
        }
        protocol_metadata = {"prompt_hashes": prompt_hashes(), "max_model_retries": MAX_RETRIES,
            "max_prompt_bytes": MAX_PROMPT_BYTES, "model_timeout_seconds": 25,
            "provider_generation_seed": None, "holdout_accessed": False}
        manifest.update(status="running", api_host=api_host, model_requested=configured_model)
        _write_json(manifest_path, manifest)
        try:
            for variant, financial_path in (("B1-M", None), ("B1-E", evidence_path)):
                factory = lambda: ModelBackend(
                    client, configured_model, budget, None, max_retries=MAX_RETRIES,
                    max_prompt_bytes=MAX_PROMPT_BYTES, temperature=TEMPERATURE,
                    reasoning_effort=REASONING_EFFORT,
                )
                stats = run(out / "tasks.jsonl", data_path, out / variant, factory, model_config,
                    budget, seed, True, evidence_path=financial_path, run_label=variant,
                    protocol_metadata=protocol_metadata)
                rows_for_variant = _load_jsonl(out / variant / "tasks.jsonl")
                summary_rows.extend(_summary_rows(variant, scope, stats))
                if variant == "B1-M":
                    m_rows = rows_for_variant
                    m_stats = stats
                else:
                    e_rows = rows_for_variant
                    e_stats = stats
            paired = _paired(m_rows, e_rows, seed)
            actual = sorted({call.get("actual_model") for rows in (m_rows, e_rows) for row in rows
                             for call in row.get("llm_calls", []) if call.get("actual_model")})
            mismatch = not actual or any(name != DEFAULT_MODEL for name in actual)
            manifest.update(
                status="model_identity_unconfirmed" if mismatch else "completed",
                actual_models_observed=actual,
                b1_m_summary=m_stats,
                b1_e_summary=e_stats,
                paired_comparison=paired,
                finished_at=datetime.now(UTC).isoformat(),
            )
            if mismatch:
                blocked_reason = "The API returned a model identity different from the frozen glm-5.3 request."
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
