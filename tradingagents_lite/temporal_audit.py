"""Execute frozen GLM recall probes through the existing atomic SQLite budget."""
import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from . import b1
from .b1_v3 import _lock_archive
from .data import sha256
from .data_v2_market import write_json
from .provider import DEFAULT_BASE_URL, ModelBackend, ModelCallError, client_from_env
from .schema import Budget
from .temporal_probes import PROMPT, SCORING, freeze_probes, question, score, summarize
from .token_budget import TokenBudget


MODEL = "glm-5.3-flash"
CAP = 60_000
PHASE = "Reserve"
OUTPUT_LIMIT = 128
MAX_PROMPT = 2000
RELEASE = "2026-08-27"
EXPERIMENT_ID = "v3-c-temporal-knowledge-audit-001"


def freeze(market, sec, out):
    manifest = freeze_probes(market, sec, out)
    manifest.update({
        "api": {"endpoint": DEFAULT_BASE_URL, "model": MODEL, "temperature": 0, "reasoning_effort": "low",
                "max_output_tokens": OUTPUT_LIMIT, "max_prompt_bytes": MAX_PROMPT, "retries": 0,
                "tools": False, "history": False},
        "code_commit": b1._git("rev-parse", "HEAD"),
        "code_hashes": {str(path): sha256(path) for path in [
            Path(__file__).resolve().relative_to(Path.cwd()),
            Path("tradingagents_lite/temporal_probes.py"), Path("tradingagents_lite/provider.py"),
            Path("tradingagents_lite/token_budget.py"), Path("uv.lock")]},
        "budget_phase": PHASE, "cap": CAP, "official_product_release": RELEASE,
    })
    write_json(out / "freeze.json", manifest)


def execute(out):
    with _lock_archive(out):
        manifest = json.loads((out / "freeze.json").read_text())
        if (sha256(out / "probes.jsonl") != manifest["probe_sha256"]
                or manifest["scoring"] != SCORING or manifest["prompt"] != PROMPT
                or any(sha256(Path(path)) != expected for path, expected in manifest["code_hashes"].items())):
            raise ValueError("Probe, code, prompt or scoring hash differs from the freeze")
        if (out / "attempts.jsonl").exists():
            raise ValueError("An audit run already exists. Preserve its attempts and summary.")
        rows = b1._load_jsonl(out / "probes.jsonl")
        if len(rows) != 60 or len({row["probe_id"] for row in rows}) != 60:
            raise ValueError("Expected sixty unique frozen probes")
        path = b1._ledger_path()
        # Validate the existing ledger using the project guard before opening it with the audit cap.
        b1._open_b1_budget(path, "development", True)
        ledger = TokenBudget(path, experiment_limit=CAP)
        if ledger.attempts(experiment_id=EXPERIMENT_ID):
            raise ValueError("Audit ledger already has attempts under the frozen experiment ID")
        backend_budget = Budget(max_tokens=OUTPUT_LIMIT)
        client, model, host = client_from_env(default_model=MODEL, trust_env=False, follow_redirects=False)
        results, stop = [], None
        started = datetime.now(UTC).isoformat()
        try:
            if model != MODEL or host != "open.bigmodel.cn" or str(client.base_url).rstrip("/") != DEFAULT_BASE_URL:
                raise ValueError("The knowledge audit is pinned to the official Flash API")
            with (out / "attempts.jsonl").open("x", encoding="utf-8") as stream:
                for row in rows:
                    backend = ModelBackend(client, MODEL, backend_budget, None, max_retries=0,
                                           max_prompt_bytes=MAX_PROMPT, temperature=0, reasoning_effort="low",
                                           token_budget=ledger, phase=PHASE, require_model_identity=True)
                    backend.set_budget_context(experiment_id=EXPERIMENT_ID, task_id=row["probe_id"], phase=PHASE)
                    try:
                        reply = backend._call([{"role": "system", "content": PROMPT},
                                               {"role": "user", "content": question(row)}], [], record_version=True)
                        result = (score(row, reply["content"]) if not reply.get("tool_calls") else
                                  {**score(row, ""), "error": "unexpected_tool_calls"})
                    except ModelCallError as exc:
                        result = {**score(row, ""), "error": exc.code}
                        if exc.code not in {"output_truncated", "invalid_response_shape"}:
                            stop = exc.code
                    record = {**result, "calls": backend.call_log}
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    if stop:
                        break
                    results.append(result)
                    if len(results) % 10 == 0:
                        used = ledger.capacity(experiment_id=EXPERIMENT_ID, phase=PHASE)["experiment_used"]
                        print(f"Probes {len(results)}/60, {used} tokens charged", flush=True)
        finally:
            client.close()
        snapshot = ledger.snapshot(experiment_id=EXPERIMENT_ID, phase=PHASE)
        stat = summarize(results)
        report = {**stat, "requested": len(rows), "scored": len(results), "stop": stop, "started_utc": started,
                  "finished_utc": datetime.now(UTC).isoformat(), "tokens": snapshot,
                  "audit_sha256": sha256(out / "attempts.jsonl"), "freeze_sha256": sha256(out / "freeze.json"),
                  "official_product_release": RELEASE}
        write_json(out / "summary.json", report)
        print(json.dumps({"scored": len(results), "stop": stop, "boundary": stat["boundary"],
                          "tokens": snapshot["experiment_used"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--out", type=Path, default=Path("results/temporal-knowledge-audit-001"))
    parser.add_argument("--market", type=Path, default=Path("data/data-expansion-v3/market"))
    parser.add_argument("--sec", type=Path, default=Path("data/data-expansion-v3/sec"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.market, args.sec, args.out)
    else:
        execute(args.out)


if __name__ == "__main__":
    main()
