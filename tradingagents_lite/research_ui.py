"""Read-only dashboard data and the explicit single-task research boundary."""
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from .agents import A3Config
from .data import read_tasks, sha256
from .data_v2_tasks import frozen_tasks
from .experiment import execute_experiment
from .token_budget import DEFAULT_LEDGER_PATH, GLOBAL_TOKEN_LIMIT, PHASE_TOKEN_LIMITS


ROOT = Path(__file__).resolve().parents[1]
PAGES = ("Research", "Agent Trace", "Evidence & Memory", "Benchmark", "Token Dashboard")
V2_DATA = ROOT / "data/data-expansion-v2"
V2_TASKS = ROOT / "tasks/data-expansion-v2"
V2_PARTITION = "expansion-v2"
ARCHIVES = {
    "A0": "smoke-a0-001/summary.csv",
    "B0": "b0-development-001/summary.csv",
    "A1": "a1-rag-local-001/public-four-mode-summary.json",
    "A2": "a2-memory-local-001/summary.json",
    "A3": "a3-offline-001/summary.json",
}


def ui_root():
    return Path(os.getenv("TALITE_UI_RUNS", str(ROOT / "data/ui-runs")))


def research_tasks(dataset):
    if dataset == "Legacy":
        return read_tasks(ROOT / "tasks/development.jsonl")
    if dataset == "Development V2":
        return frozen_tasks(V2_TASKS, "development")
    raise ValueError("Unsupported interactive task partition")


def v2_coverage():
    path = ROOT / "results/data-expansion-v2-001/summary.json"
    return json.loads(path.read_text())["coverage"]["development"] if path.is_file() else None


def archive_for(stage: str, root: Path = ROOT):
    relative = ARCHIVES[stage]
    path = root / "results" / relative
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as reader:
            return next(csv.DictReader(reader))
    return json.loads(path.read_text())


def load_run(run_id: str | None):
    if not run_id or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        return None
    file = ui_root() / run_id / "run/tasks.jsonl"
    if file.is_file():
        with file.open(encoding="utf-8") as stream:
            row = json.loads(next(stream, "null"))
        metadata = file.with_name("run.json")
        if row and metadata.is_file():
            row["run_config"] = json.loads(metadata.read_text())["config"]
        return row
    return None


def start_run(task, *, run_id, backend, rag, memory, debate, transport, verifier, repair):
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Invalid research run identity")
    if backend == "llm" and not os.getenv("TALITE_API_KEY", "").strip():
        raise ValueError("missing_api_key")
    if task.task_id.startswith("v2-"):
        market = V2_DATA / "market"
        if task not in frozen_tasks(V2_TASKS, "development", market):
            raise ValueError("Only the frozen Development V2 partition is interactive")
        evidence = V2_DATA / "evidence"
        index = V2_DATA / "rag" if rag != "Off" else None
        if index is not None:
            signature = json.loads((index / "manifest.json").read_text())
            if (signature["partition"] != V2_PARTITION
                    or signature["source_manifest_sha256"] != sha256(evidence / "source_manifest.json")):
                raise ValueError("V2 evidence and index versions differ")
        fundamentals = V2_DATA / "sec/fundamentals"
        # Any V2 episode lives in its own run database. No legacy seed is copied.
        memory_seed, partition = None, V2_PARTITION
    else:
        market, evidence = ROOT / "data/yahoo-2024-v1", ROOT / "data/financial-evidence-dev-003"
        index = ROOT / "data/a1-development-001" if rag != "Off" else None
        memory_seed, fundamentals, partition = ROOT / "data/memory-run-004/store/episodes.sqlite", None, "development"
    folder = ui_root() / run_id
    folder.mkdir(parents=True, exist_ok=False)
    tasks = folder / "task.jsonl"
    tasks.write_text(task.model_dump_json() + "\n")
    cfg = A3Config(debate=debate, transport=transport, verifier=verifier, repair=repair, partition=partition)
    return execute_experiment(tasks, market, folder / "run", cfg,
        backend=backend, evidence=evidence, rag_index=index, rag_mode=rag,
        memory_mode=memory, models=ROOT / "data/local-models",
        memory_seed=memory_seed, run_id=run_id, fundamentals=fundamentals)


def new_run_id():
    return uuid4().hex


def token_dashboard(path: Path | None = None):
    path = path or Path(os.getenv("TALITE_TOKEN_BUDGET_DB") or DEFAULT_LEDGER_PATH)
    records = []
    estimate_by_model = {}
    if path.is_file():
        con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            records = [dict(r) for r in con.execute("""
                SELECT phase, SUM(CASE WHEN usage_known=1 THEN charged_tokens ELSE 0 END) AS settled,
                       SUM(CASE WHEN usage_known=0 THEN charged_tokens ELSE 0 END) AS pending,
                       SUM(COALESCE(input_tokens,0)) AS input, SUM(COALESCE(output_tokens,0)) AS output,
                       SUM(COALESCE(cached_tokens,0)) AS cached, SUM(COALESCE(reasoning_tokens,0)) AS reasoning
                FROM token_attempts GROUP BY phase
            """)]
            bills = con.execute("""
                SELECT model, SUM(input_tokens) AS input, SUM(output_tokens) AS output,
                       SUM(cached_tokens) AS cached,
                       SUM(CASE WHEN reasoning_tokens > output_tokens
                                THEN reasoning_tokens-output_tokens ELSE 0 END) AS separate_reasoning,
                       COUNT(*) AS attempts
                FROM token_attempts WHERE usage_known=1 GROUP BY model
            """).fetchall()
            rates = {
                "glm-5.3": {"input": 8.0, "cached_input": 2.0, "output": 28.0},
                "glm-5.3-flash": {"input": 0.8, "cached_input": 0.23, "output": 2.8},
            }
            for bill in bills:
                price = rates.get(bill["model"])
                if price and bill["attempts"]:
                    prompt = bill["input"] or 0
                    cached = min(bill["cached"] or 0, prompt)
                    estimate_by_model[bill["model"]] = (
                        (prompt-cached)*price["input"] + cached*price["cached_input"] +
                        ((bill["output"] or 0) + (bill["separate_reasoning"] or 0))*price["output"]
                    ) / 1_000_000
        finally:
            con.close()
    indexed = {row["phase"]: row for row in records}
    phases = []
    for name, cap in PHASE_TOKEN_LIMITS.items():
        usage = {key: indexed.get(name, {}).get(key, 0)
                 for key in ("settled", "pending", "input", "output", "cached", "reasoning")}
        phases.append({"phase": name, "budget": cap, **usage,
                       "remaining": max(0, cap - usage["settled"] - usage["pending"])})
    total = {key: sum(row[key] for row in records)
             for key in ("settled", "pending", "input", "output", "cached", "reasoning")}
    return {"exists": path.is_file(), "limit": GLOBAL_TOKEN_LIMIT, **total,
            "remaining": max(0, GLOBAL_TOKEN_LIMIT-total["settled"]-total["pending"]),
            "phases": phases,
            "list_price_estimates_cny": estimate_by_model,
            "list_price_estimate_cny": estimate_by_model.get("glm-5.3"), "invoice_cny": None}
