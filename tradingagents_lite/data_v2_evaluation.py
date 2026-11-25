"""Input quality and fixed-rule scoring on Offline Extended, never Holdout."""
import argparse
from contextlib import ExitStack
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import random
import resource
from statistics import fmean
from time import perf_counter

from .data import DataError, HistoricalData, close_available, sha256 as file_hash
from .data_v2_corpus import PARTITION
from .data_v2_market import ISSUERS, ROOT, write_json
from .data_v2_sec import write_rows
from .data_v2_tasks import frozen_tasks
from .evidence import EvidenceStore
from .fundamentals import FundamentalsView, METRICS
from .rag import RagConfig, RagView, Retriever
from .runner import clustered_bootstrap
from .schema import Budget
from .vectors import LocalEmbedding, VectorStore


RANDOM_SEEDS = (17, 43, 101, 211, 503)
BOOTSTRAP_SEED = 1207


def rule_directions(task, indicators, budget=Budget()):
    momentum = indicators["return_5"]
    trend = indicators["close"] / indicators["sma_20"] - 1
    threshold = budget.momentum_threshold
    directions = {"momentum_5": (momentum > threshold) - (momentum < -threshold),
                  "sma_20": (trend > 0) - (trend < 0), "always_hold": 0}
    for seed in RANDOM_SEEDS:
        key = f"{seed}:{task.task_id}".encode()
        directions[f"random_{seed}"] = random.Random(int.from_bytes(sha256(key).digest(), "big")).choice((-1, 1))
    if indicators["daily_volatility_20"] > budget.volatility_limit:
        return {name: 0 for name in directions}
    return directions


def score_offline(tasks, market, *, partition="offline"):
    if partition not in {"offline", "development"} or any(
        not task.task_id.startswith(f"v2-{partition}-") for task in tasks
    ):
        raise ValueError("Evaluation accepts only the specified frozen Offline Extended or Development partition")
    rows = []
    for task in tasks:
        row = {"task_id": task.task_id, "symbol": task.symbol, "decision_date": str(task.decision_date),
               "sector": ISSUERS[task.symbol][0]}
        try:
            indicators = market.view(task).indicators()
            directions = rule_directions(task, indicators)
            outcome = market.outcome(task)
        except (DataError, KeyError):
            rows.append({**row, "completed": False, "status": "unavailable_market_or_settlement", "policies": {}})
            continue
        policies = {
            name: {"direction": direction, "alpha": direction * (outcome.asset_return - outcome.benchmark_return),
                   "directional_hit": direction * outcome.asset_return > 0 if direction else None}
            for name, direction in directions.items()
        }
        rows.append({**row, "completed": True, "volatility_gate": indicators["daily_volatility_20"] > Budget().volatility_limit,
                     "settlement": outcome.model_dump(mode="json"), "policies": policies})
    return rows


def _score(rows, name):
    completed = [{**row["policies"][name], "decision_date": row["decision_date"]}
                 for row in rows if row["completed"]]
    active = [row for row in completed if row["direction"]]
    hits = sum(row["directional_hit"] for row in active)
    n = len(rows)
    return {
        "requested": n, "completed": len(completed), "task_completion_rate": len(completed) / n if n else None,
        "non_hold": len(active), "non_hold_coverage": len(active) / n if n else None,
        "directional_hits": hits, "directional_hit_rate": hits / len(active) if active else None,
        "mean_alpha": fmean(row["alpha"] for row in completed) if completed else None,
        "alpha_date_bootstrap_95": clustered_bootstrap(completed, BOOTSTRAP_SEED),
        "hit_date_bootstrap_95": clustered_bootstrap(
            [{"decision_date": row["decision_date"], "alpha": float(row["directional_hit"])} for row in active],
            BOOTSTRAP_SEED),
        "date_clusters": len({row["decision_date"] for row in completed}),
    }


def _quality(tasks, evidence, fundamentals, retriever):
    rows = []
    for task in tasks:
        cutoff = close_available(task.decision_date)
        view = evidence.view(task)
        filings, news = view.visible("sec_filing"), view.visible("news")
        metrics = FundamentalsView.load(fundamentals, task, PARTITION)
        by_period = {period: metrics.query(METRICS, period)["items"] for period in ("quarter", "ytd", "annual")}
        results = RagView(retriever, task, PARTITION, RagConfig(), provenance=True).search(
            "Recent earnings, financial results, risk and operating performance")
        rag = results["retrieval_trace"]
        supplied = [*filings, *news]
        violations = sum(item.eligible_at > cutoff or item.ticker != task.symbol for item in supplied)
        violations += sum(row["ticker"] != task.symbol or row["dataset_partition"] != PARTITION
            or datetime.fromisoformat(row["eligible_at"]) > cutoff for row in results["items"])
        violations += sum(row["ticker"] != task.symbol or datetime.fromisoformat(row["eligible_at"]) > cutoff
            for values in by_period.values() for row in values)
        rows.append({"task_id": task.task_id, "ticker": task.symbol, "decision_date": str(task.decision_date),
                     "sector": ISSUERS[task.symbol][0], "sec_candidates": len(filings), "news_candidates": len(news),
                     "xbrl_metrics": {period: [row["metric"] for row in values] for period, values in by_period.items()},
                     "rag_candidates": rag["eligible_chunks"], "rag_returned": len(results["items"]),
                     "rag_latency_ms": rag["latency_ms"], "pit_violations": violations})
    return rows


def quality_summary(rows):
    n = len(rows)
    return {
        "tasks": n, **{key: sum(row[key] > 0 for row in rows)
                       for key in ("sec_candidates", "news_candidates", "rag_candidates", "rag_returned")},
        "mean_rag_candidates": fmean(row["rag_candidates"] for row in rows),
        "pit_violations": sum(row["pit_violations"] for row in rows),
        "xbrl_fields": {
            period: {name: sum(name in row["xbrl_metrics"][period] for row in rows) for name in METRICS}
            for period in ("quarter", "ytd", "annual")
        },
        "issuer_coverage": {key: sorted({row["ticker"] for row in rows if row[key] > 0})
                            for key in ("sec_candidates", "news_candidates", "rag_candidates")},
    }


def acceptance_gates(sec: dict, coverage: dict) -> dict:
    targets = {
        "sec_issuers": (len(sec["documents_by_ticker"]), 20),
        "company_facts_issuers": (sum(entry["rows"] > 0 for entry in sec["company_facts"].values()), 20),
        "development_rag_tasks": (coverage["development"]["rag_candidates"], 72),
        "offline_rag_tasks": (coverage["offline"]["rag_candidates"], 180),
    }
    return {
        **{key: {"actual": actual, "minimum": target, "passed": actual >= target}
           for key, (actual, target) in targets.items()},
        "pit_violations": {"actual": sum(row["pit_violations"] for row in coverage.values()), "maximum": 0,
                           "passed": not any(row["pit_violations"] for row in coverage.values())},
    }


def evaluate(root: Path, tasks: Path, output: Path, models: Path, *, score_development=False):
    clock = perf_counter()
    if output.exists():
        raise FileExistsError(output)
    market_path = root / "market"
    offline = frozen_tasks(tasks, "offline", market_path)
    development = frozen_tasks(tasks, "development", market_path)
    market = HistoricalData(market_path)
    evidence_path = root / "evidence"
    evidence = EvidenceStore.load(evidence_path)
    index = root / "rag"
    index_manifest = json.loads((index / "manifest.json").read_text())
    if (index_manifest["partition"] != PARTITION
            or index_manifest["source_manifest_sha256"] != file_hash(evidence_path / "source_manifest.json")):
        raise ValueError("RAG source version is inconsistent")
    with ExitStack() as stack:
        encoder = LocalEmbedding(models, allow_download=False)
        store = stack.enter_context(VectorStore(index, encoder, "chunks"))
        retriever = Retriever(store)
        coverage = {group: _quality(group_tasks, evidence, root / "sec/fundamentals", retriever)
                    for group, group_tasks in (("offline", offline), ("development", development))}
    if any(row["pit_violations"] for group in coverage.values() for row in group):
        raise ValueError("Point-in-time leak detected")
    rows = score_offline(offline, market)
    policies = ["momentum_5", "sma_20", "always_hold", *(f"random_{seed}" for seed in RANDOM_SEEDS)]
    scores = {name: _score(rows, name) for name in policies}
    by_sector = {sector: {name: _score([row for row in rows if row["sector"] == sector], name) for name in policies}
                 for sector in sorted({row["sector"] for row in rows})}
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "offline-cases.jsonl", rows)
    for name, group in coverage.items():
        write_rows(output / f"{name}-coverage.jsonl", group)
    report = {
        "market_sha256": file_hash(market_path / "manifest.json"), "tasks_manifest_sha256": file_hash(tasks / "manifest.json"),
        "research_evidence_sha256": file_hash(evidence_path / "source_manifest.json"),
        "rag_manifest_sha256": file_hash(index / "manifest.json"),
        "fundamentals_manifest_sha256": file_hash(root / "sec/fundamentals/manifest.json"),
        "policy": "five sessions, next shared open to fifth close, direction × excess OHLC return vs SPY",
        "momentum_threshold": Budget().momentum_threshold, "volatility_limit": Budget().volatility_limit,
        "random_seeds": list(RANDOM_SEEDS), "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": 5000,
        "scores": scores, "by_sector": by_sector, "coverage": {name: quality_summary(group) for name, group in coverage.items()},
        "holdout_scored": 0, "glm_requests": 0, "glm_tokens": 0,
        "elapsed_seconds": round(perf_counter() - clock, 3),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if score_development:
        dev_rows = score_offline(development, market, partition="development")
        write_rows(output / "development-cases.jsonl", dev_rows)
        report["development_scores"] = {name: _score(dev_rows, name) for name in policies}
        report["development_by_sector"] = {sector: {
            name: _score([row for row in dev_rows if row["sector"] == sector], name) for name in policies
        } for sector in sorted({row["sector"] for row in dev_rows})}
        sec = json.loads((root / "sec/source_manifest.json").read_text())
        report["acceptance"] = acceptance_gates(sec, report["coverage"])
    write_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tasks", type=Path, default=Path("tasks/data-expansion-v2"))
    parser.add_argument("--models", type=Path, default=Path("data/local-models"))
    parser.add_argument("--output", type=Path, default=Path("results/data-expansion-v2-001"))
    parser.add_argument("--score-development", action="store_true", help="Score the 80 Development tasks separately")
    args = parser.parse_args()
    summary = evaluate(args.root, args.tasks, args.output, args.models, score_development=args.score_development)
    print({name: row["completed"] for name, row in summary["scores"].items()})
