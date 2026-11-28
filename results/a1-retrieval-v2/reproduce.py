"""Export the public retrieval results and chart from a frozen local run."""
import argparse
from pathlib import Path
import shutil
from statistics import mean

from tradingagents_lite.retrieval_v2 import (
    BASE, DEPTHS, Ranking, load_split, measure, read_json, read_rows, write_json,
)


def depth_timing(source, run):
    calibration = read_json(run / "calibration.json")
    result = {"24": next(row["latency"] for row in calibration["measured"] if row["label"] == BASE.label)}
    for depth in (50, 100):
        _, stats = measure(source, run, run / "original-index", Ranking(candidate_k=depth),
                           load_split(source, "calibration"), f"cal-depth-{depth}")
        result[str(depth)] = stats["latency"]
    write_json(run / "depth-timing.json", result)


def diagnostics(rows):
    gold = sum(row["gold_pages"] for row in rows)
    pages = [p for row in rows for p in row["line_status"]]
    absent, candidates, fusion, retrieved = [], [], [], []
    for row in rows:
        if row["legal_gold_pages"] != row["gold_pages"]:
            absent.append(row["id"])
        if row["candidate_recall"]["24"]["union"] < 1:
            candidates.append(row["id"])
        if row["rrf_gold_recall_at_5"] < row["candidate_recall"]["24"]["union"]:
            fusion.append(row["id"])
        if row["rrf_gold_recall_at_5"] == 1:
            retrieved.append(row["id"])
    return {
        "legal_gold_pages": sum(row["legal_gold_pages"] for row in rows), "gold_pages": gold,
        "gold_pages_not_indexed_ids": absent, "incomplete_candidate_union_at_24_ids": candidates,
        "fusion_demoted_at_least_one_gold_id": fusion, "all_gold_in_top_5_ids": retrieved,
        "top_5_duplicate_chunks": sum(row["duplicate_page_chunks_at_5"] for row in rows),
        "mean_top_5_duplicate_chunks": mean(row["duplicate_page_chunks_at_5"] for row in rows),
        "candidate_recall": {str(k): {arm: mean(r["candidate_recall"][str(k)][arm] for r in rows)
                                     for arm in ("bm25", "dense", "union")} for k in DEPTHS},
        "metadata_line_totals": {key: sum(page.get(key, 0) for page in pages) for key in
                                 ("identifiable_numeric_lines", "complete_in_legacy_chunk",
                                  "years_observed", "years_colocated", "units_observed", "units_colocated")},
        "line_proxy_top5": {k: sum(row[k] for row in rows) for k in rows[0]
                            if k.startswith("observable_lines_at_5__")},
    }


def plot(test, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    groups = [(("E0" if label == "E0" else "A1-V2"), group) for label, group in test["groups"].items() if "alias_of" not in group]
    colors = ("#677383", "#136b64", "#ac510f")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "a1-retrieval-v2"})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")
    fig.suptitle("FinanceBench public test · 99 questions", fontsize=16, fontweight="bold")
    for i, (label, group) in enumerate(groups):
        for ax, title, values, names in (
            (axes[0], "Page recall", [group["page"][f"recall@{k}"] for k in (1, 5, 10)], ("@1", "@5", "@10")),
            (axes[1], "Cold retrieval latency", [group["latency"]["cold"]["total_ms"][s] for s in ("mean", "p50", "p95")],
             ("Mean", "P50", "P95")),
            (axes[2], "Hot retrieval latency", [group["latency"]["hot"]["total_ms"][s] for s in ("mean", "p50", "p95")],
             ("Mean", "P50", "P95")),
        ):
            width = .7 / len(groups)
            bars = ax.bar([x + (i - (len(groups) - 1) / 2) * width for x in range(3)], values,
                          width=width * .9, label=label, color=colors[i])
            formatted = [f"{v:.1%}" if ax is axes[0] else f"{v:.0f}" for v in values]
            ax.bar_label(bars, labels=formatted, fontsize=8, padding=3)
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_xticks(range(3), names)
            ax.grid(axis="y", alpha=.25)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.margins(y=.22)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes[1:]:
        ax.set_ylabel("milliseconds")
    axes[0].legend(loc="upper left", frameon=False, ncols=len(groups))
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    lines = path.read_text().splitlines()
    path.write_text(chr(10).join(line.rstrip() for line in lines) + chr(10))


def export(run, out):
    calibration = read_json(run / "calibration.json")
    test = read_json(run / "test/summary.json")
    groups = test["groups"]
    summary = {
        "calibration": calibration, "test": test,
        "calibration_diagnostics": diagnostics(read_rows(run / "calibration-audit.jsonl")),
        "candidate_depth_timing": read_json(run / "depth-timing.json"),
        "glm_requests": 0, "glm_tokens": 0,
    }
    write_json(out / "summary.json", summary)
    shutil.copyfile(run / "freeze.json", out / "freeze.json")
    shutil.copyfile(run / "calibration-audit.jsonl", out / "calibration-audit.jsonl")
    for label, group in groups.items():
        if "alias_of" not in group:
            shutil.copyfile(run / "test" / f"{label}.jsonl", out / f"test-{label}.jsonl")
    plot(test, out / "recall-latency.svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("depth-timing", "export"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.command == "depth-timing":
        if args.source is None:
            parser.error("depth-timing requires --source")
        depth_timing(args.source.resolve(), args.run.resolve())
    else:
        export(args.run.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
