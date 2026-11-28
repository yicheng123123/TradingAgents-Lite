"""Publish detached, redacted archives from a completed A1-V3 run."""
import argparse
from pathlib import Path
import shutil

from tradingagents_lite.data import sha256
from tradingagents_lite.retrieval_v2 import bootstrap, read_json, read_rows, write_json

TRACE_COLUMNS = ("chunk_hash", "bm25_rank", "dense_rank", "reference_rrf_rank", "rerank_score", "final_rank")


def chart(calibration, test, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "talite-a1-retrieval-v3"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    colors = {"S0": "#657189", "S1": "#8f6b20", "S2": "#55649a",
              "S3": "#33785d", "S4": "#236c76", "S5": "#965068"}
    for label, ax, groups in zip(("Calibration · 49 questions", "Public Test · 99 questions"),
                                 axes, (calibration, test), strict=True):
        for name, group in groups.items():
            ms = group["latency"]["hot"]["total_ms"]["mean"]
            recall = group["page"]["recall@5"]
            ax.scatter(ms, recall, s=72, color=colors[name], zorder=3)
            offsets = {"S0": (-5, -13), "S1": (6, 6), "S2": (6, 6), "S3": (-10, 9),
                       "S4": (7, -11), "S5": (7, 8)}
            ax.annotate(name, (ms, recall), xytext=offsets[name], textcoords="offset points", fontsize=10)
        e0_cutoff = 2 * groups["S0"]["latency"]["hot"]["total_ms"]["mean"]
        ax.axvline(e0_cutoff, color="#6d737d", linewidth=1, linestyle="--", label="2× E0 latency")
        ax.axhline(.50, color="#9ca3af", linewidth=1, linestyle=":", label="50% recall goal")
        ax.set_title(label, loc="left")
        ax.set_xlabel("Mean hot retrieval latency (ms)")
        ax.set_ylabel("Page Recall@5")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.2)
        ax.margins(x=.12, y=.25)
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def export(run, out):
    freeze = read_json(run / "freeze.json")
    cal = read_json(run / "calibration/summary.json")
    test = read_json(run / "test/summary.json")
    checks = read_json(run / "test/checks.json")
    if checks["freeze_sha256"] != sha256(run / "freeze.json"):
        raise ValueError("Formal test does not match frozen configuration")
    outputs = {
        "freeze.json": run / "freeze.json",
        "probe.json": run / "probe.json",
    }
    for split, summary in (("calibration", cal), ("test", test)):
        for name in (*summary, "ranking-trace"):
            outputs[f"{split}-{name}.jsonl"] = run / split / f"{name}.jsonl"
    paired_to_v2 = {arm: bootstrap(read_rows(run / "test" / f"{arm}.jsonl"),
                                   read_rows(run / "test/S1.jsonl")) for arm in test}
    out.mkdir(parents=True, exist_ok=True)
    for name, source in outputs.items():
        shutil.copyfile(source, out / name)
    write_json(out / "summary.json", {
        "calibration": cal, "test": test, "checks": checks,
        "quality_winner": freeze["quality_winner"],
        "calibration_deploy_candidate": freeze["calibration_deploy_candidate"],
        "s5_base": freeze["s5_base"], "s5_trigger": freeze["s5_trigger"],
        "trace_candidate_columns": TRACE_COLUMNS,
        "paired_to_A1_V2": paired_to_v2,
        "artifact_sha256": {name: sha256(out / name) for name in outputs},
        "glm_requests": 0, "glm_tokens": 0,
    })
    chart(cal, test, out / "quality-latency.svg")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    export(args.run, args.out)


if __name__ == "__main__":
    main()
