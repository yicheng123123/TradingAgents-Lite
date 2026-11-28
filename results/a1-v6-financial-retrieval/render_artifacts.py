"""Render the frozen results with the V6 unique-page field alias."""
import argparse
from pathlib import Path

from tradingagents_lite.retrieval_v6 import report

read_archive = report.load


def compatible_archive(out, name):
    archive = read_archive(out, name)
    if name == "summary":
        public = archive["public_v6"]
        public["unique_pages_at_5_mean"] = public["unique_pages_at_5"]
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    out = parser.parse_args().output
    report.load = compatible_archive
    report.generate(out)

    source = read_archive(out, "summary")
    expansion = read_archive(out, "query-expansion-ablation")["groups"]
    fixed = read_archive(out, "feature-ablation")["groups"]
    archive = out / "REPORT.md"
    text = archive.read_text(encoding="utf-8")
    stage = expansion["Q3-150/BGE"]["page"]["recall@5"]
    winner = expansion[source["choice"]]["page"]["recall@5"]
    raw = expansion["Q0"]["page"]["recall@5"]
    fixed_raw = fixed["E3/F0-residual-none"]["page"]["recall@5"]
    top_fixed = max(row["page"]["recall@5"] for key,row in fixed.items() if key.startswith("E3/"))
    details = (
        f"扩展同池对照中，原 BGE 从 {report.pct(raw)} 变为 {report.pct(stage)}，金融重排将其提高至 "
        f"{report.pct(winner)}，是本轮最大的最终召回改善。固定 E3 候选内，金融特征只从 "
        f"{report.pct(fixed_raw)} 提高到 {report.pct(top_fixed)}。"
        "结构索引扩大了覆盖，但扩大覆盖的部分仍需要更好的排序才能进入前 5。\n\n"
        "保留当前 S3。V6 可作为后续端到端验证的研究候选，现有 S3/E3 的配对区间与独立泛化证据仍不足以支持替换。"
        "延迟门槛是相对的，基于每公司一题的顺序测量，实际部署还需按业务服务目标验证 P95。\n\n"
        "首次冻结预检在加载公开题前报错，保留 freeze-preflight.json。"
        "正式 99 题预测保存后，旧 V3 汇总字段不兼容，由标准化导出脚本从原预测恢复。"
        "两次修复均未更改选定配置或重复运行公开检索。\n\n"
    )
    archive.write_text(text.replace("## 冻结公开复评\n\n",details+"## 冻结公开复评\n\n",1),encoding="utf-8")


if __name__=="__main__":
    main()
