#!/usr/bin/env python3
"""Render A1-V5 report and SVG plots from archived JSON using only stdlib."""
import html
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def gkey(arm, budget, lam=None):
    label = f"{arm}|lambda={lam}" if lam is not None else arm
    return f"{label}|budget={budget}"


def pct(x):
    return "Unavailable" if x is None else f"{x:.1%}"


def number(x):
    return "Unavailable" if x is None else f"{x:.1f}"


def chart(filename, title, subtitle, xlabels, series, source):
    width, height = 1040, 510
    left, top, pw, ph = 84, 105, 880, 300
    colors = ["#748094", "#38BDF8", "#FBBF24", "#A78BFA", "#34D399", "#F87171"]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="510" viewBox="0 0 1040 510">',
           '<rect width="100%" height="100%" fill="#0B101A"/>',
           f'<text x="{left}" y="42" fill="#E6EDF7" font-family="sans-serif" font-size="22" font-weight="700">{html.escape(title)}</text>',
           f'<text x="{left}" y="68" fill="#94A3B8" font-family="sans-serif" font-size="13">{html.escape(subtitle)}</text>']
    for t in range(5):
        val = t / 4
        y = top + ph * (1 - val)
        svg += [f'<line x1="{left}" y1="{y}" x2="{left+pw}" y2="{y}" stroke="#273244"/>',
                f'<text x="{left-12}" y="{y+4}" text-anchor="end" fill="#94A3B8" font-family="monospace" font-size="12">{val:.0%}</text>']
    svg += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#536174"/>',
            f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#536174"/>']
    xs = [left, left + pw]
    for x, label in zip(xs, xlabels, strict=True):
        svg.append(f'<text x="{x}" y="{top+ph+24}" text-anchor="middle" fill="#94A3B8" font-family="monospace" font-size="12">{html.escape(label)}</text>')
    svg.append(f'<text x="{left+pw/2}" y="{top+ph+47}" text-anchor="middle" fill="#94A3B8" font-family="sans-serif" font-size="13">Context budget / first-turn capacity (bytes)</text>')
    for i, (label, values) in enumerate(series):
        color = colors[i % len(colors)]
        pts = []
        for x, val in zip(xs, values, strict=True):
            if val is not None:
                y = top + ph * (1 - max(0, min(1, val)))
                pts.append((x, y))
        if len(pts) == 2:
            svg.append(f'<polyline points="{pts[0][0]},{pts[0][1]} {pts[1][0]},{pts[1][1]}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in pts:
            svg.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}"/>')
        col, row = i % 3, i // 3
        lx, ly = left + col * 300, 472 + row * 18
        svg += [f'<line x1="{lx}" y1="{ly-4}" x2="{lx+20}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{lx+27}" y="{ly}" fill="#C8D3E2" font-family="sans-serif" font-size="11">{html.escape(label)}</text>']
    svg += [f'<text x="{left}" y="504" fill="#64748B" font-family="sans-serif" font-size="10">Source: {html.escape(source)}; archived metrics, no synthetic datapoints</text>', "</svg>"]
    (OUT / filename).write_text("\n".join(svg) + "\n", encoding="utf-8")


def render_charts(cal, test, freeze):
    csel = cal["selected"]
    entries = [("C0 · raw prefix", "C0", None), ("C1 · query-aware", "C1", None),
               ("C2 · BGE-M3 extractive", "C2", None),
               (f"C3 · MMR λ={csel['best_c3_mmr_lambda']:.2f}", "C3", csel["best_c3_mmr_lambda"]),
               (f"C4 · header/global λ={csel['best_c4_mmr_lambda']:.2f}", "C4", csel["best_c4_mmr_lambda"])]
    for metric, title, filename in [("delivered_page_recall_mean", "Calibration · delivered page recall", "quality-budget.svg"),
                                    ("complete_financial_row_coverage_micro", "Calibration · full financial-row coverage", "quality-budget-rows.svg")]:
        series = [(name, [cal["groups"][gkey(arm, b, lam)].get(metric) for b in (1600, 3200)])
                  for name, arm, lam in entries]
        chart(filename, title, "49 Calibration questions · fixed S3 Top-5 · 35% first-turn reserve",
              ["1,040", "2,080"], series, "calibration-summary.json")
    cfg = freeze["selected_configuration"]
    tg = test["groups"]
    series = []
    for metric, name in [("delivered_page_recall_mean", "Delivered page recall"),
                         ("complete_financial_row_coverage_micro", "Complete row coverage")]:
        series.append((name + " · C1", [tg[gkey("C1", b)].get(metric) for b in (1600, 3200)]))
        series.append((name + " · frozen " + cfg["arm"], [tg[gkey(cfg["arm"], b, cfg.get("mmr_lambda"))].get(metric) for b in (1600, 3200)]))
    chart("frozen-test-comparison.svg", "Frozen public Test · quality vs budget",
          "99 public questions · same S3 Top-5 candidates",
          ["1,600", "3,200"], series, "test-summary.json")


def quality_rows(groups, entries, budget):
    lines = ["| Variant | Delivered page | Complete rows / total | Year | Values | Units | Header bound / attempts | Citation | Source exact |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, arm, lam in entries:
        g = groups[gkey(arm, budget, lam)]
        complete = f"{g.get('financial_rows_covered')}/{g.get('financial_rows_total')} ({pct(g.get('complete_financial_row_coverage_micro'))})"
        attempts = g.get("header_binding_attempts", 0)
        accuracy = g.get("header_binding_accuracy")
        bound = round(accuracy * attempts) if accuracy is not None else 0
        binding = f"{bound}/{attempts} ({pct(accuracy)})" if attempts else "0/0 (N/A)"
        vals = [pct(g.get("delivered_page_recall_mean")), complete,
                pct(g.get("year_retention_micro")), pct(g.get("value_retention_micro")),
                pct(g.get("unit_retention_micro")), binding,
                pct(g.get("citation_integrity_rate")), pct(g.get("source_fact_consistency_rate"))]
        lines.append("| " + name + " | " + " | ".join(vals) + " |")
    return lines

def cost_rows(groups, entries, budget):
    lines = ["| 配置 | 平均上下文 bytes | 平均请求 / 最大请求 bytes | 来源压缩比 | BGE tokenizer 代理 tokens | 选择均值 / P95 ms | embedding 均值 ms | 超预算题数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, arm, lam in entries:
        g = groups[gkey(arm, budget, lam)]
        lines.append(f"| {name} | {number(g.get('mean_context_bytes'))} | {number(g.get('mean_request_bytes'))} / {g.get('max_request_bytes_observed')} | {pct(g.get('mean_source_compression_ratio'))} | {number(g.get('mean_proxy_evidence_tokens'))} | {number(g.get('selection_latency_mean_ms'))} / {number(g.get('selection_latency_p95_ms'))} | {number(g.get('embedding_latency_mean_ms'))} | {g.get('budget_failures')} |")
    return lines


def render_report(cal, test, freeze):
    sel = cal["selected"]
    cfg = freeze["selected_configuration"]
    c3, c4 = sel["best_c3_mmr_lambda"], sel["best_c4_mmr_lambda"]
    cal_entries = [("C0 raw prefix", "C0", None), ("C1 current query-aware", "C1", None),
                   ("C2 BGE-M3 extractive", "C2", None), (f"C3 λ={c3:.2f}", "C3", c3),
                   (f"C4 λ={c4:.2f}", "C4", c4)]
    test_entries = [("C1 current query-aware", "C1", None),
                    (f"Frozen {cfg['arm']} λ={cfg.get('mmr_lambda'):.2f}", cfg["arm"], cfg.get("mmr_lambda"))]
    c1 = test["groups"][gkey("C1", 3200)]
    new = test["groups"][gkey(cfg["arm"], 3200, cfg.get("mmr_lambda"))]
    model = test["model"]
    lines = [
        "# A1-V5：Financial RAG Context Compression", "",
        "## 结论", "",
        f"- Calibration 选择的新配置为 **{cfg['arm']}（MMR λ={cfg.get('mmr_lambda'):.2f}）**；按相同质量排序，C1 仍是全组综合赢家。",
        f"- 固定 S3 Top-5 在公开 Test 的页级检索 Recall@5 为 **{pct(c1['retrieval_recall_at_5_mean'])}**。所有压缩组复用相同候选，检索 Recall 未改变。",
        f"- 3200-byte Test：C1 送达页召回 {pct(c1['delivered_page_recall_mean'])}、完整财务行覆盖 {pct(c1['complete_financial_row_coverage_micro'])}；C4 分别为 {pct(new['delivered_page_recall_mean'])}、{pct(new['complete_financial_row_coverage_micro'])}。",
        "- C4 减少了上下文用量，但页级证据、完整财务行、年份、数值、单位保留都低于 C1。本轮没有证据支持替换现有压缩器，保留负结果。",
        "", "## Calibration：49 题，固定 A1-V3 S3 Top-5", "",
        "选择顺序：完整财务行覆盖、送达页召回、原文一致、表头绑定、延迟。答案和官方证据只在评分阶段使用。", "",
        "### 3200-byte 主预算：质量", "",
    ]
    lines += quality_rows(cal["groups"], cal_entries, 3200)
    lines += ["", "### 3200-byte 主预算：上下文与耗时", ""]
    lines += cost_rows(cal["groups"], cal_entries, 3200)
    lines += ["", "### 1600-byte 压力预算：C1 与最佳新配置", ""]
    lines += quality_rows(cal["groups"], [cal_entries[1], ("Frozen-new " + cfg["arm"], cfg["arm"], cfg.get("mmr_lambda"))], 1600)
    lines += [""]
    lines += cost_rows(cal["groups"], [cal_entries[1], ("Frozen-new " + cfg["arm"], cfg["arm"], cfg.get("mmr_lambda"))], 1600)
    lines += ["", "## 冻结公开 Test：99 题", "",
              "公开 FinanceBench Test 已在此前 A1 迭代多次复评，不是独立未见集。本轮配置冻结后只评测一次，99 题全数留在分母。", "",
              "### 3200-byte 主预算：质量", ""]
    lines += quality_rows(test["groups"], test_entries, 3200)
    lines += ["", "### 3200-byte 主预算：上下文与耗时", ""]
    lines += cost_rows(test["groups"], test_entries, 3200)
    lines += ["", "### 1600-byte 压力预算", ""]
    lines += quality_rows(test["groups"], test_entries, 1600)
    lines += [""]
    lines += cost_rows(test["groups"], test_entries, 1600)
    lines += ["", "## 方法、冻结与范围", "",
              f"- BGE-M3 revision {model['revision']}；1024 维 dense-only、CLS pooling、L2 normalization、FP16 CUDA；设备 {model['gpu']}；输入上限 {model['input_limit']} tokens。",
              "- C0 原文顺序前缀；C1 原 FinanceBench V2 Query-aware selection（CaseSearch._deliver）；C2 语义提取；C3 加 MMR；C4 加表头绑定与跨候选全局预算。",
              "- 每题复用原问题、S3 查询和同一冻结 Top-5 chunk；无新检索、无 reranker 改动。所有事实 span 均可回溯到原文，document/page/evidence 引用完整。",
              "- 主预算 3200 bytes，35% 首轮预留后 2080 bytes；压力预算 1600 bytes，首轮有效 1040 bytes；完整请求最多 8000 bytes。",
              "- tokenizer 数量是 BGE-M3 代理，不代表真实 GLM token 节省。Top-10 按预设门槛跳过：Calibration 仅节省约 12 bytes，低于 100 bytes。",
              "- GLM 请求 0、GLM Token 0；无训练、无模型下载；未访问 Private Test Gold 或 V3 Holdout；原索引只读。",
              "- Freeze SHA256 " + test["freeze_sha256"] + "。", "",
              "## 产物", "",
              "- summary.json：各组聚合指标；freeze.json：配置与数据/代码指纹。",
              "- calibration-rows.jsonl 与 test-rows.jsonl：脱敏逐题指标，无原文、公司名或来源 URL。",
              "- 图表由真实归档 JSON 渲染：quality-budget.svg、quality-budget-rows.svg、frozen-test-comparison.svg。",
              "- COMMANDS.md 包含离线复现命令；render_artifacts.py 使用 Python 标准库生成图表和本报告。",
              "", "## 测试与限制", "",
              "- 99 题评测与所有 396 条逐题预算/配置行均已落盘；内置 matplotlib 导出因环境缺包退出，随后由标准库 renderer 从同一 JSON 生成报告与 SVG，未重跑或改写评测。",
              "- 新增不变量测试 7 passed；不含数据归档用例的安全回归子集 147 passed、2 skipped。",
              "- 完整 pytest：239 passed、2 skipped、15 failed、13 errors。失败/错误来自隔离 worktree 缺少 Yahoo 与 counterfactual development 测试 manifests；未补拷受保护数据。",
              "- 财务行覆盖是基于公开证据全文/摘录的字符串匹配代理；压缩质量不能直接推断真实模型 QA 准确率。",
              "- 当前 C2/C3/C4 没有胜过 C1；真实 GLM 验证需独立预算授权。", "",
              "## 图表", "",
              "![Calibration delivered page recall](quality-budget.svg)", "",
              "![Calibration financial row coverage](quality-budget-rows.svg)", "",
              "![Frozen public Test comparison](frozen-test-comparison.svg)", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    cal, test, freeze = load("calibration-summary.json"), load("test-summary.json"), load("freeze.json")
    render_charts(cal, test, freeze)
    render_report(cal, test, freeze)
    print("Rendered REPORT.md and three SVG charts from archived summaries (standard library only).")


if __name__ == "__main__":
    main()
