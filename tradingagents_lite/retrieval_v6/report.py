"""Render the experiment's archived statistics without rerunning retrieval."""
import json
from html import escape
from pathlib import Path

PALETTE = ("#2563a7","#218373","#c57923","#8262a7")


def load(out,name):
    return json.loads((out/(name+".json")).read_text())


def pct(value):
    return f"{100*value:.2f}%"


def svg(out,name,title,subtitle,labels,series,scale=1.,unit="fraction",points=None):
    width,height=990,460
    x0,y0,pw,ph=82,82,830,280
    plot=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 990 460" role="img">',
          '<rect width="990" height="460" fill="#f9fafb"/>']
    def text(x,y,caption,size=14,anchor="start"):
        plot.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="{size}" '
                    f'text-anchor="{anchor}" fill="#213348">{escape(str(caption))}</text>')
    text(x0,33,title,23)
    text(x0,56,subtitle,12)
    for i in range(5):
        y=y0+ph*(1-i/4)
        plot.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="#d6dfe8"/>')
        val=i*scale/4
        text(x0-9,y+4,f"{val:.0%}" if unit=="fraction" else f"{val:.0f}",12,"end")
    if points:
        max_ms=max(item[1] for item in points)*1.15
        for i in range(5):
            text(x0+pw*i/4,y0+ph+21,round(max_ms*i/4),12,"middle")
        text(x0+pw/2,y0+ph+53,"Warm mean (ms)",13,"middle")
        for label,ms,quality in points:
            x=x0+pw*ms/max_ms
            y=y0+ph*(1-quality/scale)
            plot.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{PALETTE[0]}"/>')
            text(x+8,y-7,label,12)
    else:
        step=pw/len(labels)
        bar=step*.72/len(series)
        for i,label in enumerate(labels):
            x=x0+step*(i+.5)
            text(x,y0+ph+23,label,12,"middle")
            for j,(series_label,values) in enumerate(series):
                value=values[i]
                bh=ph*value/scale
                bx=x+(j-len(series)/2)*bar
                plot.append(f'<rect x="{bx:.1f}" y="{y0+ph-bh:.1f}" width="{bar-2:.1f}" height="{bh:.1f}" fill="{PALETTE[j]}">'
                            f'<title>{escape(series_label)} {value:.4f}</title></rect>')
        for j,(label,_) in enumerate(series):
            x=x0+205*j
            plot.append(f'<rect x="{x}" y="420" width="13" height="11" fill="{PALETTE[j]}"/>')
            text(x+19,430,label,12)
    plot.append("</svg>")
    (out/(name+".svg")).write_text("\n".join(plot)+"\n")


def figures(out,s,f):
    v=s["public_v6"]
    historical=s["public_baselines"]
    hist_by={**historical,"V6":v}
    labels=["R3","S1","S3","E3","V6"]
    recalls=[historical["R3"]["page"]["recall@5"],.41919191919191917,
             historical["S3"]["page"]["recall@5"],historical["E3"]["page"]["recall@5"],v["page"]["recall@5"]]
    svg(out,"historical-recall","Page Recall@5 across A1 experiments","FinanceBench public Test · 99 reused questions",
        labels,[("Recall@5",recalls)])
    labs=list(hist_by)
    svg(out,"oracle-gap","Candidate Oracle and final page Recall@5","Public Test · scores aggregated by question",labs,
        [("Oracle@5",[hist_by[k]["candidate_oracle_at_5"] for k in labs]),
         ("Recall@5",[hist_by[k]["page"]["recall@5"] for k in labs])])
    feat=load(out,"feature-ablation")["groups"]
    svg(out,"feature-ablation","Fixed-candidate financial feature ablation","49 Calibration · bounded raw-logit residual",
        [f"F{i}" for i in range(6)],[(arm,[feat[f"{arm}/F{i}-residual-none"]["page"]["recall@5"] for i in range(6)])
                                   for arm in ("S3","E3")])
    if (out/"query-expansion-ablation.json").exists():
        query=load(out,"query-expansion-ablation")["groups"]
        qnames=["Q0","Q1-100","Q2-100","Q3-100","Q3-150"]
        svg(out,"expansion-coverage","Query expansion and candidate coverage","49 Calibration · before and after cap",qnames,
            [("Before cap",[query[k]["pre_cap_oracle_at_5"] for k in qnames]),
             ("After cap",[query[k]["candidate_oracle_at_5"] for k in qnames])])
    same=s["deployment"]["arms"]
    points=[("S3",same["S3"]["warm"]["mean"],feat["S3/F0-residual-none"]["page"]["recall@5"]),
            ("V6",same[s["choice"]]["warm"]["mean"],s["calibration"]["page"]["recall@5"])]
    svg(out,"quality-latency","Quality and warm latency","Calibration quality (49 questions) · same-process timing (one per company)",
        [],[],points=points)
    names=["previously_top5","previously_6_10","previously_11_20","previously_21_plus","previously_absent"]
    moved=[f["public_e3_to_v6"].get(k,0) for k in names]
    svg(out,"gold-rank-migration","Gold pages entering V6 Top-5","Public Test · previous rank from archived E3",
        ["1–5","6–10","11–20","21+","Absent"],[("Gold pages",moved)],max(moved)*1.15 or 1,"count")
    failure=f["public_counts"]
    svg(out,"failure-distribution","Remaining public Test losses","Mutually exclusive gold-page counts across all 99 questions",
        ["Top-5","In pool, outside Top-5","Absent from pool"],
        [("Gold pages",[failure["A_top5"],failure["B_in_pool_not_top5"],failure["C_not_in_pool"]])],
        max(failure.values())*1.15,"count")


def generate(out):
    s=load(out,"summary")
    f=load(out,"failure-attribution")
    feat=load(out,"feature-ablation")
    expansion=load(out,"query-expansion-ablation") if (out/"query-expansion-ablation.json").exists() else None
    structure=load(out,"chunking-ablation") if (out/"chunking-ablation.json").exists() else None
    v=s["public_v6"]
    p=v["page"]
    e3=s["public_baselines"]["E3"]
    r5=p["recall@5"]
    oracle=v["candidate_oracle_at_5"]
    interval=s["paired"]["E3"]["bootstrap_95_ci"]
    ss=s["paired"]["S3"]
    cold=v["latency"]["cold_total_ms"]
    warm=v["latency"]["warm_total_ms"]
    bench=s["deployment"]
    timing=bench["arms"][s["choice"]]["warm"]["mean"]/bench["arms"]["S3"]["warm"]["mean"]
    rejection="达到" if r5>=.70 else "未达到"
    dep="通过" if bench["passes"] else "超过"
    lines=[
        "# A1-V6 Financial-aware Retrieval and Reranking Optimization","",
        f"**冻结配置 {s['choice']} 在 99 道公开 Test 的 Page Recall@5 为 {pct(r5)}，{rejection} 70% 目标。** "
        f"Candidate Oracle@5 为 {pct(oracle)}，与最终召回相差 {100*(oracle-r5):.2f} 个百分点。","",
        f"同进程 Calibration 延迟对照中，V6 的热均值为 S3 的 {timing:.2f} 倍，{dep}预注册的 1.25 倍部署门槛。"
        "生产替换还需要独立泛化验证。公开 99 题已多次用于 A1 历史实验，本轮在配置冻结后只评分一次。","",
        "## 校准与模块消融","",
        "同一套 49 题、11 家公司。所有规则都只读取问题与候选原文。留一公司选择用其余公司选择固定配置，"
        "再在被留出的公司评分。逐题结果保留全部固定配置和 OOF 选中记录。C1 上下文选择不变。","",
        f"- **Phase 0/1**。S3 与 E3 固定池分别为 {pct(feat['groups']['S3/F0-residual-none']['page']['recall@5'])} 和 "
        f"{pct(feat['groups']['E3/F0-residual-none']['page']['recall@5'])}。金融特征最高 {pct(feat['groups'][feat['best']]['page']['recall@5'])}"
        f"（{feat['best']}），该阶段 OOF 为 {pct(feat['oof']['page']['recall@5'])}。语义保底，年份来自正文且仅作软分。",
    ]
    if expansion:
        winner=expansion["groups"][expansion["best"]]
        lines.append(f"- **Phase 2**。{expansion['best']} 达到 {pct(winner['page']['recall@5'])}，Oracle 从 "
                     f"{pct(expansion['groups']['Q0']['candidate_oracle_at_5'])} 到 {pct(winner['candidate_oracle_at_5'])}。"
                     f"该阶段 OOF 为 {pct(expansion['oof']['page']['recall@5'])}。原问保留，两路以加权 RRF 合并后按 Chunk ID 去重。")
    if structure and "index" in structure:
        winner=structure["groups"][structure["best"]]
        ix=structure["index"]
        lines.append(f"- **Phase 3**。{ix['documents']} 份原 PDF 中重新解析 {ix['pages_reparsed']:,} 页，写入 {ix['indexed']:,} 个额外证据单元，"
                     f"其中 {ix['row']:,} 个财务行、{ix['block']:,} 个表块、{ix['verified_bindings']} 条经布局对齐的绑定。"
                     f"{structure['best']} 达到 {pct(winner['page']['recall@5'])}，Oracle {pct(winner['candidate_oracle_at_5'])}。"
                     "未识别或不适合绑定的内容保留旧 Chunk。表头验证只保证版面对应，无法审计所有财报会计语义。")
    lines.extend([
        f"- **Phase 4**。手工特征相对原排序最多只在 {s['ranking_model']['max_improved_companies']} 家公司获得正向净改善，未达拟合门槛。未训练 Learning-to-Rank。",
        f"- **冻结**。全量 Calibration 选择 {s['choice']}，召回 {pct(s['calibration']['page']['recall@5'])}，"
        f"全部候选共同参与的留一公司选择 OOF 为 {pct(s['out_of_fold']['page']['recall@5'])}。固定选择规则优先 Recall@5，再比较 Recall@1、MRR、财务行覆盖与候选规模。","",
        "## 冻结公开复评","",
        "| 配置 | Recall@1 | Recall@5 | Recall@10 | Oracle@5 | MRR@10 | nDCG@10 | Document Recall@5 | Top-5 唯一页 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name,item in [*s["public_baselines"].items(),("V6",v)]:
        page=item["page"]
        lines.append(f"| {name} | {pct(page['recall@1'])} | {pct(page['recall@5'])} | {pct(page['recall@10'])} | "
                     f"{pct(item['candidate_oracle_at_5'])} | {page['mrr@10']:.3f} | {page['ndcg@10']:.3f} | "
                     f"{pct(item['document']['recall@5'])} | {item['unique_pages_at_5_mean']:.2f} |")
    lines.extend(["",
        f"- **对 E3 的配对差**为 {100*s['paired']['E3']['mean_delta_recall_at_5']:+.2f} 个百分点，"
        f"公司聚类 95% CI [{100*interval[0]:+.2f}, {100*interval[1]:+.2f}]。"
        f"对 S3 为 {100*ss['mean_delta_recall_at_5']:+.2f} 点，改/降/平为 {ss['improved']}/{ss['regressed']}/{ss['tied']}，"
        f"95% CI [{100*ss['bootstrap_95_ci'][0]:+.2f}, {100*ss['bootstrap_95_ci'][1]:+.2f}]。",
        f"- **覆盖与排序**。相对 E3，Oracle 变化 {100*(oracle-e3['candidate_oracle_at_5']):+.2f} 点，最终召回变化 "
        f"{100*(r5-e3['page']['recall@5']):+.2f} 点。两者不同，前者度量候选潜力，后者取决于最终排序。"
        f"V6 的 Gold 页计数为 Top-5 {f['public_counts']['A_top5']}、池内未入 Top-5 {f['public_counts']['B_in_pool_not_top5']}、池外 {f['public_counts']['C_not_in_pool']}。",
        f"- **完整行**。V3 口径 Top-5 {v['line_proxies']['v3']['top5_complete']:,}/{v['line_proxies']['v3']['identified']:,}；"
        f"V5.1 风格严格口径 {v['line_proxies']['v51_style']['top5_complete']:,}/{v['line_proxies']['v51_style']['identified']:,}。"
        "严格口径从原 PDF 行筛除日期及标题并保留符号和标点，分母与历史压缩试验不同。两者均为代理。",
        f"- **V6 延迟**。冷 Mean/P50/P95 {cold['mean']:.0f}/{cold['p50']:.0f}/{cold['p95']:.0f} ms，"
        f"热 {warm['mean']:.0f}/{warm['p50']:.0f}/{warm['p95']:.0f} ms，峰值已分配显存 {v['peak_allocated_bytes']/2**30:.2f} GiB。"
        "冷测清除应用范围缓存，保留 OS 文件缓存；模型加载另记。冷热均属于同一次冻结运行，评分使用首轮排名。",
        "- **泛化**。没有冻结独立新题集。排名迁移和错误指标、期间或报表分类为启发式归因，不能当作人工审定的因果标签。",
        "",
        "## 边界与审计","",
        "Final Top-5 最多五个 Chunk。Recall@10、MRR、nDCG 使用前 10 仅作诊断。文档指标与页指标分开。"
        "检索使用冻结 BGE-small/BGE-M3 与相同 BGE Cross-Encoder。扩展用同一 Qdrant 持久向量的精确余弦实现，"
        "原始 Top-50 集合做逐题比对；分数相同或近似时的顺序差异见 baseline-audit。"
        "两个历史索引只读；结构证据保存源文档哈希、0 基页与原文跨度。","",
        "真实 GLM 请求 **0**，Token **0**。未访问 Private Test Gold 或 V3 Holdout。"
        "下一步按项目计划转向 Agent 端到端验证。","",
        "[复现命令](COMMANDS.md) · [测试报告](TESTS.md) · [冻结清单](freeze.json) · [基线审计](baseline-audit.json)","",
    ])
    figures(out,s,f)
    for name,label in [("historical-recall","历代召回"),("oracle-gap","Oracle 与最终召回"),
                       ("feature-ablation","金融特征消融"),("expansion-coverage","扩展覆盖"),
                       ("quality-latency","质量与延迟"),("gold-rank-migration","排名迁移"),
                       ("failure-distribution","损失归因")]:
        if (out/(name+".svg")).exists():
            lines.extend([f"![{label}]({name}.svg)",""])
    (out/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
