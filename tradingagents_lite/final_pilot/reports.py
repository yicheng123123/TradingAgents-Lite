"""Render archived Pilot metrics to standalone SVG and Markdown."""
from html import escape

ARMS=("T0","T1","T2","T3","T4")
COLORS=("#2171a6","#289482","#b46736")


def value(number,digits=3):
    return "不可用" if number is None else f"{number:.{digits}f}"


def plot(out,name,title,labels,series,unit="",scatter=None):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="910" height="430" viewBox="0 0 910 430" role="img">',
         '<rect width="910" height="430" fill="#f7f9fc"/>']
    def text(x,y,s,anchor="start",size=13):
        svg.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}"'
                   f' font-family="sans-serif" fill="#19344b">{escape(str(s))}</text>')
    text(70,32,title,size=21)
    text(70,53,"V3 Development Pilot · frozen 20 tasks per arm · actual recorded results",size=11)
    vals=([p[2] for p in scatter] if scatter else [v for _,group in series for v in group if v is not None])
    if not vals:
        return
    low=min(0,min(vals)); high=max(0,max(vals))
    if low==high:
        high=1
    extent=(high-low)*1.10
    high=low+extent
    def y(v):
        return 315-235*(v-low)/extent
    for i in range(5):
        v=low+extent*i/4
        yp=y(v)
        svg.append(f'<line x1="78" x2="856" y1="{yp:.1f}" y2="{yp:.1f}" stroke="#d1dce7"/>')
        text(72,yp+4,f"{v:.2g}{unit}","end",11)
    if scatter:
        max_x=max(p[1] for p in scatter)*1.10 or 1
        for i in range(5):
            text(80+775*i/4,337,round(max_x*i/4),"middle",11)
        for label,x,v in scatter:
            xp=80+775*x/max_x
            svg.append(f'<circle cx="{xp:.1f}" cy="{y(v):.1f}" r="5" fill="{COLORS[0]}"/>')
            text(xp+7,y(v)-7,label,size=11)
        text(466,365,"Charged tokens per task","middle")
    else:
        step=775/len(labels)
        width=step*.70/len(series)
        for pos,label in enumerate(labels):
            mid=80+(pos+.5)*step
            text(mid,339,label,"middle",11)
            for ix,(legend,group) in enumerate(series):
                v=group[pos]
                if v is None:
                    continue
                x=mid+(ix-len(series)/2)*width
                svg.append(f'<rect x="{x:.1f}" y="{min(y(v),y(0)):.1f}" '
                           f'width="{width-2:.1f}" height="{max(1,abs(y(v)-y(0))):.1f}" fill="{COLORS[ix]}">'
                           f'<title>{escape(legend)} {v:.5g}</title></rect>')
        for ix,(legend,_) in enumerate(series):
            x=80+230*ix
            svg.append(f'<rect x="{x}" y="367" width="12" height="9" fill="{COLORS[ix]}"/>')
            text(x+18,376,legend,size=11)
    text(70,410,"Source: summary.json and paired-ablation.json",size=11)
    svg.append("</svg>")
    (out/(name+".svg")).write_text("\n".join(svg)+"\n",encoding="utf-8")


def generate(out,report):
    groups=report["groups"]
    cap=report["budget"]
    snapshot=cap["pilot"]
    cfg=report["configuration"]
    selected=report["selected"]
    lines=[
        "# Phase 3.0–3.1 Final Agent Pilot","",
        f"集成配置 **{cfg.get('retriever') or '未冻结'} / {cfg.get('context') or '未冻结'}**。"
        f"完成预测归档 {report['observed']}/100 个实例，真实模型实例 {report['glm_instances']}/80。","",
        f"真实 BigModel HTTP **{cap['actual_b2_requests']}** 次。账本实际计费/保守记账 **{cap['actual_glm_tokens']:,}** Tokens，"
        f"其中有可用供应商用量 **{snapshot['usage_known_tokens']:,}**。B2 余 **{snapshot['phase_remaining']:,}**，"
        f"项目余 **{snapshot['global_remaining']:,}**。","",
    ]
    if selected:
        lines += [f"按预注册准入，推荐进入后续 Development 的配置为 **{selected}**。",""]
    else:
        stop=report.get("stop")
        lines += ["**正式 Pilot 未全部完成，结算仍封存。** 因此没有配对 Alpha 或正式配置选择。",""]
        if stop:
            lines += [f"执行级停止原因为 **{stop['fault']}**，保存已完成检查点及账本。",""]
        elif cap.get("smoke") and not cap["smoke"]["passed"]:
            lines += ["真实 Smoke 未通过路径或预算检查，正式运行保持关闭。",""]
    lines += ["## 各组结果","",
              "| 组 | 完成/20 | 非 HOLD | 条件方向命中 | Mean / Median Alpha | 引用合法 | Tokens | Mean / P95 s |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in ARMS:
        g=groups[key]
        lines.append(f"| {key} | {g['ok']}/20 | {g['nonhold_coverage']:.0%} | "
                     f"{value(g['conditional_direction_accuracy'])} ({g['direction_denominator']}) | "
                     f"{value(g['mean_alpha'],5)} / {value(g['median_alpha'],5)} | "
                     f"{g['citation_task_passed']}/{g['citation_task_denominator']} | {g['total_tokens']:,} | "
                     f"{value(g['latency_mean_seconds'],2)} / {value(g['latency_p95_seconds'],2)} |")
    lines += ["","Alpha 使用方向调整后的股票相对 SPY 五日价格收益。任务失败保留在分母，条件方向准确率只统计已结算的非 HOLD。"
              "这里没有模拟交易成本、可执行组合或实际投资收益。引用合法性仅检验当前范围和可用性。",""]
    if selected:
        lines += ["## 配对与准入",""]
        for key,info in report["pairs"].items():
            interval=info["bootstrap_95_ci"]
            ci=(", ".join(f"{x:+.5f}" for x in interval)) if interval else "不适用"
            lines.append(f"- {key} 配对 {info['paired']}/20，Mean Alpha 差 {value(info['mean_alpha_difference'],5)}，"
                         f"95% 区间 [{ci}]（{info['time_blocks']} 个时间块）。")
        lines += [""]
        for key,criteria in report["gate"].items():
            failed=[name for name,passed in criteria.items() if not passed]
            lines.append(f"- {key} "+("通过全部准入条件。" if not failed else "未通过 "+", ".join(failed)+"。"))
        remainder=cap["remaining_development"]
        fit="足够" if remainder["estimated_fit"] else "不足"
        lines += ["",f"后续 60 题按实测均值估算 **{remainder['expected_60_observed_mean']:,.0f}** Tokens，"
                   f"当前 B2 账面按该估算{fit}。双倍压力估算 {remainder['two_times_observed_mean']:,.0f} Tokens。"]
    else:
        lines += ["未完成配对，不能判断 RAG、Memory 或 Debate 是否改善决策，也不能从本轮 Smoke 选择 Development 配置。"]
    t3,t4=groups["T3"],groups["T4"]
    lines += ["","## Memory 与 Debate","",
              f"T3、T4 实际注入经验的任务分别为 {t3['memory_used_tasks']}/{t3['observed']} 与 "
              f"{t4['memory_used_tasks']}/{t4['observed']}。两组注入共 {t3['memory_injected_total']+t4['memory_injected_total']} 条。"
              "候选稀缺、空记忆与模型实际曝光分开计数。Warm-up 来自规则策略和成熟的独立历史结算。","",
              f"T4 触发 {t4['debate_trigger_count']} 次，两角色成功 {t4['debate_completed_roles']} 次，"
              f"动作冲突 {t4['debate_conflict_count']} 次。角色和裁决共增加 {t4['debate_tokens']:,} Tokens、"
              f"{t4['debate_latency_seconds']:.2f} 秒。跨组动作差属于整个分支的观察对比。","",
              "该 Pilot 的小样本和可能的模型预训练污染限制推断。预测隔离只约束可见输入与本地工具，"
              "不能证明模型没有见过公开金融事件。Holdout 继续封存。","",
              "[冻结配置](freeze.json) · [协议修订](protocol-amendment.md) · [预算](budget-summary.json) · "
              "[测试](TESTS.md) · [复现](COMMANDS.md)",""]
    plots=[
        ("direction","Conditional direction accuracy","conditional_direction_accuracy",""),
        ("mean-alpha","Mean direction-adjusted relative alpha","mean_alpha",""),
        ("tokens","Charged tokens by arm","total_tokens",""),
        ("completion","Task completion rate","completion_rate",""),
        ("memory","Actual memory use over planned tasks","memory_use_rate",""),
        ("debate","Debate triggers over planned tasks","debate_trigger_rate",""),
    ]
    for name,title,field,unit in plots:
        plot(out,name,title,ARMS,[(field,[groups[a][field] for a in ARMS])],unit)
    plot(out,"latency","Task latency (seconds)",ARMS,
         [(label,[groups[a][k] for a in ARMS]) for label,k in
          (("Mean","latency_mean_seconds"),("P95","latency_p95_seconds"))]," s")
    if selected:
        comparisons=report["pairs"]
        plot(out,"paired-alpha","Paired mean alpha differences",list(comparisons),
             [("Difference",[r["mean_alpha_difference"] for r in comparisons.values()])])
        points=[(a,groups[a]["mean_tokens"],groups[a]["mean_alpha"]) for a in ARMS if groups[a]["mean_alpha"] is not None]
        plot(out,"pareto","Mean alpha versus charged token cost",[],[],scatter=points)
    for name in [p[0] for p in plots]+["latency","paired-alpha","pareto"]:
        if (out/(name+".svg")).exists():
            lines.extend([f"![{name}]({name}.svg)",""])
    (out/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
