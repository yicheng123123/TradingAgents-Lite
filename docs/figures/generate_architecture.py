"""Generate both README architecture diagrams with the Python standard library.

Run from the repository root: python docs/figures/generate_architecture.py
"""

from pathlib import Path
from string import Template


PALETTES = {
    "system-architecture.svg": {
        "canvas": "#F8FAFC",
        "surface": "#FFFFFF",
        "ink": "#111827",
        "muted": "#5A6471",
        "border": "#DCE1E7",
        "connector": "#8491A1",
        "accent": "#3976C5",
        "rag": "#EAF0F7",
        "bull": "#EAF5ED",
        "bear": "#FBECEC",
        "memory": "#E5F3F1",
        "subtle": "#F1F4F8",
    },
    "system-architecture-dark.svg": {
        "canvas": "#161B22",
        "surface": "#1E252E",
        "ink": "#F0F3F7",
        "muted": "#B6C0CB",
        "border": "#43505E",
        "connector": "#91A1B3",
        "accent": "#86B3EB",
        "rag": "#28394C",
        "bull": "#293D35",
        "bear": "#432F35",
        "memory": "#233B3E",
        "subtle": "#29333E",
    },
}

# Explicit baselines keep mixed Chinese and English text on the intended lines.
DIAGRAM = Template("""\
<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1240" viewBox="0 0 1000 1240" role="img" aria-labelledby="title description">
  <title id="title">TradingAgents-Lite 系统架构</title>
  <desc id="description">用户研究任务经 Analyst 生成 MarketReport，直接交给 Trader。条件式辩论与历史经验提供可选信息。Verifier 与 Risk Manager 处理提案，最终决策经独立历史结算后更新 Memory。Agent Harness 贯穿整个生命周期。</desc>
  <style>
    text {
      font-family: system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC", sans-serif;
      text-anchor: middle;
      fill: $ink;
    }
    .brand { font-size: 42px; font-weight: 700; }
    .stage { font-size: 30px; font-weight: 650; }
    .module { font-size: 24px; font-weight: 650; }
    .detail { font-size: 22px; fill: $muted; }
    .small { font-size: 21px; fill: $muted; }
    .canvas { fill: $canvas; }
    .panel { fill: $surface; stroke: $border; stroke-width: 1.5; }
    .analyst { stroke: $accent; stroke-width: 1.8; }
    .accent { fill: $accent; }
    .rag { fill: $rag; }
    .bull { fill: $bull; }
    .bear { fill: $bear; }
    .memory { fill: $memory; }
    .subtle { fill: $subtle; }
    .connector {
      fill: none;
      stroke: $connector;
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
  </style>

  <rect class="canvas" width="1000" height="1240" rx="24"/>
  <g aria-hidden="true">
    <path class="connector" d="M500 130v23m-7-7 7 7 7-7"/>
    <path class="connector" d="M500 392v274m-7-7 7 7 7-7"/>
    <path class="connector" d="M275 625v12q0 9 9 9h216m225-21v12q0 9-9 9H500"/>
    <path class="connector" d="M500 773v24m-7-7 7 7 7-7"/>
    <path class="connector" d="M500 914v25m-7-7 7 7 7-7"/>
  </g>

  <!-- Research task -->
  <text class="brand" x="500" y="58">TradingAgents-Lite</text>
  <text class="module" x="500" y="93">用户金融研究任务</text>
  <text class="detail" x="500" y="120">股票 · 决策日期 · 预测周期</text>

  <!-- Analyst -->
  <rect class="panel analyst" x="64" y="166" width="872" height="226" rx="20"/>
  <text class="stage accent" x="500" y="207">① Analyst · 信息研究</text>
  <rect class="subtle" x="88" y="222" width="400" height="74" rx="14"/>
  <rect class="rag" x="512" y="222" width="400" height="74" rx="14"/>
  <text class="module" x="288" y="252">Function Calling</text>
  <text class="detail" x="288" y="281">行情 · XBRL · 金融工具</text>
  <text class="module" x="712" y="252">Agentic RAG</text>
  <text class="detail" x="712" y="281">BM25 · Dense · RRF</text>
  <rect class="subtle" x="88" y="306" width="824" height="44" rx="14"/>
  <text class="module" x="500" y="336">Context Engineering</text>
  <text class="detail" x="500" y="376">生成 MarketReport 与引用证据</text>

  <!-- Optional sources. The center connector carries MarketReport directly. -->
  <rect class="panel" x="64" y="420" width="422" height="205" rx="20"/>
  <rect class="panel" x="514" y="420" width="422" height="205" rx="20"/>
  <text class="stage" x="275" y="462">② Conditional Debate</text>
  <text class="detail" x="275" y="494">按条件触发 · 可选</text>
  <text class="stage" x="725" y="462">③ Episodic Memory</text>
  <text class="detail" x="725" y="494">召回已结算的历史经验 · 可选</text>
  <rect class="bull" x="88" y="513" width="177" height="55" rx="14"/>
  <rect class="bear" x="285" y="513" width="177" height="55" rx="14"/>
  <rect class="memory" x="538" y="513" width="374" height="55" rx="14"/>
  <text class="module" x="176.5" y="549">Bull</text>
  <text class="module" x="373.5" y="549">Bear</text>
  <text class="module" x="725" y="549">SQLite + Qdrant</text>
  <text class="small" x="275" y="600">Bull 先于 Bear，顺序执行</text>
  <text class="small" x="725" y="600">Recall · Ranking · Compression</text>

  <!-- Trader and checks -->
  <text class="stage" x="500" y="704">④ Trader · 综合决策</text>
  <text class="detail" x="500" y="735">当前研究报告 + 历史经验 + 辩论结果</text>
  <text class="module" x="500" y="765">Buy / Sell / Hold</text>

  <rect class="panel" x="64" y="808" width="872" height="106" rx="20"/>
  <text class="stage" x="500" y="845">⑤ Verifier &amp; Risk Manager</text>
  <text class="detail" x="500" y="874">结构校验 · 引用合法性 · 时点约束 · 风险规则</text>
  <text class="small" x="500" y="901">Verifier 可选 · Risk Manager 使用确定性规则</text>

  <!-- Decision and independent settlement -->
  <text class="stage" x="500" y="973">Final Decision · 最终决策</text>
  <text class="detail" x="500" y="1004">独立历史结算 → Reflection → Memory 更新</text>
  <text class="small" x="500" y="1031">结果成熟后才生成 Reflection</text>

  <!-- Shared infrastructure -->
  <rect class="panel" x="64" y="1053" width="872" height="169" rx="20"/>
  <text class="stage" x="500" y="1089">Agent Harness</text>
  <rect class="subtle" x="88" y="1101" width="264" height="38" rx="14"/>
  <rect class="subtle" x="368" y="1101" width="264" height="38" rx="14"/>
  <rect class="subtle" x="648" y="1101" width="264" height="38" rx="14"/>
  <rect class="subtle" x="88" y="1146" width="264" height="38" rx="14"/>
  <rect class="subtle" x="368" y="1146" width="264" height="38" rx="14"/>
  <rect class="subtle" x="648" y="1146" width="264" height="38" rx="14"/>
  <text class="module" x="220" y="1129">LangGraph</text>
  <text class="module" x="500" y="1129">Pydantic</text>
  <text class="module" x="780" y="1129">MCP</text>
  <text class="module" x="220" y="1174">Token Budget</text>
  <text class="module" x="500" y="1174">Trace</text>
  <text class="module" x="780" y="1174">Recovery</text>
  <text class="small" x="500" y="1211">贯穿整个 Agent 生命周期</text>
</svg>
""")


def main() -> None:
    figures = Path(__file__).resolve().parent
    for filename, palette in PALETTES.items():
        (figures / filename).write_text(
            DIAGRAM.substitute(palette), encoding="utf-8", newline="\n"
        )


if __name__ == "__main__":
    main()
