# 金融数据、披露时点与可验证检索

## 从原始数据到任务

V2 的行情快照来自 Yahoo 日线，跨市场日历用 XNYS 校验。原始响应、约束后的 OHLCV 以及 Adjusted Close、拆股、股息分开保存并记录 SHA256。工具从 `HistoricalData.view` 取得截止时点以前的行情。结算函数单独读取未来五个共享交易日并计算任务级收益。股票池固定和历史供应商修订是独立的研究局限。[数据说明](../data-expansion-v2.md)

SEC 的 CIK 标识申报人，accession 标识一次提交，filed 和 acceptance 用来推导保守可用时点。PIT 等级还需要证明取到的版本当时已经存在。没有确切版本的首见证据时，用下一交易日作历史代理。若后续修订含有较早财年的值，修订版本要到它自己的披露可用时点才能参与查询。

## XBRL 工具和 RAG

Company Facts 优先提供结构化字段，官方历史原文的 iXBRL 可作为降级来源。解析只使用已知 US-GAAP 概念、USD 或 USD/股和相同发行人的无维度上下文。先选同一披露口径的截止报告期，再选择当时可用的版本。

- 期末资产与负债使用时点数据。收入、利润和现金流使用明确的开始和结束日期。季度、年初至今、全年分别匹配。现金流只披露累计值时，季度查询可以是 Unavailable。
- 同比匹配同一标签、相近财年起止跨度与同一单位。利润率、自由现金流与资产负债比要求有效分母和对齐期间。总债务缺少明确标签时保持 Unavailable，同时保留其余字段。
- 工具返回指标、数值、单位、期间、各操作数的 accession 和可验证证据 ID。Local 与 MCP 使用同一个注册表。模型只提供指标及期间，由任务绑定 ticker、cutoff 和数据分区。Trace 保留调用和证据 ID，Verifier 核查上下文、来源和时点。
- A1 的结构化分块复用已缓存本地嵌入模型。Qdrant 在 BM25、余弦检索和 RRF 前先限制 ticker、PIT 等级、数据分区和回看窗口。V2 索引拒绝使用已更改的证据源清单。

## 源码入口

- `tradingagents_lite/data_v2_market.py` 第 62 行解析并审计 Yahoo 行情，第 123 行生成独立快照。
- `tradingagents_lite/data_v2_sec.py` 第 110 行计算披露 eligibility，第 188 行组织采集和断点缓存。
- `tradingagents_lite/data_v2_tasks.py` 第 40 行只根据输入冻结任务，第 123 行检查分区和版本。
- `tradingagents_lite/fundamentals.py` 第 50 行规范化 Company Facts，第 72 行读取内联 XBRL，第 185 行查询时点数据，第 216 行计算派生指标。
- `tradingagents_lite/tool_registry.py` 第 57 行按配置公开工具，第 63 行执行参数校验与分发。 `tradingagents_lite/mcp_finance.py` 第 28 行把 MCP Server 绑定到同一任务。
- `tradingagents_lite/data_v2_corpus.py` 第 49 行按研究边界编译语料，第 103 行构建 V2 索引。`tradingagents_lite/data_v2_evaluation.py` 第 44 行只对 Offline Extended 做结算，第 87 行验证输入覆盖。
- `tradingagents_lite/research_ui.py` 第 35 行公开 Legacy 与 Development V2，第 71 行验证单题任务、市场和索引版本。

## 面试追问

1. 为什么同一 CIK 可能多次变更名称，而同一 ticker 也可能更换 CIK？
2. 如何证明 FOIA 或 EDGAR 后续提供的版本在历史交易日已经可见？
3. 哪些分子和分母看起来单位一致，但仍然属于不同期间？
4. 如果下一份财报把上一年重新列报，应该怎样绑定修订值的起用日期？
5. 为什么逐任务 Alpha 可以做日期聚类置信区间，却无法直接复原组合净值？
6. 当股票日线完整但真实盘中曾停牌时，日线回测遗漏了哪些执行约束？

## 实操练习

1. 为同一季度做两份不同 accession 的模拟值，在修订发布日期前后各查询一次，核对证据 ID 和结果。
2. 给伪造的工具参数加入 ticker 或 cutoff，验证 Local 和 MCP 都拒绝。
3. 用同一标题构建两个分区的 chunk，检查其版本与 ID 是否不同，再查询未来时点、其他股票和 Holdout。
4. 对 Offline Extended 的 200 个基线响应重算日期聚类区间，检查 Always Hold 的方向分母为空。
