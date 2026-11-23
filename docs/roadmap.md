# TradingAgents-Lite 实现与评测路线图

## A0 与 A0.5

A0 的四题离线动量基线已归档。A0.5 已加入时点过滤、SEC accession 级原文、发行人新闻元数据和 20 题证据覆盖报告。当前来源均属于 `historical_proxy` 或 `unavailable`。证据数据模型保留 ticker、来源、稳定文档/版本 ID、哈希、UTC 时间和时点依据，供所有后续检索器共用。详见[证据协议](point-in-time.md)和[覆盖结果](../results/evidence-dev-a05-001/REPORT.md)。

## B1：真实模型基线和有界多智能体

先用同一批冻结行情、证据和 20 题，验证现有 OpenAI-compatible API 的 Market Analyst → Trader → Risk Manager 实际运行并归档工具轨迹、证据链接和模型用量。用固定模型、提示、预算与任务比较只用行情和接入证据两种条件。接入一轮 Bull/Bear Debate，保留结构化提案与风险否决。不因发现结果差而调整 Holdout。后续可单独增加财务和新闻分析节点。

## B2：完整 Agentic RAG 与向量数据库

1. 解析冻结的财报、许可允许的新闻与历史经验，按章节或文档结构切片，保留片段与父级内容哈希、原始 URL 和时间依据。
2. 为片段生成带模型版本的 embedding，持久化到支持 metadata 和日期过滤的向量数据库。按 ticker、来源、版本、PIT 等级和 `eligible_at <= cutoff` 保守过滤，再进行 BM25 + Dense Hybrid Retrieval、融合和 rerank。
3. Agent 通过有界 ReAct 选择搜索与补证工具。回答经过 Evidence Grounding 校验。为历史日期独立标注可见证据，评测 Recall@5、MRR、引文正确率和未来证据泄漏率。版本和索引更新均保存记录。
4. 基于同一工具注册表实现 MCP Server/Client，验证本地和 MCP 返回同样的证据 ID、范围和错误。分别切换 RAG、ReAct 和 MCP 后汇报延迟、调用成功/恢复率与成本。

## B3：跨任务长期记忆与消融

- A1 用 Recent Memory，A2 加入 Evidence-Aware Episodic Memory，A3 加入 Context Compaction。A0 为 No Memory。
- 交易结果到结算可见时点才成为反馈。Reflection 生成带证据和版本的经验，保存、更新和检索在用户/任务组隔离下跨任务复用。向量检索必须按经验 `eligible_at` 和数据集分区过滤。更新形成新版本，保留旧版本的历史可见性。
- 使用同样模型、工具、token 上限、数据、任务和预算测量 Memory Retrieval Quality、证据保留率、工具成功/恢复及 Token Cost。Compaction 测试引用 ID、数值与时间边界；不跨 Dev/Holdout 共享受其结果影响的经验。

## B4：Benchmark 和风险指标

锁定 20 个 Development 任务的稳定基线后，预注册 60–100 个独立 Holdout 单元。本轮不创建 Holdout。以任务/日期为单位报告方向命中率、有效非 Hold 样本、覆盖率、Alpha、配对差值与区间。检索与记忆分别报告样本和指标。Token Cost 记录实际 token、报价版本及未知项。Sharpe 和 Maximum Drawdown 需要有明确仓位、资金、费用、日历和未成交规则的连续收益曲线，不能把独立任务的收益直接当成组合曲线。Lite 与上游结果分别归档，说明数据和结算口径。
