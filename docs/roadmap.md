# 下一阶段计划

1. 先建立有发布时点的财报与新闻。为 SEC 文件冻结 accession、accepted timestamp 和源文档，为新闻保存发布时间及修订版本，统一时间过滤后再进入 Evidence Store。接入 Fundamentals 和 News/Sentiment，Bull/Bear 只辩论一轮。RAG 采用 BM25 + Embedding + Reranker 并冻结证据标注。MCP 仅转发同一工具注册表。
2. 补齐 A1 Recent Memory、A2 Evidence-Aware Episodic Memory、A3 A2 + Compaction。以 `known_at <= 决策截止` 过滤结算经验，记录检索命中和引用。Compaction 做证据 ID 保留验证，Scratchpad 只记可核查摘要。各模式固定模型、工具/Token 预算和任务快照。RAG、ReAct、MCP 分开切换。
3. 先在预先固定的 20 个 Development 任务上调通完整评测，再独立锁定 60–100 个有时点证据的 Holdout 任务。固定股票/日期采样和分组规则，隔离 overlapping outcome windows，同时计算配对差值、覆盖率和按日期分组的区间。另行记录上游版本和结算差异，避免把结果混作同一个官方基线。

当前已固定 4 个 Smoke、20 个 Development 单元。本阶段只执行 Smoke。记忆、RAG、Compaction 的指标暂时记录为缺失。
