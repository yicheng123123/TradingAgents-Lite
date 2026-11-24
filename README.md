# TradingAgents-Lite

一个可审计、可复现的轻量金融研究 Agent 项目。当前版本把历史任务、行情视图、时点证据、有限工具调用、结构化决策和独立结算连接起来。项目用于离线历史研究，不执行交易。

上游设计参考 TauricResearch/TradingAgents；本项目独立实现。来源、许可证和声明见 [docs/upstream-audit.md](docs/upstream-audit.md)、[LICENSE](LICENSE) 和 [NOTICE](NOTICE)。

## 阶段状态

- **A0**：4 题确定性五日动量 smoke，结果保存在 results/smoke-a0-001。该归档保持不变。
- **A0.5**：20 个 Development 时点的历史证据层。SEC 财报为 20/20 历史代理可用；Apple Newsroom 标题为 10/20；Microsoft 新闻显式缺失；严格版本时点认证为 0/20。逐题证据见 [A0.5 覆盖报告](results/evidence-dev-a05-001/REPORT.md)。
- **B0**：20 题确定性 Development 基线，独立保存于 results/b0-development-001。
- **B1**：真实 GLM 的 B1-M 行情组与 B1-E 行情加 A0.5 证据组。调用、对照和逐题轨迹已实现；当前真实 API smoke 需在 WSL 配置 API Key 后执行。没有 API Key 时不生成模型成绩。状态见 [B1 smoke 报告](results/b1-smoke-003/REPORT.md)。

B1 当前默认模型为 GLM-5.3，OpenAI Chat Completions Base URL 为 https://open.bigmodel.cn/api/paas/v4。智谱官方文档列出了此模型、兼容协议与 Function Calling；本机尚未通过账户认证或收到模型响应，因此账号权限和响应中的实际模型身份仍待两题 smoke 实测。参考[模型文档](https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3)与[工具调用文档](https://docs.bigmodel.cn/cn/guide/capabilities/function-calling)。

## B1 冻结协议

B1 使用固定的 Development 20 题、A0.5 行情快照和证据快照。两个模型组共用核心提示、模型、temperature=0、每次调用最多 1,200 tokens、最多 2 轮分析师工具调用、每轮至多 2 个工具调用、25 秒超时、每次逻辑请求最多 1 次模型重试，以及同一风险门和结算代码。

- B1-M：行情工具。
- B1-E：相同行情工具，另开放按决策时点过滤的财务证据工具。
- 首轮 smoke 固定为 dev-01 / AAPL 与 dev-02 / MSFT，两题同一决策日期。
- 模型生成不设置供应商 seed；seed 42 仅用于统计重采样。
- 请求提示与工具定义的 SHA-256、模型调用次数、返回模型名、token 用量、延迟和失败状态都写入 run manifest 与逐题 trajectory JSONL。
- 结算为决策后下一共同交易日开盘到第 N 个共同交易日收盘的价格收益，Alpha 减 SPY 同期价格收益；不含费用、股息、融券与组合执行。
- smoke 的粗略预期成本约 ¥0.2–0.5；按 32 KiB 输入包上限、1,200 输出 token 和全部重试均计费估算的保守封顶为 ¥9.594880。20 题双组的保守上限为 ¥95.948800。全量 Development 命令要求调用者显式提供不低于计算上限的预算。

成本预估使用截至 2026-11-24 的智谱标准单价：GLM-5.3 输入 ¥8 / 百万 tokens，输出 ¥28 / 百万 tokens。账户折扣、缓存计费和平台最终账单可能不同。价格来源：[智谱 API 定价](https://docs.bigmodel.cn/cn/guide/start/pricing)。

## 安装与测试

~~~sh
cd ~/projects/tradingagents-lite
uv sync --locked
uv run --locked pytest -q
~~~

核心依赖为 Python 3.12、LangGraph、Pydantic 与 OpenAI-compatible Python SDK。项目不需要 GPU，也不自动安装大型模型依赖。

## 运行 B0

首次为冻结的 Development 20 题生成离线规则基线：

~~~sh
uv run --locked talite b0   --tasks tasks/development.jsonl   --data data/yahoo-2024-v1   --output results/b0-development-001   --seed 42
~~~

输出包括 run.json、summary.csv、REPORT.md、tasks.jsonl、trajectory.jsonl 和本地 SQLite 账本。SQLite 文件不纳入 Git。

## 配置 GLM 并运行两题 B1 smoke

模型名与 Base URL 有默认值。只需在 WSL 当前 shell 配置 Key；不要将 Key 写入仓库或聊天：

~~~sh
read -rsp "TALITE_API_KEY: " TALITE_API_KEY
printf '\n'
export TALITE_API_KEY
uv run --locked talite b1   --scope smoke   --budget-cap-cny 9.59488   --output results/b1-smoke-004
unset TALITE_API_KEY
~~~

若成功，输出含 B1-M、B1-E 两组独立 run manifest、逐题 JSONL trajectory、汇总 CSV、逐题配对 Alpha 和报告。smoke 输出目录必须是新路径，程序不会覆盖已有结果。API Key 缺失时会记录明确的 blocked 状态，不会发出请求。

两题 smoke 用于确认 GLM Function Calling、Pydantic 结构校验、证据引用、费用统计和结算链路。它不是性能结论。模型失败、无效引用与无法解析输出均留在失败分母，不会伪装成正常 Hold。

## 扩大到 20 题

先检查 B1 smoke 的模型身份、调用轨迹、费用和失败状态，再由项目维护者确认费用上限。全量运行需要显式传入不低于当前协议上限的预算，例如：

~~~sh
uv run --locked talite b1   --scope development   --budget-cap-cny 96   --output results/b1-development-001
~~~

若价格、模型预算或重试协议变化，应重新计算成本上限并新建实验配置，不覆盖旧归档。当前开发集不是 Holdout；本项目尚未访问最终 Holdout。

## 指标和边界

报告包含 Task Completion Rate、Structured Output Valid Rate、Tool Success / Recovery、Non-Hold Coverage、非 Hold 方向命中率及分母、Mean Alpha、日期聚类区间、B1-E−B1-M 配对 Alpha、有效/无效证据引用、输入/输出 tokens、模型调用、延迟与标准价格估算。按 AAPL、MSFT 和全体分别统计。样本很小的 smoke 不提供虚假的显著性结论。

行情是历史下载的 Yahoo Finance 回溯快照，存在供应商事后修订和复权限制。A0.5 证据多数为历史代理，不是经首次可见时间认证的 point-in-time 数据。模型预训练知识也无法按历史日期截断。该实验只能称为冻结数据上的历史研究基线，不能称为严格历史回测、可交易策略或真实收益。

## 后续模块

完整 Agentic RAG、带 metadata / 时间过滤的向量数据库、BM25 + Dense Hybrid Retrieval、Reranker、Evidence Grounding、长期 Episodic Memory、交易结果反馈与 Reflection、Bull/Bear Debate、ReAct 扩展、MCP Server/Client、消融和独立 Holdout 均保留在 [路线图](docs/roadmap.md)。B1 不启用这些新模块，以隔离模型与证据访问条件。

完整调用链、状态传递与练习见 [docs/learning/b1.md](docs/learning/b1.md)；协议和字段定义见 [docs/b1-protocol.md](docs/b1-protocol.md)。
