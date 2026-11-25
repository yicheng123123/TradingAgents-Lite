# TradingAgents-Lite 架构

## 当前执行图

固定任务 -> 纽约 16:15 cutoff -> MarketView（仅历史行情） -> Market Analyst -> 有界本地 Function Calling -> MarketReport 与引用校验 -> Trader Proposal 与引用校验 -> Risk Manager -> 独立 Settlement -> trajectory JSONL / CSV / 报告。

A0.5 EvidenceView 在任务构造时绑定同一 ticker 与 cutoff。B1-M 不开放财务证据工具；B1-E 可在原有行情工具上调用截止时点以前的财务证据。两个条件共享相同核心提示。

## B1 配对条件

B1-M 与 B1-E 共用同一个 Analyst 系统提示、Trader 系统提示、模型参数、任务、行情、Risk Manager 和结算方式。B1-E 只多一个受 cutoff 约束的 financial_evidence 函数工具。模型看到的具体 tool schema 和检索结果自然不同；共享核心 prompt 的 SHA 会写入 manifest，每次 API 请求还保存动态 messages/tools hash。

当前 B1 固定预算为分析师最多两轮工具调用、每轮最多两个函数、每次最多 512 output tokens、messages + tools 最多 5,000 bytes、temperature 0 和 25 秒请求超时。GLM-5.3 thinking 保持 enabled，reasoning_effort 固定为 low；供应商字段通过 SDK `extra_body` 发送。B1 应用重试数和 SDK 自动重试数均为 0。通用 `talite run --backend llm` 使用同一 SQLite 预算闸门并逐尝试记账。项目级预算和失败预留规则见[Token 预算说明](learning/token-budget.md)。模型 Key 从环境读取，不传入 LangGraph State，也不会写入日志。

## State 节点职责

- prepare：建立共用提示和当前任务 user message。
- analyst：向模型提供当轮工具 schema；验证函数调用预算或 MarketReport JSON。
- tools：校验 Pydantic 参数、执行固定函数、保存工具事件和观测结果，再以 tool message 返回模型。
- trader：只提供 MarketReport 引用过的证据，验证 Proposal schema 与引用。
- risk：在现有波动率阈值上独立否决方向，失败时标记 workflow_failure。
- runner：调用只读 MarketView 后独立取得未来结算数据；模型工具对象不持有结算接口。

每个 State 包含 messages、pending calls、observed evidence、tool_trace、structured_output_trace、citation_trace、report、proposal、errors 与 final decision。失败任务会保留 model_error / workflow_error / runtime_error 等 status；虽然 fail closed 的 Decision 可能表现为 Hold，但 status 非 ok 时 alpha 不计分，因而不会混成普通 Hold。

## 时点与结算边界

NYSE 决策 cutoff 为纽约时间 16:15，DST 由 America/New_York 时区处理。MarketView 只提供决策日及以前的行情。EvidenceView 只提供同 ticker、eligible_at 不晚于 cutoff、且处于来源回看窗口的版本。ingested_at 只用于采集审计。

Outcome provider 只在 Agent graph 完成后由 runner 调用。入场为下一共同交易日开盘，出场为第 N 个共同交易日收盘；Alpha 是方向乘以股票价格收益减 SPY 价格收益。当前历史快照与证据是回溯数据或时间代理，不能认证严格 PIT，也不包含费用、股息与组合执行。

## 未来架构

路线图保留完整文档解析、chunking、BM25 + Dense Hybrid Retrieval、Reranker、向量数据库 metadata/time filter、Evidence Grounding、长期 episodic memory、交易反馈与 Reflection、多 Agent Debate、ReAct 扩展及 MCP Server/Client。未来模块必须继续调用同一 PIT 约束层；不得通过向量索引、记忆或 MCP 旁路日期过滤。
