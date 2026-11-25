# TradingAgents-Lite 路线图

路线图保留项目最终需求：完整 Agentic RAG、向量数据库、长期记忆、多 Agent、ReAct + MCP，以及真实且可复现的 Benchmark。当前工作按阶段交付，不把尚未开发的能力写成已经完成。

## A0：离线端到端基线（已完成）

冻结行情 → Market Analyst → Trader → Risk Manager → 独立结算。四题 smoke 结果保存在 results/smoke-a0-001，使用确定性五日动量策略，不调用 LLM。

## A0.5：Point-in-Time 金融证据（已完成，代理时点）

- 建立版本化证据契约：稳定 document/version ID、内容 hash、来源、UTC 时间、eligible_at 和明确 PIT 等级。
- 接入 SEC accession 级 10-K / 10-Q / 8-K 历史申报；对无法认证首次公开时间的材料使用保守的下一交易日代理。
- 接入 Apple Newsroom 标题元数据；Microsoft 新闻缺失显式标记。
- Market Analyst 可调用受单任务 cutoff 约束的财务证据工具。
- Development 20 题覆盖报告和 hash 固定在 results/evidence-dev-a05-001。
- 现有快照未通过严格 first-seen 认证，不能称作 verified PIT。

## B0：固定 20 题确定性 Development 基线（已实现）

复用五日动量规则对同一 Development 任务结算。保存逐题轨迹、配置、数据 hash、Alpha、非 Hold 方向命中、覆盖率和置信区间。B0 与官方上游结果分开报告。

## B1：真实 GLM Development Baseline（执行准备完成，真实调用待凭据）

1. 先在 dev-01 / dev-02 上运行 B1-M 行情组与 B1-E 行情加 A0.5 证据组。
2. 两组固定相同 GLM-5.3、temperature、token 上限、提示模板、工具轮数、重试、超时、风险门和结算；仅 B1-E 暴露时点过滤的金融证据工具。
3. 对比模型响应身份、结构化输出、工具轨迹、证据引用、调用/重试、tokens、延迟和真实账单估算。
4. 两题 smoke 是集成验证，不作为性能结论。确认预算上限后才运行 20 题 Development。
5. B1 暂不启用向量 RAG、长期记忆、Bull/Bear Debate 或 MCP。现有有限本地 Function Calling 只用于读取截断行情及 A0.5 证据。

旧编号映射：原 B2 RAG → A1，原 B3 Memory → A2，原 B4 多 Agent / MCP → A3，原 B5 评测 → B2 Benchmark / Ablation 与 B3 Final Holdout。

## A1：完整 Agentic RAG 与向量数据库

1. 文档解析：支持 SEC 申报和许可允许的新闻格式，记录 parser 版本、原文 hash、来源 URL、文档版本和时间来源。
2. Chunking：按文档结构切片，保留 chunk 与父文档 hash、offset、标题和 PIT 元数据。
3. Embedding：使用本地模型向量化，记录模型版本。Reranker 同样在本地运行。
4. Vector Store：持久化向量和检索元数据，支持 ticker、source_type、PIT 等级、dataset partition 与 eligible_at 时间过滤。检索前执行 cutoff 过滤。
5. Retrieval：BM25 + Dense Hybrid Retrieval、候选融合与 Reranker；保留每一步候选 ID、分数和版本，保证离线可重放。
6. Evidence Grounding：回答引用必须回指当前任务实际检索到的 chunk ID；失败时返回缺失或无证据状态。
7. Retrieval Evaluation：构造有标注的 Development 查询，输出 Recall@5、MRR、citation precision / validity 和未来证据泄漏测试。不得用收益指标替代检索质量。
8. 消融：固定任务、索引、模型与预算，比较无 RAG、BM25、Dense、Hybrid、Reranker 的独立影响；保存逐查询 JSONL 和索引 manifest。

## A2：长期 Episodic Memory 与反馈

- Memory 数据模型保存任务、可用时点、数据分区、反思文本、依据证据、结果反馈和版本 hash。
- 交易结果只在结算实际可见时写入反馈；Reflection 将结果与当时证据关联，不把未来结果提前写回过去。
- 实现 episodic memory retrieval、Memory Update、新旧版本留存及跨任务复用；检索必须按 eligible_at 和任务分区过滤。
- 消融版本：M0 No Memory、M1 Recent、M2 Semantic、M3 Semantic + Evidence-aware Ranking + Context Compaction。
- 评估 Memory Retrieval Quality、跨任务复用、Context Evidence Preservation、未来记忆泄漏、tokens 与延迟。Dev 经验不得污染 Holdout。

## A3：多 Agent、MCP 与 Verifier（离线已验证）

- 完成 Market Analyst、Bull Researcher、Bear Researcher 单轮 Debate、Trader 和独立 Risk Manager。
- 结构化 Agent State 明确每个节点可读写的信息、引用和失败状态。
- ReAct 工具选择设最大轮数、参数 schema、权限边界、超时、重试和失败恢复；模型不能构造任意 ticker、cutoff 或结算输入。
- MCP Server / Client 只适配已有金融工具注册表；本地函数调用与 MCP 必须走同一授权和执行实现，并测试输出/错误等价性。
- 独立评估开启/关闭 ReAct 与 MCP 的工具成功率、恢复事件、延迟、tokens 和成本，不默认穷举全组合。
- Verifier 优先采用确定性规则，核对证据、格式与时点约束。

[A3 评测归档](../results/a3-offline-001/REPORT.md)含真实 STDIO 与离线故障对照。真实模型实验仍待执行。

## B2：Benchmark 与消融

- 固定 smoke 4 题与 Development 20 题，用于比较检索、记忆与工具配置。
- 每项指标同时给出样本数、分母、差值和合适区间：Task Completion、Structured Output Validity、Tool Success / Recovery、非 Hold Directional Hit Rate、Non-Hold Coverage、Mean Alpha、配对 Alpha、Retrieval Recall@5 / MRR、Memory Retrieval Quality、Evidence Preservation、Token Usage、Latency 和 Token/API Cost。
- Sharpe 与 Maximum Drawdown 只在明确组合仓位、资金、交易日历、成本、现金流和成交假设的连续收益序列上计算；不能把独立任务 Alpha 拼成投资组合。
- 失败任务留在报告与分母中；Hold 不当作运行失败；无分母时报告 null。
- 保存逐题 trajectory JSONL、汇总 CSV、Markdown 报告、source/data/index manifest、模型与提示 hash、Git SHA、环境版本和费用价目快照。
- Lite 内部基线与 TauricResearch/TradingAgents 的数据、Agent、提示和评分口径分别报告，不混成同一官方基准。

## B3：独立 Final Holdout

- 在模型、数据哈希和评估协议冻结后，预注册 60–100 个独立单元。
- 按独立分区运行并保留逐题轨迹与冻结配置。

## 阶段验收约束

- 所有历史行情、新闻、财报、索引 chunk 与记忆都必须满足任务决策 cutoff。
- 若来源无法证明内容在历史时点可见，标记为 historical proxy 或 unavailable。
- Holdout 建立后禁止用其调 Prompt、调阈值、更新记忆或选择模型。
- 本项目结果是研究用途，不构成投资建议或真实交易绩效。
