# 最终 Benchmark 协议与仓库审计

## Phase 0 审计

审计基点为 `d22be5c7226c26df9302d78984fe7be4f408ac6c`。开始时本地 `main` 与 GitHub `main` 一致，已跟踪文件无改动。旧 Smoke 原始轨迹及 SQLite 检查点仍留在工作区。后续三个真实实验都将使用独立目录。

- **FinanceBench** 固定源版本 `cc39aeb4afdf33909ee1412188bf89035950c2eb`，按公司划分为校准 51 题、测试 99 题。原 R1–R4 检索归档中，正式测试 99/99 可映射官方证据页，校准 49/51 可映射。84 份本地 PDF 中，83 份已建索引、1 份不可解析，影响两道校准题。已验证原 Qdrant 目录和本地 BGE Small 权重存在。原测试集 R3 page-proxy Recall@5 为 0.3990，Recall@10 为 0.4899。这是已用于检索诊断的公开分区。当前尚无真实端到端金融问答成绩。[检索归档](../results/a1-rag-local-001/REPORT.md) · [上游公开数据](https://github.com/patronus-ai/financebench) · [许可](https://huggingface.co/datasets/PatronusAI/financebench)
- **Flash V3** 已真实执行两题、两臂，4/4 实例、2/2 配对通过。16 次请求实际返回 `glm-5.3-flash`，记账 17,863 Tokens，8/8 首轮结构化输出通过。现有 V3 入口负责校准 Smoke 和实例续跑；通用 Runner 支持真实逐题执行、账本与逐题归档，并在任务内立即结算。最终 Holdout 所需的预测和评分隔离入口尚未实现。[Flash V3](../results/b1-flash-v3-smoke-001/SUMMARY.json)
- **Data Expansion V3** 已冻结 Development 80 题和 Holdout 100 题。分区 SHA-256 分别为 `2453f2785cc81504dee71c5656efd071a5e2daa87b1824dfa185e3aa58148b4b` 和 `85fac2965f9210fa25a3555ddca66eeaf6dbc5ff55f21ca95cffa81bb688b71d`。行情版本 `yahoo-us-v3-001`，市场 manifest SHA-256 为 `b432387175f12f949344cfce1b96c68be37e2e9e94779ffe39779169bd37e2e8e`。通用行情 API 与 Runner 对 V3 Holdout 保持封存。[任务清单](../tasks/data-expansion-v3/manifest.json)
- **A1–A3** 已有 R3 BM25/Dense/RRF、A2 按结算推进的分区与实验组隔离、A3 D0/D1/D2 条件辩论、MCP STDIO、Verifier 和有界恢复。A2 的 M3 在原离线回放中将上下文从 29,620 字节降至 18,534 字节，保留 48/48 个上下文引用。A3 的 D2 触发 9/20，MCP 输出等价 8/8，决策等价 5/5；60 种故障设计下，F0 完成 5/60，F1 完成 45/60，两个条件均拒绝 5/5 例未来证据。均为离线、脚本或 Mock 记录。[A2 回放](../results/a2-memory-local-001/REPORT.md) · [A3 回放](../results/a3-offline-001/REPORT.md)
- **合成控制** 本地私有文件包含 4 个已运行的确定性反事实案例，验证数值改变、动态检索分支和时点拒绝。它们尚不具备 15 个母案例的私有测试规模，也没有真实模型结果。[控制汇总](../results/temporal-knowledge-audit-001/controls.json)
- **测试与账本** 既有回归覆盖时点数据、RAG、Memory、MCP、Verifier、恢复、Flash 身份、保守计量和 SQLite 原子预留。只读检查确认同一账本累计 40,370 Tokens，B1 为 31,986，Reserve 为 8,384。项目剩 14,959,630。B1 余 968,014，Reserve 余 2,991,616；A1 余 2,500,000，A2 余 1,500,000，A3 余 2,000,000，B2 余 1,000,000，B3 余 4,000,000。

## 通用执行与预算

真实请求使用官方 BigModel 上的 `glm-5.3-flash`、temperature 0、reasoning low、无 SDK 或传输重试，核验响应模型身份。所有真实 HTTP 调用继续使用已有项目 SQLite 账本原子预留。沿用 15,000,000 全局上限及各阶段上限。先审计模型身份、结构、检索及账本，再以各臂 Smoke 中最大实测记账量的两倍和最大单次请求预留量的较大者估算实例成本。该估算用于规模选择，逐请求仍由账本限额。预算不足时优先保留私有反事实、FinanceBench 核心对照、真实金融 Basic 与 Final。所有正式计划在生成对应测试预测前保存确定性顺序与哈希。

配置、数据、账本、检查点或模型身份不符时停止。认证、传输、预算和索引来源错误属于运行级故障。普通任务失败保留原始分母与原因。终态实例在续跑时跳过。若进程中断且已产生无法和检查点对应的账本记录，停止并要求人工核对，避免重复计费。报告归档配置与代码 SHA、数据 hash、实际模型、完整分母、用量、失败、逐题结果和挑战范围。数据正文、私有 Gold 和账本留在本地。

## Phase 1 FinanceBench 端到端公开诊断

- **FB0** 使用问题和共同答案契约。**FB1** 以原问题和去模板词关键词串执行最多两次固定 R3 检索。**FB2** 使用同一 R3，但由 Analyst 根据问题和返回内容选择查询，最多两轮、每轮一个调用。**FB3** 只观察本题官方标注且与上一版索引页对齐的证据摘录。证据遵守共同上下文预算，摘录可能截断。Oracle 是给定支持证据条件的对照。
- FB1 和 FB2 复用同一 BGE Small、BM25、RRF、Qdrant 语料、公司范围、候选数 24、每次返回最多 4 项、合计证据 JSON 3,200 bytes。每次模型请求限制 8,000 bytes 和 512 输出 Tokens，所有组最终采用 tool-free JSON mode、同一 Pydantic 契约及证据 ID 约束。QA 上限为 FB2 三次 HTTP，其他组各一次。两道固定校准题为 `financebench_id_04735` 和 `financebench_id_07507`，所有组均运行。
- 输入准备核验原版本、数据、索引和 PDF 哈希，明确记录不可用证据。问题、Oracle 和评分 Gold 各自投影，评分 Gold 在独立路径。预测程序只读取问题和索引，FB3 单独读取 Oracle；每实例独立上下文和证据 ID。索引只包含原始允许文档。逐题归档不含检索正文。模型可能在预训练阶段接触公开基准，已有检索比较也使用过这里的测试集。
- 数值保守解析，按问题声明或明确答案单位统一货币数量级、百分比与基点。误差取目标金额的 0.5% 与显示单位 0.01 的较大者，百分比最小误差为 0.01 个百分点。文本仅对规范化一致和明确二元相反的情况自动裁决。模糊单位和复合推理答案未能可靠匹配时入人工复核队列。输出已裁决正确率、完整分母正确率的上下界、可解析数值子集、页级证据召回、引用有效率、任务完成率、工具调用数、用量和延迟。引用有效率检查 ID 是否实际送达。
- Smoke 在 A1 下单独限 120,000 Tokens。12 次 HTTP 的保守预留上界为 `12 × (8,000 + 128 + 512 + 512) = 109,824`。正式运行仍用 A1，实例计划上限 2,000,000。需要本地已通过的真实 Smoke，覆盖 8 个实例、RAG 检索和 Oracle 证据，匹配配置及账本。预算足时运行全 99 道、四组。否则保留 FB1/FB2 成对比较；仍不足时按固定哈希排序取预算支持的前缀，最少 20 对。模型结果或标准答案均不作为删题依据。

目前只交付准备、预测、评分与校准 Mock。真实 Smoke 需要另行执行以下命令：

```bash
cd /home/gaoyi/projects/tradingagents-lite
# 第一次在缺少冻结投影时运行，已存在则复用。
uv run --locked python -m tradingagents_lite.financebench prepare
source /home/gaoyi/.config/tradingagents-lite/glm.env
export TALITE_MODEL=glm-5.3-flash
uv run --locked python -m tradingagents_lite.financebench run --scope smoke --backend llm \
  --out data/final-financebench-real-smoke-001
uv run --locked python -m tradingagents_lite.financebench score \
  --run data/final-financebench-real-smoke-001 --out data/final-financebench-real-smoke-score-001
```

校准 Mock 使用同一入口，将 `--backend` 设为 `mock` 并使用独立输出目录。续跑在同一路径使用 `--resume`。正式测试入口另需明确的 `--confirm-test` 和 `--smoke`。复核通过独立评分命令的 `--reviews` 读取 JSONL，字段为 `id`、`arm`、`verdict`、`reviewer` 和 `note`。

## Phase 2 私有反事实

复用 `temporal_controls.py` 的小文档设计，待实现 15 个独立母案例、每类 5 个、每个 2 个变体。Development 按类别分配 2/2/1 个母案例，Private Test 取余下 3/3/4 个母案例。母案例的变体在同一分区，文档、虚构公司、数字、时点、干扰项和必要研究目标各自独立。CF0 固定 R3，CF1 由模型根据中间结果自主检索。独立 Python Gold 计算负责会计关系、目标、容差和时点谓词。接受不同顺序的等价路径。

测试运行使用任务和变体独立会话、工具范围、缓存以及与 Gold 分离的本地权限边界。预测进程只挂载授权文档和本题输入。若普通进程权限不足，使用最小只读挂载容器或等效隔离。Gold、随机种子、配对关系与评分结果在评分进程可见。所有测试预测冻结后统一评分，并测试跨案例、跨分区和未来数据访问。指标包括数值准确率、答题准确率、母案例双正确率、Planning Success、拒答与未来证据接受率、Token 和工具开销。母案例是重采样单元。拟用 A3 下 600,000 Tokens 实验上限，先在 10 道 Development 测定用量。若现有额度不足以支持 20 道完整成对 Private Test，停止并报告缺口。

## Phase 3 V3 金融决策消融

Pilot 在运行前按 ticker 从各自四道冻结 Development 中选出一题，选择固定键的 SHA-256 最小候选，覆盖全部 20 股及 8 行业。保留 T0 动量、T1 Flash 的行情及既有 task-bound 金融工具、T2 加 R3、T3 加 Memory、T4 加条件辩论。正式配置须共享模型、基本 Prompt、风险、Verifier 和恢复策略。Memory 仅可读合法分区中在决策前已经结算的经验，空候选记空；Holdout Memory 只用冻结 Development 历史。待实现的薄适配包括 V3 执行和语料绑定、时间顺序、配对与成本汇总、安全续跑。

配置筛选规则先于 Pilot 成绩冻结。增强配置需完成至少 18/20、配对可结算至少 18、引用合法率至少 95%（没有引用时单列并排除）、未来引用为 0、Non-Hold 覆盖与 T1 差不超过 15 个百分点、平均 Token 与 P95 延迟均不超过 T1 两倍。满足条件且配对平均 Alpha 为正时择最高配对 Alpha，差距小于 0.001 时优先选择更低平均 Token，再优先模块更少。若无合格配置，选择 T1，保留负结果。Development 扩展只保留 T0、T1 和选中的配置，尽可能复用同协议 Pilot 预测。以 B2 的 1,000,000 Tokens 为共同硬上限，现有 5,645 单实例最大实际用量对应 Pilot 约 451,600 Tokens，仅为计划估算。剩余预算不足以完成全 80 题时，保持 Holdout 封存。

Final Holdout 使用独立的显式授权入口，校验最终配置与统计协议哈希、B3 容量、冻结任务和 Development 完成记录。预测进程只读决策时点行情及证据，完成并锁定 H0 动量、H1 Basic 和 H2 Final 的预测后，由独立结算进程统一读取标签。H2 与 H1 配置完全相同时引用同一预测，并声明无可独立比较增强组。现有 B3 4,000,000 Tokens 上限沿用。最终报告给出任务完成、Non-Hold、条件方向命中、Alpha、配对差值、引用、Token 和延迟，区分正常 HOLD、风险拒绝、失败。配对按共有成功任务计算并报告流失数，将同一决策日一起纳入以 10 个交易日为块的时间重采样，显式保留同日相关和五日窗口重叠的影响。该结果属于时间隔离的历史研究。供应商知识污染和市场执行限制仍然存在。
