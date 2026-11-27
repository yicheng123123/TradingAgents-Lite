# Phase 2 私有反事实 Benchmark 实施交接

基线提交为 `c52b494a0954e22b20f4d28ad2ef71160ecaf761`。本文件中的任务设计、实验组、数据分区和预算约束为冻结协议；Phase 2.1 已补齐离线 Runner、独立评分器及进程隔离验收。正式 Private Test 仍未生成，真实模型评测仍未执行。

Phase 2.3 为该冻结协议新增正式 Private Test 本地生成器和显式确认 Runner。当前仍未生成正式私有数据或运行正式评测；实现、freeze 指纹及 WSL 命令见 [Phase 2.3 生成与执行说明](counterfactual-phase2.3.md)。

## 任务设计与分区

| 类型 | Development 母案例 / 任务 | Private Test 母案例 / 任务 | 合计任务 |
| --- | ---: | ---: | ---: |
| 财务数值与多步推理 | 2 / 4 | 3 / 6 | 10 |
| 动态 Planning | 2 / 4 | 3 / 6 | 10 |
| 时间冲突与证据缺失 | 1 / 2 | 4 / 8 | 10 |
| 合计 | 5 / 10 | 10 / 20 | 30 |

每个母案例有两个独立执行的变体。虚构企业、短文档、数值和证据由程序生成。Development 包含自由现金流、调整后现金转换率、两种审批路由和授信公告时点对照。后续使用不同虚构企业和独立标识生成正式母案例。时间类正式案例覆盖旧版本与未来修订冲突、恰逢截止时间、未来才发布的关键文件和永久缺失，冻结前平衡为 4 道有解、4 道应弃答。

- 财务类从不同短文档取数，组合两个以上运算步骤并明确单位、符号和舍入。为变体规定有意义的结果变化。
- Planning 类先读说明，取得当前获批材料的随机引用码，再查对应材料。两条分支应都看似合理且答案不同。变体切换审批状态，在同一母案例内保留配套数值。引用码、文档 ID 和顺序独立随机化，问题中没有目标码。Runner 只接收模型返回的查询。
- 时间类按带时区的 `eligible_at <= cutoff` 选取可用文档。对同一文档取最新的可用版本。回溯填写的报告期间不能替代披露时点。未来专属数值与合法结论须经程序确认可区分。

## 生成、投影与冻结

Development 生成器为 `tradingagents_lite.counterfactual_dev`，只接受输出目录。目前生成包位于 `data/counterfactual-development-v1-001`，保存在 Git 忽略目录。用新目录复现：

    python3 -B -m tradingagents_lite.counterfactual_dev --out data/counterfactual-development-repro

`authoring/` 保存各案例的完整短文档及未来版本。`agent-inputs/<task_id>.json` 每文件只含本题、截止时间和通过投影的文档。`scoring/reference.jsonl` 单独保存母案例、类型、目标值、必要文档版本、时点反例和 Planning 预期分支。根 manifest 记录 Development 种子及文件哈希，供数据管理者检查。Development 参考值随生成离线计算，与后续 Agent 执行无关。

正式发起请求前先冻结并哈希模型与 endpoint、Prompt、工具和答案 Schema、R3/上下文参数、适配层与评分器代码、执行顺序规则及预算停止规则。然后在独立数据生成进程用操作系统随机源创建私密种子，使用独立正式命名空间产出 20 道 Private Test。种子和生成清单留在作者侧。固定冻结哈希决定母案例顺序，按母案例连续安排四个实例（两个变体 × 两组），逐实例新建运行环境。只用预先登记的确定性质量约束拒绝不合法生成数据，记录拒绝原因。冻结清单记录验收数量、数据投影与语料哈希、病例顺序、每题索引哈希与评分规则哈希。真实预测开始后保持配置和测试集不变。

## Runner 复用与两组限制

- **CF0 Fixed Hybrid RAG** 对题干执行 FinanceBench V2 `fixed_queries()`，至多两次检索，随后生成最终 JSON。第二个查询不能读取第一轮观察。
- **CF1 Agentic Hybrid RAG** 用同一 `rag_search` 工具。模型自行发首轮查询，根据已送达文档选择是否以及如何发第二轮查询。每轮至多一次工具调用，最后用同一结构化输出契约。Python 只校验查询和调用上限，不能按 Gold、审批状态或题型为模型生成或改写后续查询。
- 两组工具白名单、语料、过滤、上下文和资源上限相同。CF0 的固定调度实际最多发 1 次模型请求，CF1 实际最多发 3 次。模型固定 `glm-5.3-flash`，temperature 0，thinking enabled、reasoning low、512 输出 Tokens，真实返回身份必须核验。使用同一金融答案契约和 JSON mode，零重试、零修复。
- 复用 V2 的 R3 BM25/BGE Small/RRF/本地 Qdrant、候选数 24、top-k 4、3,200 证据字节、首轮为后续保留 35%、8,000 请求字节、上下文选择与送达记录。各篇合成短文档预期一个 chunk，索引只含本题许可语料，使用现有离线 Embedding 权重。
- `financebench.CaseSearch.search()` 写死了 FinanceBench 分区和截止时间，不能原样用于反事实。新增反事实适配层，复用 V2 `CaseSearch._deliver()`、检索追踪、答案验证和调用约束。通过 `Retriever.retrieve()` 先限制实例语料、带时区截止点和 `verified_known_at`，再排序与选择上下文。送达前再次验证授权文档、版本、内容哈希及时间。索引中出现未知 ID、越界时点或不符哈希时停止该运行并保存最小诊断。候选与被选中文档仍是不同的审计事件。
- 实际执行仍用 `ModelBackend` 和现有 `TokenBudget`。实例完成或明确失败后写入原子检查点，续跑按配置和数据指纹、实例 ID 与账本核对，以免重复计费。普通实例失败保留在分母。模型身份不符、来源违规、预算拒绝等运行级问题应停止请求。

## 隔离边界与未来信息控制

1. 作者侧进程生成原件、种子和评分参考。仅通过解析后字段白名单导出单题输入，由数据管理进程核验 SHA-256 并送往执行环境。执行进程目录里只有当前实例的投影、本次 R3 索引、必要代码和离线权重。运行时不挂载仓库根目录、`data/` 父目录、宿主 HOME、配置文件、作者/评分目录、其他题或先前轨迹。
2. 每实例启动新的非特权受限进程和独立临时工作区。可用经过验证的 Linux mount namespace/容器，或等效 OS 权限隔离。挂载只读依赖，进程、网络和文件系统访问明确受限，示例为默认断网并经继承的 stdio 通道请求可信 Provider broker。只有 broker 拥有官方 API 凭据和现有账本写权限，绑定当前实例和共同实验 ID，执行预留并核验请求大小、工具和调用上限。Broker 同样不挂载评分数据和生成种子。模型工具仍只有 `rag_search`，没有 shell、文件、Memory、任意网络或路径入口。若目标 WSL 无法验证此隔离，停止真实评测。
3. 两个变体和两组可使用同一许可文档内容规范，但每实例重新构造上下文、Retriever scope 缓存、模型会话和 trace。不得把其他案例的语料、请求日志、Embedding 缓存或检索对象放进其进程。只读模型权重可共享。调度器只知道本次完成记录与单题指纹。
4. 评分由另一个离线进程执行。它只在所有计划实例均有终态、预测集合与 manifest 哈希锁定后读取 Gold。未来证据探针放在作者侧及隔离验收环境，真实检索数据仍执行预投影和运行时双重校验。

实施验收用合成回归验证以下边界：

- 在非当前案例、评分目录与种子文件放置唯一标记。以实际受限 UID 验证文件不存在或无权限，并覆盖绝对路径、相对遍历、符号链接、继承环境及文件描述符。把别的任务或组的标识交给搜索工具，检查无内容和元数据泄漏。
- 注入 `eligible_at == cutoff` 和晚一秒的文件，以及报告期回溯、披露时间在未来的修订。检查投影、BM25/Dense/RRF 候选、送达、最终引用四个边界。模拟带未来版本的非法检索返回，确认二次验证拦截并停止。缺证据用例应预期弃答。
- 用脚本 Mock 记录真实调度顺序，验证 CF1 第二个查询来自上轮模型决策，变体改变路由会改变查询。在 CF1 省略第二轮、猜中结果或查询错误分支时，Planning 指标不能误判为成功。CF0 的两条固定查询只由题干产生。
- 验证隔离失败、哈希失配、部分运行、重复实例、超限、账本和检查点不一致均按约定拒绝或停止。以两个短实例检查真实入口的离线 Mock 流程，不把 Mock 结果计入真实成绩。

## 独立评分器设计

输入为完整冻结预测、评分专用参考值与已送达证据/version 映射。将 `(task_id, arm)` 与计划精确对齐并拒绝重复。评分读取结构化 Gold，不调用 Agent、Retriever 或模型。预测沿用 `status`、`answer`、`evidence_ids`。答案解析只允许一个带符号 Decimal 加明确约定的单位，可接受等价货币或百分比表示，拒绝含多候选数字、错单位、非有限值和额外解释中的不明确数值。金额以 USD millions、百分比以百分点核算。使用题干两位小数约定，Gold 用 Decimal 的 ROUND_HALF_UP 生成。等价单位先精确换算，再按相同规则舍入并比较；不复用 FinanceBench 的宽容差和文本待复核队列。

- **Answer Accuracy** 以每组全部任务为分母，比较数值或正确弃答。失败、无效输出和非法引用记错。另保留数值正确但证据未完整引用的标志，避免把猜中当作证据依赖。
- **Counterfactual Pair Accuracy** 以每组母案例为分母，两个变体均答对才记成功。另记录两种变体的必要证据均被引用的母案例数。
- **Planning Success Rate** 在 Planning 任务上检查获批目标证据是否送达并合法引用，两组均报告。CF1 另报适应性链路通过数，要求首轮确实送达路由文档和引用码，模型发出的第二轮查询含该码，随后送达目标版本。无第二轮、先猜码、固定组的预定查询均无法证明适应性。
- **Abstention Accuracy** 以预设时间类任务为分母。要求应该答时生成有效回答，应该拒答时返回结构化弃答且答案为空。分别报告应拒答/可答分子与分母，答案值正确性继续由 Answer Accuracy 计算。
- **Future Evidence Acceptance Rate** 以在作者侧确实存在未来候选的时间类试题为分母。未来版本被引用，或最终报出已登记且与所有合规答案不相同的未来专属值，记为观察到接受。分别报告引用和数值一致，并单列输出/执行失败及无法判断的数量。未来内容实际送达属于隔离缺陷，触发停止，不能只当评分扣分。数值碰巧相同不能单独证明发生了泄漏。
- **Tokens 与工具调用** 按组与分区报告合计、均值、请求数、工具调用数和预算拒绝数。失败或缺 usage 的保守记账按原账本处理。

Development 和 Private Test 分开汇总。比较 CF1 与 CF0 时按母案例对齐，统计母案例内两变体的平均差和双正确差，给出原始分子与分母。正式测试只有 10 个母案例，若给区间，应以母案例整体重采样并标记描述性结果，不能按 20 个相互独立任务计算。该设计验证合成试题的隔离、证据依赖和规划行为，无法认证闭源模型的训练数据完全无污染。

## 工作量与 Token 预检

预计实施 **2–3 个工程日**，含正式生成器与独立评分器约 0.5 日，Runner 与检索适配约 0.5 日，进程隔离与 broker 验收约 0.5–1 日，离线回归和运行报告约 0.5–1 日。隔离方式需用实际 WSL 环境确认。

- Development 20 个、Private Test 40 个真实实例，总计 60 个。按 CF0 每实例 1 次、CF1 每实例最多 3 次，请求最多 120 次。无额外 Smoke 或付费调试批次。
- 用 FinanceBench V2 正式运行 FB1、FB2 平均实测每题约 1,069.47、2,641.98 记账 Tokens 作类比。预估 Development **37,115**，Private Test **74,229**，合计 **111,344**。文档与查询不同，此值仅作计划参考。
- FinanceBench V2 Smoke 的同组保守规划系数 5,413 和 6,562 给出 **359,250**。若所有 120 次请求按 V2 8,000 字节加 128 协议、512 输出、512 推理预留计，累计 **1,098,240**。这两种情景均超出 300,000，因此全量执行无法靠静态最坏预留保证。
- 复用 A3 阶段上限 2,000,000 和全局 15,000,000。只读账本快照显示 A3 已记 0，项目已记 531,321，项目余 14,468,679。将两个分区绑定同一 `experiment_id`，设置 `experiment_limit=300_000`，每次 HTTP 前原子预留。单母案例最多 8 次请求，完整预留上界 73,216；每个母案例启动前按现有余量预检，低于该值则在母案例边界停，并保留完成记录。若途中因其他任务竞争而发生原子预留拒绝，则停止并归档当前检查点。该策略优先保住配对，也可能在尚有余额时提前停止。部分运行独立标为未完成，不能冒充预定 60 实例结果。

## Phase 2.1 实施与离线验收（2026-11-27）

实现位于 `tradingagents_lite/counterfactual_runner.py` 和 `tradingagents_lite/counterfactual_score.py`。CF0 使用固定题干查询，CF1 的后续查询来自模型工具调用；两者复用相同检索与上下文投影、证据白名单、截止时点验证和 JSON 答案契约。运行 Manifest 同时保存完整的冻结配置和配置哈希，哈希涵盖 Runner、独立评分器及其检索、Provider、预算、模式和生成依赖。每个任务和实验组单独启动进程、索引、会话及检查点。Runner 的 Mock Driver 仅为确定性工程夹具，结果不能视为模型能力或真实评测结果。

推理子进程使用 WSL/Linux user、mount、PID 和 network namespaces。实际集成验收尝试读取 Gold、生成清单/种子、作者数据、其他任务及实验组、账本、运行 Manifest 和预测文件，并尝试符号链接、目录遍历、继承凭据及网络访问；敏感路径访问均被拒绝，API Key 未传入子进程，隔离网络只有 loopback。评分器是独立命令，只在完整运行清单及预测哈希验证通过后读取评分参考和作者侧未来版本。

最终离线归档为 `results/counterfactual-development-mock-005`，来自冻结的 10 道 Development 题，共 20 个 Mock 实例：CF0 10/10 完成、10/10 自动答案正确、5/5 母案例配对正确；CF1 10/10 完成、10/10 自动答案正确、5/5 母案例配对正确，Planning 4/4，动态 Planning 链路 3/4。两组 Citation Validity 均 10/10，Abstention Accuracy 均 2/2，Future Evidence Acceptance 均 0/2。每组 Mock Tool Calls 20；Mock backend calls 分别 10、30；HTTP 请求 0，Token 0。以上仅验证工程行为，不能说明真实模型准确率。Mock 轨迹 SHA-256：`aa6409c15a9fd27da66643b729ff5f6e1f9e71e64de48aa86bbff3c8810e350e`。

最终验证：`uv run --locked pytest -q` 为 223 passed、1 skipped（以交付时最后一次完整运行结果为准）。独立评分命令：

```bash
uv run --locked python -m tradingagents_lite.counterfactual_score \
  --data data/counterfactual-development-v1-001 \
  --run results/counterfactual-development-mock-005 \
  --out results/counterfactual-development-mock-005/scores.json
```

### 后续 10 题真实 Development 执行

须由用户在含有已配置凭据的 WSL Shell 里执行。命令固定 Flash，不启用自动重试；首次真实运行会校验原 TokenBudget 账本，并把 300,000 Token 阶段上限原子用于预算预留，不会初始化或清空缺失账本。

```bash
cd /home/gaoyi/projects/tradingagents-lite
source /home/gaoyi/.config/tradingagents-lite/glm.env
export TALITE_MODEL=glm-5.3-flash
uv run --locked python -m tradingagents_lite.counterfactual_runner run \
  --data data/counterfactual-development-v1-001 \
  --out results/counterfactual-development-real-001 \
  --backend llm
uv run --locked python -m tradingagents_lite.counterfactual_score \
  --data data/counterfactual-development-v1-001 \
  --run results/counterfactual-development-real-001 \
  --out results/counterfactual-development-real-001/scores.json
```

当前不生成 Private Test、不运行真实 GLM、不读取 V3 Holdout，亦不修改 FinanceBench。未来完整计划仍为 60 个任务-实验组实例，最多 120 次请求，共用 300,000 Token 硬上限。按 Development/Private Test 类比，经验估算约 111,344 Tokens；Smoke 风格保守估计 359,250；若 120 次都按单请求最大预留核算，上界约 1,098,240 Tokens，超过阶段额度。因此不能承诺一轮完成全部实例，Runner 会在原子预算拒绝时停止、保留轨迹，并禁止重复计费；预算不足预检按母案例边界进行。金额、实际 Token 及能完成的母案例数须以真实调用账本为准。
