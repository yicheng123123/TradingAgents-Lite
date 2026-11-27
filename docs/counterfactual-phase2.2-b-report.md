# Phase 2.2-B：Counterfactual V2 修复与 Development 复验

日期：2026-11-28
基线提交：`0a00130afd7f5916b63c4b850bae5acb9ccf46b5`
协议：`counterfactual-phase2-v2`

## 范围与冻结

本轮只修复 CF1 最终生成状态转换和独立评分器的货币数量级解析。Development 数据、模型、Prompt、检索器、工具 Schema、回答 Schema、任务顺序及隔离边界均未更改。V1 预测、V1 评分和 V1 根因报告均保留；V2 使用独立归档 `results/counterfactual-development-real-003`。

V2 冻结值：

- 模型：`glm-5.3-flash`，temperature 0，reasoning effort low，最大输出 512 tokens。
- 最多两轮检索；每轮最多一个检索工具调用；最终生成单独使用无工具 JSON Mode；不自动重试或修复。
- 实验预算沿用原 SQLite 账本中的 `counterfactual-phase2-v1` 实验 ID，硬上限仍为 300,000 tokens；协议版本单独升为 V2，没有清零或替换账本。
- 数据 Manifest SHA256：`6456aedbb7972eb4d94b4ec5bf21de0194cfb8e657dcd8239250e345c473f8ae`。
- V2 配置 SHA256：`6cd09e730dcfabd7b04cd745a58cff60503c2b05bc1391c6d1e7ab64c861cf49`。
- V2 计划 SHA256：`04102520e4afb81c897545f2e7bce6b78732f64c85a05b738d64438aee8f9da1`。
- Prompt SHA256：`ba750fcc7e1743a30b34c474552da2770b1f213ad5dc98c24be1c8f2c1fb1659`；工具 Schema SHA256：`bab8f8f11a0a530e2ca08dca81eec65461caf8ad0907124dce69fdc8d3bdfef0`；回答 Schema SHA256：`3dcbfa559b77b51db4c030350a4b0263f10851cbd3c6c310e4350572ac1773bd`。
- V2 预测 SHA256：`3670419fe3a4bbf44b852d3eff6e8acf43fc28479934067db34ee8cdc4bd10c4`；冻结运行 Manifest SHA256：`e4f90858f6910eb0bfcc9bb2c52cb18c92eff1b15d87ecfeec619dab488c36bf`。

V1 与 V2 的数据 Manifest、Prompt、工具 Schema、回答 Schema、端点和本地 Embedding 版本一致。新旧结果分开评分和报告。Development 已参与诊断，V1/V2 差异仅说明工程修复在该 Development 集上的表现，不代表独立测试集上的泛化提升。

## 实现变更

1. CF1 的工具阶段和最终生成阶段现在明确分离。无工具的正常停止、一次检索后停止、两次检索到达上限，均统一进入无工具 JSON Mode 最终生成。所有已完成工具观察继续保留；最终输出仍通过原 Pydantic Schema。非法工具调用、错误 finish reason、结构化校验失败或 TokenBudget 拒绝均不会触发额外重试/修复。总请求上限为每实例三次。
2. 失败诊断仅保留阶段、异常类型、错误数量及受限的字段位置/错误类型，不保存完整响应或私有输入。
3. 评分器支持单一数值后的显式 USD 百万/十亿单位，包括 `70.89 USD millions`、`70.89 million USD` 和 `USD 70.89 million`。保留 Decimal 与 ROUND_HALF_UP；多值、算式、错误币种、解释性歧义和非有限数值仍拒绝评分。

## 离线回归

执行：`uv run --locked pytest -q`
结果：**238 passed, 1 skipped**。

新增/补充覆盖包括真实 LLM Runner 初始化与 Broker 的离线 Mock 路径；CF1 直接停止、一次/两次检索后停止、非法工具调用、最终 Schema 失败、预算拒绝；最多两轮检索/三次请求；无工具 JSON Mode 最终请求；消息配对；货币数量级解析及歧义拒绝。真实 Development 的隔离探针继续生效。

## 真实 Development 结果

V2 共归档 20/20 个实例，均为 `status=ok`；所有 40 次实际 API 请求均返回 `glm-5.3-flash`。没有失败请求、重试、缺失 usage 或结构化修复。CF0 和 CF1 均为 10 个实例。

### 答案、证据与规划

| Arm | 协议 | 结构有效 | Answer Accuracy | Counterfactual Pair | Planning | Adaptive Planning | 必要证据送达 | 必要证据引用 | 拒答准确率 | Future Evidence 接受 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CF0 | V1 | 10/10 | 10/10 | 5/5 母案例 | 4/4 | — | 10/10 | 10/10 | 2/2 | 0/2 |
| CF0 | V2 | 10/10 | 10/10 | 5/5 母案例 | 4/4 | — | 10/10 | 10/10 | 2/2 | 0/2 |
| CF1 | V1 | 3/10 | 2/10 | 1/5 母案例 | 1/4 | 1/4 | 10/10 | 3/10 | 2/2 | 0/2 |
| CF1 | V2 | 10/10 | 8/10 | 3/5 母案例 | 4/4 | 2/4 | 10/10 | 10/10 | 2/2 | 0/2 |

V1 的 20 个实例均有归档，但 CF1 有 7/10 个最终输出未通过结构校验；表中“结构有效”统计 `status=ok`。V2 两组 20/20 输出均通过结构校验。V2 的 CF1 仍有两题答错，Adaptive Planning 为 2/4，未达到全对；不据此继续调整策略。

### Tokens、请求与延迟

| Arm | 协议 | 记账 Tokens | 平均 Tokens/实例 | 检索工具调用 | HTTP 请求 | 平均延迟 | P50 | P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CF0 | V1 | 5,414 | 541.4 | 20 | 10 | 2.04 s | 1.87 s | 2.81 s |
| CF0 | V2 | 5,465 | 546.5 | 20 | 10 | 1.85 s | 1.90 s | 2.36 s |
| CF1 | V1 | 13,340 | 1,334.0 | 12 | 22 | 11.70 s | 11.33 s | 14.40 s |
| CF1 | V2 | 18,111 | 1,811.1 | 15 | 30 | 11.42 s | 12.09 s | 13.11 s |

V2 新增记账 **23,576 tokens**：prompt/input 21,490，completion/output 2,086，其中 reasoning 293（属于 completion，未重复计费）。40/40 请求 usage 齐全；每次请求均为单次尝试。20 个 JSON Mode 请求对应两组各 10 个最终生成；其余 20 个请求为启用工具的检索阶段请求。最大序列化请求为 3,336 bytes。

SQLite 账本复核：本实验累计 42,330/300,000 tokens（V1 18,754 + V2 23,576），剩余 257,670；A3 累计 42,330/2,000,000，剩余 1,957,670；全项目累计 573,651/15,000,000，剩余 14,426,349。预算拒绝数为 0。母案例静态最坏预留之和高于剩余额度，但执行时仍逐母案例预检，并且每个 HTTP 请求都先由原子账本预留；实际 usage 可在硬上限内完成全批，没有提高额度或绕过账本。

## 隔离与泄漏检查

- 20/20 推理实例的网络命名空间隔离探针通过；每个实例的 19 项禁止路径、符号链接及路径遍历探测均被拒绝。
- 20/20 子进程环境中不含 `TALITE_API_KEY`；20/20 文件描述符探针未发现 Gold、Authoring、其他任务或账本逃逸。
- Gold 只在 20 个预测完成并校验冻结哈希后由独立评分进程读取。
- 有未来候选文档的任务每组 2 个；两组均没有未来证据引用或未来数值匹配（0/2）。
- 未生成 Private Test，未访问 V3 Holdout。

## GO 判断与复现

**GO：Phase 2.2-B 工程验收通过。** 两个已确认缺陷均已修复，正常最终输出经过 JSON Mode 与原 Schema 校验，离线回归通过，真实 V2 Development 执行完整且无系统级失败。模型能力指标仍是小样本 Development 结果；不得解释为独立测试上的提升。

复现命令（密钥仅从本地 WSL 配置加载，不回显）：

```bash
cd /home/gaoyi/projects/tradingagents-lite
source /home/gaoyi/.config/tradingagents-lite/glm.env
export TALITE_MODEL=glm-5.3-flash
uv run --locked python -m tradingagents_lite.counterfactual_runner run \
  --data data/counterfactual-development-v1-001 \
  --out results/counterfactual-development-real-003 \
  --backend llm
uv run --locked python -m tradingagents_lite.counterfactual_score \
  --data data/counterfactual-development-v1-001 \
  --run results/counterfactual-development-real-003 \
  --out results/counterfactual-development-real-003/scores.json
```
