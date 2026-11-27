# Phase 2.3：Private Test 本地生成与正式评测入口

本文件补充 [Counterfactual Benchmark 冻结协议](counterfactual-benchmark-protocol.md)。Agent 推理策略仍是 Counterfactual V2；本阶段仅新增正式数据生成、完整 Private Test 入口和数据/配置验证。

## 冻结配置

- Freeze ID：`counterfactual-phase2.3-private-v1`
- 新完整配置 SHA256：`880ac42b1fb499511c492260ddeee3d9f5141064afc4b32eda7e0537dc473551`
- 冻结文件 SHA256：`33b6ba801613aae61bfe20864c7e3a0231bbfd74543e50928dc0972a664b96ac`
- V2 核心推理与评分策略 SHA256：`8b57e1a1a0ec27da2825c45e5345b1ef27cc024225628b0bbfd86523d5d4ead1`
- Private Test profile SHA256：`66eba75d4466cdbb17d9bc52e0187242ace270e398a14253cdcb7f5ce92684a0`

完整配置存于 [counterfactual-phase2.3-freeze.json](counterfactual-phase2.3-freeze.json)，涵盖模型/endpoint、thinking 与 reasoning 参数、Prompt、工具和答案 Schema、R3 检索/上下文参数、Runner/评分器/生成器代码、Python 与 `uv.lock`、Embedding 文件 SHA256、任务排序、预算停止规则和正式数据验收条件。

V2 归档的完整配置 SHA256 是 `6cd09e730dcfabd7b04cd745a58cff60503c2b05bc1391c6d1e7ab64c861cf49`。Phase 2.3 使用新的完整配置 SHA，不复用旧值。核心策略摘要与 V2 相同：Prompt/工具/答案 Schema、模型与采样参数、R3/上下文参数、V2 `_execute_case`、确定性数值评分及指标汇总核心代码均未改变。新增的数据校验和入口代码计入 Phase 2.3 的新完整配置指纹。

## 正式数据规格与安全边界

- 财务数值与多步推理：3 个母案例、6 题；动态 Planning：3 个母案例、6 题；时间冲突与证据缺失：4 个母案例、8 题。共 10 个母案例、20 道任务。
- 每个时间母案例的两个变体分别为可答与应弃答；时间类合计 4 道可答、4 道应弃答。
- 正式公司名、数值、文档 ID、版本 ID 和 Planning 引用码由独立 Private 命名空间生成。Planning 分支引用码由 OS 随机种子驱动，题面不包含目标码。
- 正式生成只调用 `secrets.token_bytes(32)` 取得 OS 随机种子，不调用 LLM。种子保存在本地 `authoring/.private-seed`，文件权限为 `0600`；Private Test 目录及其子目录权限为 `0700`。CLI 仅打印数量、类别、可答/弃答计数及数据/配置指纹，不输出种子、题面或 Gold。
- Manifest 对作者文件、Gold、模型输入和种子文件分别记录 SHA256，并生成整体数据指纹。Runner 拒绝配置/Manifest/数据哈希不符、任务数不符、未来文件进入模型输入或未登记文件；正式任务必须全部纳入计划。
- 每个任务的推理子进程只得到当前任务的合规 `agent-inputs` 投影。隔离回归实际尝试读取 Private Test Gold、当前作者文件及随机种子，均被拒绝。评分器只在 40 个模型实例预测归档并通过 SHA256 检查后读取 Gold。
- 每个母案例按两个变体、每个变体 CF0 后 CF1 连续调度，共 4 个实例。每个实例继续使用独立检索索引、进程、上下文和检查点。缺少显式确认、请求子集、配置不符或数据校验失败均在 Provider 初始化前拒绝。

## 本轮离线验收

`uv run --locked pytest -q`：**246 passed, 1 skipped**。临时夹具验证了数据分区、Gold 可区分性、Planning 随机引用码、截止时点投影、可答/弃答数量、实际 Gold/作者文件/种子读取拒绝、显式授权、配置与数据哈希失配、母案例计划顺序、断点续跑不重复实例及预算不足时在母案例边界停止。

本轮未生成正式 Private Test、未启动真实 Runner、未访问 V3 Holdout；新增真实 GLM 请求和 Token 消耗均为 **0**。账本保持原实验 ID `counterfactual-phase2-v1`；已知累计 42,330 / 300,000 Tokens，剩余 257,670。正式执行仍以 SQLite 原子预留为准，不提高预算。

## 用户本地执行命令

### 1. 生成并检查正式 Private Test

在 WSL 本地执行。此命令不需要 GLM 凭据，也不输出私有题目或种子：

```bash
cd /home/gaoyi/projects/tradingagents-lite
uv run --locked python -m tradingagents_lite.counterfactual_dev \
  --scope private-test \
  --out data/counterfactual-private-test-v1-001 \
  --freeze docs/counterfactual-phase2.3-freeze.json
```

命令成功时只打印脱敏验收摘要。将生成目录留在本地，不要复制到 Codex 对话或提交 GitHub。

### 2. 执行完整 40 实例真实评测

```bash
cd /home/gaoyi/projects/tradingagents-lite
source /home/gaoyi/.config/tradingagents-lite/glm.env
export TALITE_MODEL=glm-5.3-flash
uv run --locked python -m tradingagents_lite.counterfactual_runner run \
  --data data/counterfactual-private-test-v1-001 \
  --out results/counterfactual-private-test-real-001 \
  --backend llm \
  --confirm-private-test
```

Runner 校验全部 20 题并执行 40 个任务-实验组实例。发生进程中断时，可用同一命令追加 `--resume`；已冻结的预算停止条件不能通过续跑绕过。预算不足时保留部分检查点并标记未完成，评分器不会将其接受为完整结果。

### 3. 独立评分

仅在 Runner Manifest 状态为 `complete` 时执行：

```bash
cd /home/gaoyi/projects/tradingagents-lite
uv run --locked python -m tradingagents_lite.counterfactual_score \
  --data data/counterfactual-private-test-v1-001 \
  --run results/counterfactual-private-test-real-001 \
  --out results/counterfactual-private-test-real-001/scores.json
```

完整预测归档、Gold 和随机种子均为用户本地材料，不属于公开提交内容。
