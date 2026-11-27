# FinanceBench V2 正式真实评测报告

评测日期：2026-11-27  
协议：FinanceBench Context V2  
状态：396/396 个计划实例均已归档并完成独立评分

## 范围与冻结

- 使用 FinanceBench 原 99 道正式测试题，分别运行 FB0、FB1、FB2、FB3，共 396 个实例。V1、V2 Smoke 和既有审计归档均保留；正式预测写入新目录 `data/final-financebench-real-test-v2-001`，没有复用 V1 正式运行目录。
- 实际顺序为题目顺序优先、每题依次 FB0、FB1、FB2、FB3。计划共 396 项，计划 SHA-256 为 `1e9ebf769e709280f17702744821f72c3efd81f701de7dd4414f54a953d67003`。完整计划仍在本地运行 manifest 中。
- V2 Smoke 有 8/8 个实例、12 次请求，实际模型为 `glm-5.3-flash`；它的配置 SHA-256 与正式运行配置相同，均为 `1a86f701102015ec437aedbe1816c18e8da3ddea9ddecbd5e2067ef7d2478a1a`。正式评测未运行新 Smoke。
- 正式冻结配置：`glm-5.3-flash`、temperature 0、reasoning low、512 输出 Tokens、8,000 请求字节上限、R3、候选数 24、每轮最多返回 4 项、证据上下文 3,200 字节、最多 2 轮、后续轮预留 35%、无修复、无重试。FB1 使用固定检索查询，FB2 使用 Agent 动态查询，FB3 使用本题 Oracle 支持证据，FB0 不检索。
- 未修改 Prompt、检索参数、上下文选择器、评分规则、正式题目或 Gold。为恢复 V2 Smoke 配置指纹，`tests/test_financebench_context_v2.py` 的末尾空行按 Smoke 时的字节恢复；只有空白变化，不改变测试行为，并随本次提交保留。
- Agent 阶段只读取正式题目和各实验组获准的检索或 Oracle 输入。Gold 单独保存在评分路径；只有全部 396 个预测冻结并校验 SHA-256 后，独立评分步骤才读取 Gold。未访问 V3 Holdout。

## 结果

答案准确率以完整 99 题为分母；执行失败按错误保留在分母。数值准确率使用冻结评分器标出的 37 道数值题。页级证据召回是支持证据页的代理指标，不代表完整财务行已送达或语义证据充分。引用合法率仅对模型实际输出的引用计算。

- **FB0**：答案 12/99（12.1%）；数值 4/37（10.8%）；完成 99/99（100%）；页级证据召回 0.0%（99 题）；引用不适用（0 条）；平均记账 239.8 Tokens；平均检索调用 0；延迟均值 / P50 / P95 为 2.55 / 2.17 / 6.11 秒。
- **FB1**：答案 46/99（46.5%）；数值 17/37（45.9%）；完成 99/99（100%）；页级证据召回 34.8%（99 题）；引用合法 70/70（100%）；平均记账 1,069.5 Tokens；平均检索调用 2.00；延迟均值 / P50 / P95 为 4.75 / 4.44 / 9.89 秒。
- **FB2**：答案 68/99（68.7%）；数值 26/37（70.3%）；完成 92/99（92.9%）；页级证据召回 63.6%（99 题）；引用合法 125/125（100%）；平均记账 2,642.0 Tokens；平均检索调用 1.47；延迟均值 / P50 / P95 为 15.63 / 15.07 / 26.64 秒。
- **FB3**：答案 70/99（70.7%）；数值 25/37（67.6%）；完成 98/99（99.0%）；页级证据召回 99.5%（99 题）；引用合法 106/106（100%）；平均记账 783.1 Tokens；平均检索调用 0；延迟均值 / P50 / P95 为 2.48 / 2.03 / 6.03 秒。

### FB1 / FB2 配对

- FB1 错、FB2 对：25 题；FB1 对、FB2 错：3 题。
- 配对准确率差（FB2 − FB1）：+22/99，即 +22.2 个百分点。
- 95% 公司聚类 Bootstrap 区间：+14.3 至 +32.0 个百分点。按公司为重采样单元，21 个公司簇、20,000 次重采样，固定种子 20261127。区间是这份公开基准上的描述性不确定区间，不证明泛化提升。
- FB2 比 FB1 多记账 155,678 Tokens，总量约为 FB1 的 2.47 倍，平均每题多 1,572.5 Tokens。FB2 平均延迟约为 FB1 的 3.29 倍。

### 检索轨迹与错误归因

以每题官方 Gold 锚点页为目标，汇总所有轮次候选，并与最终送达 Agent 的页比较：

- **FB1**：52/99 题的候选中未出现目标锚点页；9/99 题候选出现目标页但最终上下文未送达；31/99 题目标页已送达且答案正确；7/99 题目标页已送达但答案错误。
- **FB2**：20/99 题的候选中未出现目标锚点页；6/99 题候选出现目标页但最终上下文未送达；53/99 题目标页已送达且答案正确；13/99 题目标页已送达但答案错误；另有 7 个执行失败。

轨迹中的典型例子（仅列任务 ID，不公开问题、答案或证据内容）：FB1 候选缺少锚点页为 financebench_id_01077；FB2 候选出现但未送达为 financebench_id_01936；FB2 锚点页已送达但仍答错为 financebench_id_10136；FB2 HTTP 400 为 financebench_id_01028。

这些分类用于区分检索候选缺失、候选未送达和证据送达后仍答错，不是新增的逐行证据正确性标注。FB1 使用 198 次固定检索调用，FB2 使用 146 次动态检索调用。FB2 的动态检索提高了页级召回和正确数，也增加了 Token、延迟与执行失败；本次结果不能说明两题校准集上的调参效果，也没有据正式成绩修改策略。

### 失败与独立复核

- FB2 的 7 个失败分别为 5 个 `tool_budget`、1 个 HTTP 400、1 个 `invalid_output`；FB3 有 1 个 `invalid_output`。这些实例均留在 99 题分母中。
- HTTP 400 发生在 `financebench_id_01028` 的一次请求；前两次请求已成功。供应商错误响应经 Provider 脱敏，无法确认服务端具体拒绝原因。该请求未重试。
- 初始确定性自动评分有 228 个待复核文本或数值结果，0 个被删除。完成预测冻结后，Codex 对 228 项逐条按原 Gold、问题和既定数值容差作出裁定；自动评分文件保持不变，最终裁定另行归档。此处的“独立复核”由 Codex 完成，不是外部人工专家审阅；正式用途前应由领域人员抽查，尤其是复杂文本和口径可能不一致的项目。
- 自动评分的正确 / 错误 / 待复核数分别为：FB0 2 / 70 / 27，FB1 5 / 47 / 47，FB2 6 / 17 / 76，FB3 10 / 11 / 78。最终统计包含原自动结果及 228 项复核裁定，逐项 ID 级结果见脱敏 JSON。

## 请求、Token 与预算

- 共 585 次 HTTP 请求，584 次成功、1 次失败；无自动或人工重试。返回模型身份均为 `glm-5.3-flash`，未回退到其他模型。
- 正式运行保守记账 468,704 Tokens，其中已收到 usage 的请求共 462,936 Tokens：输入 414,383，输出 48,553。供应商提供的 reasoning 16,956 Tokens 已包含在输出中，没有重复相加；cached 128 Tokens 属于供应商 usage 字段，不在此之外再加。
- 1 次请求没有 usage，账本按请求预留保守计入 5,768 Tokens。此数为预算记账，不是供应商确认的实际消费。
- 正式实验硬上限 2,000,000 Tokens，结束时本实验余 1,531,296。A1 阶段用量为 490,951 / 2,500,000，余 2,009,049；全项目用量为 531,321 / 15,000,000，余 14,468,679。
- 预运行保守预测为 1,928,520 Tokens；正式账本按每次请求原子预留。逐实例实测和账本快照留在本地运行归档。

## 哈希与产物

- 代码基点：`45de3852f24fa087f8057929feb1a23db3e8a308`。
- Smoke manifest：`d4819f8b454533c762de1e0a6aa1cc329a39ffa4b053c5750ee5e58ee5f045e3`；Smoke 预测：`4a8351c67f324f3189e8a304418cca0aff7544103f75bdb427cf708898763f40`。
- 正式配置：`1a86f701102015ec437aedbe1816c18e8da3ddea9ddecbd5e2067ef7d2478a1a`；计划：`1e9ebf769e709280f17702744821f72c3efd81f701de7dd4414f54a953d67003`。
- 输入 manifest：`2f9afab593a2c8422af5aa6270b7fd005065d07b2238bd599f714b6fab4414a6`；FinanceBench 源提交：`cc39aeb4afdf33909ee1412188bf89035950c2eb`。
- 题目：`6045385a14fc84c7b7e0fb298e0c51957142f9a339833f123bed2691089a8573`；Oracle：`0d7dab08382ffc24db22ea0bfecd6244c9f885d7d8800e5165735d7ab0b3ff34`；Gold SHA-256（仅哈希公开）：`1b9a23bae5437db6d78ee2a39d2fb58b5189edd9d896fe95ec958f11d10365c8`；检索索引：`b5d0685d9d0b9110b9105cc88e7614065ba6f9c455bbc20992337eb4b9081498`。
- Prompt：`59be1a24cf13973e9aca13e4eb9d70fdf0f31aec263f7ad0eb2a30df03bc6c19`；答案 Schema：`3dcbfa559b77b51db4c030350a4b0263f10851cbd3c6c310e4350572ac1773bd`；工具 Schema：`bab8f8f11a0a530e2ca08dca81eec65461caf8ad0907124dce69fdc8d3bdfef0`；评分代码：`45ad1f20bc9ae89fe45d837cf35cbfd95d5ff6c88673ca1e20f13cfd12727b5b`。
- 正式 run manifest：`3352cb5a009262afce6c9f423fe189d400cb10a38bcc0fc78cd5a4f88d7e6bac`；预测：`f336271a0d004e4b2897f0186aa86a4f373d2de2db95939b89eda78803f3fbbc`；最终聚合评分：`3f4197045cf54fcca9986c8c836eb1078d77c91e3092aaa1182aca5e24cc34a1`。

Git 中提交的是脱敏聚合报告、JSON 汇总和逐项裁定统计；不提交题目、Gold、预测答案、证据正文、私有复核说明、SQLite Token 账本、索引或缓存。逐项公开 JSON 仅含题目 ID、实验组、原始自动裁定、最终裁定、完成状态、页级计数、引用计数、Tokens 和延迟。

## 回归测试与复现

全量回归：`uv run --locked pytest -q`，结果 **210 passed, 1 skipped**。

在已准备好冻结 FinanceBench 数据、索引、V2 Smoke 归档和现有 TokenBudget SQLite 账本的 WSL 环境中，复现预测可使用：

    cd /home/gaoyi/projects/tradingagents-lite
    source /home/gaoyi/.config/tradingagents-lite/glm.env
    export TALITE_MODEL=glm-5.3-flash
    uv run --locked python -m tradingagents_lite.financebench run \
      --scope test --backend llm \
      --smoke data/final-financebench-real-smoke-v2-001 \
      --confirm-test \
      --out data/final-financebench-real-test-v2-001

评分须等预测 manifest 状态为 complete 且预测 SHA 校验通过后运行。自动评分复现命令：

    uv run --locked python -m tradingagents_lite.financebench score \
      --run data/final-financebench-real-test-v2-001 \
      --out data/final-financebench-real-test-v2-auto-score-001

正式预测和审计原件在被忽略的本地 data 归档，不随 Git 提交；干净克隆需要先恢复完全相同的数据、索引、V2 Smoke 和账本，核验上述 SHA，并确保阶段及项目预算足够。复现命令会发起真实 API 请求。

## 限制与结论

FB2 在本次公开 99 题上的答案数高于 FB1，配对差值为正；但它使用更多 Tokens、延迟更高，并有 7 个执行失败。FB3 的页级召回接近 100%，答案准确率为 70.7%，说明给定支持页并不保证正确回答。FinanceBench 是公开基准，存在预训练污染风险；页级召回也不能证明行级上下文完整。结果适用于本次冻结配置和数据，不宣称无污染、普遍收益或统计显著性。

阶段结束；本轮没有启动私有反事实 Benchmark、FinanceBench V3 优化或 V3 Holdout。
