# B1 实验协议

> 本文描述已归档的 B1 V1 GLM-5.3 协议。新 Flash 实验使用独立的
> [B1 Flash V2 协议](b1-v2-protocol.md)；本页、V1 轨迹与 V1 Token 账本均保留。

## 冻结输入

- Development 集：20 个固定 ticker/date 单元，SHA-256 为 2d0e4f92d583d64ceba729b4e3ffa0004b3bd29a5a00d990434dcdab5fceedae。
- 行情 manifest：f5e4c5095953ce737b7b99bba71b2ad26a9f30d2f34b9fba8320027e46e0f3cb。
- A0.5 evidence manifest：b46f9ef2ebc1c08022500282147940985eb9d3ceabe8cdaf78eec6efa190e4dc。
- Evidence JSONL：b0ffce595575257231432650495d756681a9f4ac10f5deee972110c7a7b7a532。
- 不读取 Holdout，不改写 A0 / A0.5 归档。
- Smoke 复用及 18 题续跑按[独立补充协议](b1-resume-protocol.md)执行；不改变本页冻结的模型、提示、工具或结算设置。

程序在运行前校验以上 digest。smoke 固定 dev-01 / dev-02，分别为 AAPL 和 MSFT，决策日期相同。两组共用同一筛选任务文件，避免 task order 和 settlement 改变。

## 分组与提示

| Variant | Market tools | A0.5 financial evidence |
|---|---|---|
| B0 | 确定性五日动量 | 不使用 |
| B1-M | GLM Function Calling | 不开放 |
| B1-E | 相同 GLM Function Calling | 开放按 cutoff 过滤的 evidence tool |

B1-M / B1-E 共用 Market Analyst 和 Trader 系统提示。E 组不追加提示说明，只增加 financial_evidence 工具定义及其工具结果。提示模板和每次请求的 messages/tools 分别以 SHA-256 记录。返回的实际 model 字段也会逐调用记录。

## 固定模型参数与资源上限

- 默认 GLM-5.3；默认 OpenAI Chat Completions Base URL 为 https://open.bigmodel.cn/api/paas/v4。
- temperature=0；reasoning enabled；reasoning_effort=low。
- 输出上限为每次请求 512 tokens。
- Market Analyst 最多两轮工具调用，每轮最多两个工具；之后强制要求 final MarketReport。随后最多一次 Trader 请求。
- B1 应用重试为 0；OpenAI SDK 自动重试为 0。
- API 请求 timeout 25 秒。messages + tools 的 UTF-8 序列化上限为 5,000 bytes，超限时在发请求前失败。
- 生成不设置供应商 seed；统计 seed 固定为 42。
- 每次请求额外保守预留 128 协议 tokens、512 output tokens 和 512 reasoning tokens。
- Function tool 本地执行为同步、限参数且本轮没有远程工具。若函数抛 TimeoutError，会记录 tool_timeout；该测试是异常路径模拟，不代表对任意未来远程工具提供通用中断保证。

## 成本预算

智谱官方价目表在 2026-11-24 页面列出 GLM-5.3 输入 ¥8 / 百万 tokens、输出 ¥28 / 百万 tokens。代码按每次 5,000-byte prompt 上限、128 协议 tokens、512 output tokens 和 512 独立 reasoning reserve 计算最坏 token reservation。每个任务实例最多 4 个模型请求，B1 不重试：

| Scope | 每组任务 | 成对逻辑调用上限 | HTTP 尝试上限 | 成本封顶 |
|---|---:|---:|---:|---:|
| Paired smoke | 2 | 16 | 16 | 98,432 tokens / ¥1.115136 |
| Full Development | 20 | 160 | 160 | 984,320 tokens / ¥11.151360 |

两题 smoke 的保守上界 98,432 tokens 低于 120,000-token 实验硬上限；其余 18 题上界 885,888 tokens。Smoke 与续跑合计 984,320，低于 B1 阶段的 1,000,000-token 额度。项目总额度与其它阶段分配见[全局 Token 预算说明](learning/token-budget.md)。执行前会核对现存账本的剩余额度，每次 HTTP 尝试再原子预留。失败、超时或 usage 缺失均不会按零 tokens 处理。实际模型用量以接口 usage 为准；reasoning 与 cached detail 不重复加到 provider 的 output / input 总量。账户折扣、缓存价格和最终账单可能不同。若持久账本不存在，B1 拒绝创建空账本并拒绝真实调用。

V1 后续真实 GLM-5.3 Smoke 归档在 `results/b1-smoke-004`：1/4 实例完成、0 个有效配对，实际记录 5,057 tokens。失败原因为 5,326-byte 请求超出 V1 的 5,000-byte 限制，以及响应请求了每轮 3/4 个工具、超过当时的 2 个上限。该归档、协议参数和账本记录保留；它们不纳入 Flash V2 成绩。Flash V2 的独立协议和后续结果见 [b1-v2-protocol.md](b1-v2-protocol.md)。

## 结算与失败

结算逻辑与 B0 相同：决策后下一共同交易日开盘入场、第 N 个共同交易日收盘退出；Alpha 为方向乘以资产价格收益减 SPY 价格收益，不计交易费用、股息现金流和融券成本。

状态非 ok 的任务属于失败，不进入 Alpha 或 Directional Accuracy 的分子/分母；Task Completion Rate 保留完整任务分母。失败任务以模型错误、workflow 错误或运行时错误写入 trajectory。不能把失败转换成正常 Hold 计分。

方向命中率报告非 Hold 分母和 Wilson 95% 区间；平均 Alpha 使用日期聚类 bootstrap。两题 smoke 只有一个决策日 cluster，配对区间为 null。样本数、分母和此限制应一起解释。
