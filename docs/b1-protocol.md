# B1 实验协议

## 冻结输入

- Development 集：20 个固定 ticker/date 单元，SHA-256 为 2d0e4f92d583d64ceba729b4e3ffa0004b3bd29a5a00d990434dcdab5fceedae。
- 行情 manifest：f5e4c5095953ce737b7b99bba71b2ad26a9f30d2f34b9fba8320027e46e0f3cb。
- A0.5 evidence manifest：b46f9ef2ebc1c08022500282147940985eb9d3ceabe8cdaf78eec6efa190e4dc。
- Evidence JSONL：b0ffce595575257231432650495d756681a9f4ac10f5deee972110c7a7b7a532。
- 不读取 Holdout，不改写 A0 / A0.5 归档。

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
- 输出上限为每次请求 1,200 tokens。
- Market Analyst 最多两轮工具调用，每轮最多两个工具；之后强制要求 final MarketReport。随后最多一次 Trader 请求。
- 每次逻辑 API 调用最多重试一次，只重试 timeout / connection / rate-limit / HTTP 408 / 5xx；SDK 内重试关闭。
- API 请求 timeout 25 秒。messages + tools 的 UTF-8 序列化上限为 32,768 bytes，超限时在发请求前失败。
- 生成不设置供应商 seed；统计 seed 固定为 42。
- Function tool 本地执行为同步、限参数且本轮没有远程工具。若函数抛 TimeoutError，会记录 tool_timeout；该测试是异常路径模拟，不代表对任意未来远程工具提供通用中断保证。

## 成本预算

智谱官方价目表在 2026-11-24 页面列出 GLM-5.3 输入 ¥8 / 百万 tokens、输出 ¥28 / 百万 tokens。代码按照 32 KiB prompt 上限、每次 512 个协议开销 tokens、1200 输出 tokens、全部逻辑调用均用满一次重试计算保守标准价封顶：

| Scope | 每组任务 | 成对逻辑调用上限 | HTTP 尝试上限 | 成本封顶 |
|---|---:|---:|---:|---:|
| Paired smoke | 2 | 16 | 32 | ¥9.594880 |
| Full Development | 20 | 160 | 320 | ¥95.948800 |

粗略预期 smoke 约 ¥0.2–0.5，采用每个任务组 3 个请求、每次 1–3k 输入与 200–500 输出 token 的假设；这是未测算的预测。实际 token usage 以接口返回值为准。若出现重试、调用失败或响应未返回 usage，API 成本标为 unknown，因为服务端可能已经计费。价目表、账户折扣、缓存和账单可能不同。

Full Development 必须以 --budget-cap-cny 显式提供不少于计算封顶的额度，未提供或低于封顶时 CLI 拒绝运行。扩大范围前先审阅 smoke 用量，并由运行者确认费用预算。

## 结算与失败

结算逻辑与 B0 相同：决策后下一共同交易日开盘入场、第 N 个共同交易日收盘退出；Alpha 为方向乘以资产价格收益减 SPY 价格收益，不计交易费用、股息现金流和融券成本。

状态非 ok 的任务属于失败，不进入 Alpha 或 Directional Accuracy 的分子/分母；Task Completion Rate 保留完整任务分母。失败任务以模型错误、workflow 错误或运行时错误写入 trajectory。不能把失败转换成正常 Hold 计分。

方向命中率报告非 Hold 分母和 Wilson 95% 区间；平均 Alpha 使用日期聚类 bootstrap。两题 smoke 只有一个决策日 cluster，配对区间为 null。样本数、分母和此限制应一起解释。
