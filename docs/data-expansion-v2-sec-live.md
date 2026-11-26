# Data Expansion V2 · SEC 联网补全 003

本次从本地配置加载 SEC User-Agent。采集器串行访问官方端点，复用历史缓存，并输出按阶段计数和预计剩余时间。联网下载 **718** 个响应，命中缓存 **35** 次。SEC 对 403 或 429 的停止条件未触发。失败记录为 no_verified_99_1_link 7、network_unavailable 2。[SEC 公平访问规则](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) · [采集清单](../results/data-expansion-v2-sec-live-003/sec-manifest.json) · [进度记录](../results/data-expansion-v2-sec-live-003/progress.jsonl)

## 数据与验收

样本期共恢复 **565** 份正式文件，内容哈希去重后 **565** 份。另有 **29** 份 2020 年财报用于输入预热。样本期文件类型为 10-K 100、10-K/A 4、10-Q 298、10-Q/A 2、8-K 84、EX-99.1 77。按年计数为 2021 年 120 份、2022 年 121 份、2023 年 121 份、2024 年 118 份、2025 年 85 份。Company Facts 规范化后共 14,964 条事实。[原文版本元数据](../results/data-expansion-v2-sec-live-003/sec-documents.jsonl) · [基本面清单](../results/data-expansion-v2-sec-live-003/fundamentals-manifest.json)

- **SEC 原文公司覆盖** 20/20，门槛 20/20，通过。
- **Company Facts 公司覆盖** 20/20，门槛 20/20，通过。
- **Development RAG 候选覆盖** 80/80，门槛 72/80，通过。
- **Offline Extended RAG 候选覆盖** 200/200，门槛 180/200，通过。
- **时点泄漏** 0，通过。

Development V2 和 Offline Extended 分别有 67/80、166/200 题可在截止时点取得季度收入。逐字段和逐题覆盖已归档。[验收摘要](../results/data-expansion-v2-sec-live-003/summary.json)

Pfizer 的业绩指引 8-K 及同一 accession 的 Exhibit 99.1 单独核验并补入后，20 家均有 10-K、10-Q、8-K 和 Exhibit 99.1。

## 独立索引及复验

RAG 003 包含 **563** 份材料、**7,236** 个 chunk。本地索引占 46,388,225 字节，构建耗时 821.1 秒，写入进程峰值 RSS 275,628 KiB，单个工作进程的最高峰值 RSS 为 1,200,640 KiB。使用 4 个本地嵌入工作进程和单一 Qdrant 写入进程，每份文档最多均匀选取 16 个分块。只有到 2024 年末前已合规的资料进入研究索引。SEC 披露的历史可用时点继续使用既定交易日代理规则。[索引清单](../results/data-expansion-v2-sec-live-003/rag-manifest.json) · [索引进度](../results/data-expansion-v2-sec-live-003/rag-progress.jsonl)

Offline Extended 200/200、Development V2 80/80 完成确定性复验。Final Holdout 结算 0 题，真实 GLM 请求和 Token 消耗均为 0。隔离副本使用新的数据版本完成 **110 项回归**。原 400 题、前端、历史结果及历史缓存保留。[测试记录](../results/data-expansion-v2-sec-live-003/tests.txt)

本地版本目录为 `data/data-expansion-v2-003`。Git 仅归档来源清单、覆盖、进度和测试结果。索引复现用 `data_v2_corpus index --version rag-expansion-v2-003 --max-chunks 16 --workers 4 --progress`，以 003 的 SEC 及新闻输入清单为准。[续跑与复现](data-expansion-v2-sec-resume.md)
