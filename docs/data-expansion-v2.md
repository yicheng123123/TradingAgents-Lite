# Data Expansion V2

V2 保留 A0–A3、B0、B1 的协议、归档和索引。新数据使用独立目录，真实 GLM 请求和推理 Token 均为 0。项目 Token 上限仍为 15,000,000。

## 股票池与行情

按给定的 20 家美国大型上市公司固定股票池。每只股票绑定历史申报人的 CIK 和一个 State Street 行业 ETF，并用 SPY 作为结算基准。迁移记录包括 META 在 2022 年由 FB 改为 META、WMT 在 2025 年转至 Nasdaq 时保留 WMT、GOOGL 使用 Alphabet Class A，以及 XOM 在样本期使用 34088。CIK 2115436 是 2026 年后的 XOM 继任申报人。这个按今天已知公司选取的等票样本存在幸存者、行业分类和大型股偏差。[SEC 股票代码表](https://www.sec.gov/files/company_tickers_exchange.json) · [Meta 代码变更](https://www.sec.gov/Archives/edgar/data/1326801/000132680122000070/may312022-exhibit991.htm) · [Walmart 转板公告](https://www.sec.gov/Archives/edgar/data/104169/000010416925000177/pressrelease-transfertonas.htm) · [ExxonMobil 继任公告](https://www.sec.gov/Archives/edgar/data/2115436/000119312526291990/d71068d8k12b.htm) · [行业 ETF](https://www.ssga.com/us/en/individual/capabilities/equities/sector-investing/sector-and-industry-etfs)

29 个 Yahoo Finance 日线序列全部可用。各有 1,330 个 XNYS 交易日，合计 38,570 条 OHLCV 和 38,570 条 Adjusted Close，记录 10 次拆股及 499 次股息事件。样本从 2020-10-01 到 2026-01-16，第一个 V2 任务前有指标预热，2025 年底后仍有结算缓冲。所有序列与 SPY 和交易日历对齐。校验发现 META 在 2022-02-03 收盘下跌约 26.39%，保留该 bar 并标记。未观察到重复、缺失或零成交量日，日线粒度无法排除盘中停牌。[Yahoo Finance 历史数据](https://finance.yahoo.com/quote/AAPL/history/) · [NYSE 特殊休市](https://www.nyse.com/publicdocs/nyse/markets/american-options/rule-interpretations/2025/National_Day_of_Mourning_20250102.pdf)

原始行情、供应商 Adjusted Close 与公司行为保存在本机。Yahoo OHLC 的拆股回溯量纲按任务时点还原后，仅将决策日及之前的 bars 交给工具。结算用同一快照的拆股尺度和 OHLC 价格收益。Adjusted Close 专用于审计，股息现金流、交易成本、借券和执行摩擦均未建模。历史快照可能包含供应商修订，量价也缺少当时可见性的独立认证。[行情版本与 SHA256](../results/data-expansion-v2-001/market-manifest.json)

## SEC、新闻和基本面

- 当前发布首先回放官方历史缓存，恢复 **11 份**不同版本的 SEC 原文。AAPL 5 份、MSFT 6 份。表单为 1 份 10-K、2 份 10-Q、7 份 8-K 和 1 份 8-K/A。Exhibit 99.1 原文与 SEC Company Facts 下载数量均为 0。实时批量采集命令已实现，本次执行环境无法读取配置了真实联系人邮箱的 `SEC_USER_AGENT`，所以未达到 400–600 份目标。[SEC 访问规则](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) · [原文版本清单](../results/data-expansion-v2-001/sec-documents.jsonl)
- 有网络配置时，采集器先读取官方 submissions 与 Company Facts，再优先下载 10-K、10-Q 及修订版，余量用于 Item 2.02 的 8-K 和同一 accession 的 Exhibit 99.1。单进程按至少 0.45 秒间隔发请求。403 或 429 后停止新 SEC 请求，缓存读取仍可继续。新抓取约束为最多 600 份原文。快照文件记录 URL、下载时间、SHA256 和缺失原因。[SEC 快照清单](../results/data-expansion-v2-001/sec-manifest.json)
- 原文按 accession、表单、文件日期和 acceptance 时间归档。eligibility 采用文件日期与纽约时区 acceptance 日期较晚者之后的第一个交易日 16:15 ET。版本历史初见时间仍为空。11 份原文以及进入 RAG 的 65 条资料均为 `historical_proxy`，`verified_known_at` 为 0。
- 新闻新增 20 家公司的官方 IR 或 Newsroom 标题和链接，发布日取发行人页面，保留标题级内容版本，发布时间时分和历史正文版本无法验证时留空。加上源自 Legacy 的可用标题，研究版共有 54 条新闻。20/20 是发行人来源覆盖，30 日时点覆盖率见下文。[官方新闻来源投影](../tasks/data-expansion-v2/issuer-news.json)
- 基本面模块可消费 Company Facts，并可从官方原文内联 XBRL 提取同一 issuer 的无维度 US GAAP 数字。本次从 AAPL 和 MSFT 文档分别恢复 81、68 行原始字段，绑定原始 accession、来源 URL、SHA256 和有效时点。Development V2 的季报口径中，列出的 14 个指标里有 12 个各覆盖 8/80 题，Total Debt 与 Debt / Assets 均为 0/80。年初至今和年度覆盖逐项可查。已有的最近缓存报表可能落后于任务期当时实际最新的公司披露。[基本面来源清单](../results/data-expansion-v2-001/fundamentals-manifest.json) · [分字段覆盖](../results/data-expansion-v2-001/summary.json)

## 冻结任务与隔离

按股票在预先定义的等交易日区间抽样。目标位置由固定哈希轻微错开，偶数区间优先靠近已经合规的财报事件。日期选择只用交易日历、已知披露和当日以前的 21 个完整行情日。收益与结算标签未参与日期选择。用于本次事件抽样的 SEC 元数据来自当时的缓存版本，随任务清单冻结。

- Offline Extended 200 题覆盖 2021–2023 年，每只股票 10 题。其中 2 题靠近冻结披露事件。20 日波动率分档低于 1%、1–2%、2–4%、至少 4% 分别为 30、99、64、7 题。
- Development V2 80 题覆盖 2024 年，每只股票 4 题。有 2 题靠近冻结披露事件。相同波动率分档为 9、40、26、5 题。
- Final Holdout 100 题覆盖 2025 年，每只股票 5 题。只冻结任务 ID、股票、日期、五日周期和输入摘要。UI、A3 执行与结算入口封存该分区。发布物没有 Holdout 结算标签或 Memory。

原有 20 题保持原文件和原周期。新题没有重复股票日期，380 题之间有 0 个重叠的同股票结算窗口。与 Legacy 一起审计时发现 2 个重叠窗口。本轮三个分区的配额均填满。[任务冻结清单](../tasks/data-expansion-v2/manifest.json)

## 本地 RAG 与离线评测

独立 A1 兼容索引使用 HTML 结构感知分块、现有 BGE 本地权重和向量缓存、Qdrant、BM25 与 RRF。65 份可用材料产生 362 个 chunk，覆盖 20 只股票和 8 个行业。每份文档最多取均匀分布的 72 个 chunk。索引占 2,139,137 字节，实测构建 4.489 秒，峰值 RSS 298,936 KiB。600 份源文档的构建耗时需要采集完成后另测。未重跑 FinanceBench 或修改原有冻结索引。[RAG 清单](../results/data-expansion-v2-001/rag-manifest.json)

行情本地快照占 13,177,127 字节，SEC 缓存回放读取 8,774,820 字节，耗时 2.876 秒。评测和输入审计耗时 4.997 秒，峰值 RSS 316,252 KiB。

覆盖计数使用既有 SEC 180 日、新闻 30 日和合规 cutoff。Offline Extended 的合规财报、新闻、RAG 候选覆盖分别为 **0/200、6/200、6/200**。Development V2 为 **6/80、2/80、6/80**。逐题候选数以及模型入口前的 PIT 审计已归档，两组共 **0/280** 题发现泄漏。[Offline 覆盖](../results/data-expansion-v2-001/offline-coverage.jsonl) · [Development 覆盖](../results/data-expansion-v2-001/development-coverage.jsonl)

所有基线都在 Offline Extended 完成 200/200。共同规则为下一个共享交易日开盘进入，第五个共享交易日收盘结束，以方向乘以相对 SPY 超额价格收益记 Alpha，非 Hold 方向受同一 4% 日波动率风险门约束。主动方向的命中率以个股价格收益方向为准。置信区间从 117 个决策日聚类中固定种子重采样 5,000 次。

- 五日动量覆盖 165/200，方向命中 78/165，平均 Alpha −0.098%，95% 日期聚类区间约为 [−0.550%，0.384%]。
- 20 日均线覆盖 193/200，方向命中 94/193，平均 Alpha +0.136%，区间约为 [−0.350%，0.633%]。
- Always Hold 覆盖 0/200，平均 Alpha 0%。随机方向用 5 个固定种子，覆盖各为 193/200，平均 Alpha 约在 −0.293% 到 +0.044%。

逐题结果、行业分组、随机对照及置信区间均公开。独立任务的结果没有拼接为组合净值。Holdout 实际结算 0 题。[评测摘要](../results/data-expansion-v2-001/summary.json) · [逐题基线](../results/data-expansion-v2-001/offline-cases.jsonl) · [测试记录](../results/data-expansion-v2-001/tests.txt)

## 复现

在 WSL 项目根目录运行。真实采集前在本机 Shell 安全设置 `SEC_USER_AGENT`，值由项目名和授权联系人邮箱组成。公开账户配置文件和命令历史无需保存个人邮箱。下载、正文、模型权重和中间索引仅保留在工作区的 `data/`。

```bash
uv sync --locked --group data-v2 --group ui
uv run --locked --group data-v2 python -m tradingagents_lite.data_v2_market
uv run --locked --group data-v2 python -m tradingagents_lite.data_v2_sec
```

缓存回放可以为 SEC 命令添加 `--offline`，缺失文件会明确归入报告。已冻结任务可直接用于评测。要重建本次抽样，应使用相同市场输入 SHA256、Legacy 任务和冻结采样元数据。为新供应商版本建立独立目录并保留已归档结果。

```bash
uv run --locked --group data-v2 python -m tradingagents_lite.data_v2_tasks \
  --filings tasks/data-expansion-v2/sampling-filings.jsonl --output data/data-expansion-v2/freeze-replay
uv run --locked --group data-v2 python -m tradingagents_lite.data_v2_corpus compile
uv run --locked --group data-v2 python -m tradingagents_lite.data_v2_corpus index
uv run --locked --group data-v2 python -m tradingagents_lite.data_v2_evaluation \
  --output data/data-expansion-v2/eval-replay
uv run --locked --offline --group data-v2 --group ui pytest -q
```

已有索引可以直接用于评测。索引构建拒绝覆盖既有目录。新版本发布前，任务、市场、财报、基本面和索引的签名都应对照源版本。
