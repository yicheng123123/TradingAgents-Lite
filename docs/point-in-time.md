# A0.5 时点金融证据协议

研究批次为 20 个固定 Development 单元，NYSE 时点为决策日纽约时间 16:15，实际 UTC 随 DST 变化。行情快照的 SPY 交易日为代理日历。附件含下载与证据 SHA-256，时间统一转为 UTC。原始行情和 A0 留存在既有目录。

## 数据来源与可见时间

- **SEC EDGAR** 从 Submissions API 获取 accession、表单、报告日期、filing date 和 acceptance timestamp，随后按 accession 下载主文档。本批冻结 11 份原文并保存原文 SHA-256。SEC metadata 中的 report date 在 10-K/10-Q 表示报告期终点，8-K 则保留 SEC 对应报告日期。Filing date 与 acceptance 的纽约日期取较晚者，下一观察交易日 16:15 才进入证据。没有认证的实际公开时间时保持 `public_available_at=null`、`first_seen_at=null`、`published_at=null`。主文档 HTML 取短片段，保留原文件路径和 hash 关联，附件及完整口径化会计提取留待后续。
- **Apple Newsroom** 从可访问的分页存档取 URL、发行人标题、日期与类别。本批 34 条标题。存档只有日期，按日历下一交易日放行。标题、元数据投影和原始索引响应分别有哈希。缓存只保存投影，不缓存页面或新闻全文。动态页码会随新文章移动，需要从实际窗口调整参数。
- **Microsoft** 公司博客和 Source 月度入口返回 403。四篇可读取的 Microsoft Security Blog 文章带历史发布时间，但现存版本的 `article:modified_time` 均在 2025 年。将四个版本标为 `unavailable`。不假设今天看到的标题在 2024 年已知。GDELT DOC 2.0 在实际探测返回 HTTP 429，且其文档的任意历史日期查询受滚动窗口限制，未纳入生产证据。

SEC 请求使用明确标识项目和用途的 User-Agent，支持 `TALITE_SEC_USER_AGENT` 提供有权用于请求的联系人标识。采集进程串行执行 SEC 请求，间隔至少 1.25 秒。遇到限流或暂时的 5xx 最多三次尝试并尊重 Retry-After。403 原样作为不可用报告，避免对同一拒绝响应自动重试。本机 uv 下载版 Python 3.12.14 在 SEC 正文窗口返回 403，系统 Python 3.12.3 使用相同项目 User-Agent 能成功访问官方接口。缓存重放后续不再访问网络。

来源说明：[SEC EDGAR API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)、[SEC Fair Access](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)、[Apple Newsroom](https://www.apple.com/newsroom/archive/)、[Microsoft Security Blog](https://www.microsoft.com/en-us/security/blog/)、[GDELT DOC 2.0](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)。

## 证据边界

`EvidenceStore` 用 ticker、source type、document_id 与内容哈希构造稳定证据 ID。不同哈希版本不会合并。同一文档多版本如果共享同一个代理可见时间且无法证明对应内容当时存在，检索层同时扣留这些冲突代理版本。只有明确的版本观察时间、哈希绑定与来源佐证才可使用 `verified_known_at`，本轮没有此类实测记录。

证据窗口为 eligible_at 起 180 天内的 SEC 文档和 30 天内的新闻，且 `eligible_at <= decision_cutoff`。实际采集范围还受交易日快照首日限制。无可用版本返回 `unavailable`。`ingested_at` 仅为采集审计字段，从不参与历史资格判断。原始请求状态、SHA-256 与来源可用性进入 manifest，未来文档、原始 ingestion 时间与网络状态不送入历史 Agent。

Market Analyst 只持有当前 ticker / cutoff 的 `EvidenceView`。可选 `financial_evidence` 工具暴露来源类型和数量参数，无法覆盖 ticker 或 cutoff。返回有限片段和已观察证据 ID，Trader 只能引用被 Market Analyst 选中的 ID。后续向量索引和 MCP 应调用同一可见性约束，避免索引命中后再补过滤。

本轮不能认证无历史供应商修订、无页面编辑、初次公开可见时间，或模型不会凭预训练知道未来。覆盖率衡量窗口内至少有一个可用版本，不能推导来源完整性、事实命中率或收益。

## 重放

```sh
uv sync --locked --python /usr/bin/python3
uv run --locked pytest -q
uv run --locked talite evidence-fetch --tasks tasks/development.jsonl \
  --data data/yahoo-2024-v1 --cache data/evidence-cache-001 \
  --output data/financial-evidence-dev-replay --offline
uv run --locked talite evidence-coverage --tasks tasks/development.jsonl \
  --evidence data/financial-evidence-dev-003 --output results/evidence-dev-a05-new
```

请为新重放选择新输出目录。首次准备时去除 `--offline`，可按当日发行人分页传 `--apple-pages`。本机真实数据归档与缓存均在忽略的 `data/`，可提交 manifest、覆盖 CSV、JSONL 和报告，原文与版权正文不会随 Git 分发。离线 fixtures 全部为清楚标注的合成测试输入。
