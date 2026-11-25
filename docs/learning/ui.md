# 本地研究界面

## 启动

在 WSL 的项目根目录执行：

~~~sh
uv run --locked --group ui streamlit run streamlit_app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
~~~

在本机浏览器打开 Streamlit 输出的 localhost 地址。默认使用 Offline，启动不需要 API Key。冻结行情和 A0.5 证据路径沿用项目约定。RAG 用本地建好的 A1 索引及模型缓存。可用 `talite rag-index` 生成索引。Memory 启用后会读取已存在的本地冻结 Development 回放存储，将符合当前时点的历史记录复制进本次的独立 scope。没有历史存储时使用空历史。

## 数据流

- Research 选择冻结任务、执行后端和实验配置。明确执行才创建 UUID 和独立输出目录。LLM 还要求用户启用确认框、API Key 存在、现有总账与 A3 预算同意。页面只显示估算上界和状态。
- Agent Trace 读取已归档的节点顺序、工具结果、Bull/Bear 论点、耗时、错误与重试摘要。
- Evidence & Memory 从同一任务输出读取当前证据、各阶段检索排名、历史引用和分数。
- Benchmark 为静态归档建立适配。A0/B0 是规则研究，A1/A2/A3 各自显示分母和来源。B1 真实评测状态为 Pending。
- Token Dashboard 的项目区用 SQLite mode=ro 打开实际账本。未知或失败尝试保留保守记账。已知费用按历史标准价估计，发票字段保持不可用。没有账本时显示配置额度。

页面刷新仅依据 session state 或 URL 中的有效 run_id 加载已经生成的任务记录。模型调用属于执行按钮分支。研究数据写入 data/ui-runs 的新目录，正式 Benchmark 归档由只读适配读取。

Research 的任务集选择支持 Legacy 和 Development V2，并按 V2 归档显示本地证据覆盖。V2 单题使用对应的行情、证据、基本面、RAG 和独立 Memory，Final Holdout 不在选项中。[V2 数据说明](../data-expansion-v2.md)

## BigModel Account

在启动应用的 WSL 环境中设置现有的 `TALITE_API_KEY`，然后在 Token Dashboard 点击 **Refresh BigModel account**。打开页面、切换页面和普通重跑都只读取当前会话状态。余额与资源包各自缓存成功结果 60 秒。失败时显示 **Unavailable** 或带原同步时间的 **Stale cache**。更换密钥会丢弃旧会话数据。

账户查询仅对 `open.bigmodel.cn` 发出两个只读 HTTPS GET，请求带 Bearer 凭据。客户端禁用重定向和环境代理。密钥与原始响应留在请求过程中，解析后的账户值只保存在当前 Streamlit 会话内。两个端点属于控制台，Bearer 权限和成功响应格式可能变化。[社区余额实现](https://github.com/hucuyuu/zhipu-balance/blob/main/scripts/check_balance.py)

- 人民币区显示账户报告中的 `availableBalance`、`totalSpendAmount` 和 `giveAmount`。赠送金额按账户报告显示，赠金剩余余额需要单独的字段。合法数值 0 显示为 ¥0.00，缺失值显示 **Unavailable**。
- 资源包区对照智谱控制台将 `tokenBalance` 显示为当前余额，将 `availableBalance` 显示为可用余额。只有按 Token 计量、处于生效状态、未到期且明确列出 `glm-5.3` 的资源包进入统计。到期取 `packageExpirationTime`，无时区的控制台时间按 UTC+08:00 解释。只读第一页，若返回的总行数表明还有其他行，或部分适用范围无法确认，汇总显示 **Unavailable**。[智谱控制台资源包页面](https://bigmodel.cn/finance-center/resource-package/package-mgmt)
- 原始资源包总额度需要独立字段。当前可核实的接口字段只覆盖剩余和可用余额，原始总额度显示 **Unavailable**。Coding Plan 配额由控制台单独查询。项目 15,000,000 Tokens 硬上限及各阶段额度始终取自本地项目配置和账本。

## 代码导读

- streamlit_app.py 的 research 第 33 行、agent_trace 第 115 行、evidence_memory 第 144 行、benchmark 第 183 行、bigmodel_account 第 237 行及 local_token_budget 第 277 行。
- tradingagents_lite/research_ui.py 的 archive_for 第 48 行、load_run 第 57 行、start_run 第 71 行、token_dashboard 第 109 行。
- tradingagents_lite/zhipu_account.py 的 `_balance` 第 86 行解析人民币金额，`_packages` 第 123 行筛选资源包，`_get` 第 158 行执行请求，`account_snapshot` 第 199 行处理点击和会话缓存。

## 本地验证

~~~sh
uv run --locked --group ui pytest -q
~~~

AppTest 驱动真实应用代码，检查无 Key、离线冻结任务、STDIO、错误显示、归档指标一致性和只读账本。浏览器截图使用真实本地应用与实际生成的离线任务。

账户测试使用内存 HTTP 模拟。AppTest 检查手动刷新、缺失密钥、已知零值、未知金额和项目上限显示，并比较刷新前后的 SQLite 文件哈希。

## 面试问题

1. 为什么需要 session state 和持久 run_id？
2. 何时应该创建新的 run_id？
3. 刷新页面和显式点击执行分别触发哪些分支？
4. 为什么不能用一个可写的 TokenBudget 对象给只读 Dashboard 查数据？
5. pending reservation 可能在哪些情况下持续存在？
6. 实际账单和基于 usage 的价格估算为什么有差异？
7. 为什么历史 Memory 复制到独立 scope？
8. AppTest 与实际浏览器截图分别验证什么？
9. 为什么 Trace 展示摘要和 ID，而非完整请求？
10. 如何保证页面归档数字与机器读的 JSON 保持一致？
11. 为什么资源包余额与可用余额分开显示？怎样证明一个资源包适用于当前模型？
12. 为什么余额和资源包需要独立缓存？如何防止密钥变更后显示上一个账户的数据？

## 自行完成的练习

1. 为 Benchmark 页新增一个依赖明确分母的筛选器，写一个 AppTest 验证。
2. 用临时 SQLite 造一条未结算预留，检查 Token 面板的分桶。
3. 设计一个对比两个独立研究运行的视图，处理缺失指标和不同模式。
