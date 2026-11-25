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
- Token Dashboard 用 SQLite mode=ro 打开实际账本。未知或失败尝试保留保守记账。已知费用按历史标准价估计，发票字段保持不可用。没有账本时页面只展示配置额度。

页面刷新仅依据 session state 或 URL 中的有效 run_id 加载已经生成的任务记录。模型调用属于执行按钮分支。研究数据写入 data/ui-runs 的新目录，正式 Benchmark 归档由只读适配读取。

## 代码导读

- streamlit_app.py 的research 第 32 行、agent_trace 第 104 行、evidence_memory 第 133 行、benchmark 第 172 行、tokens 第 216 行。
- tradingagents_lite/research_ui.py 的archive_for 第 30 行、load_run 第 39 行、start_run 第 53 行、token_dashboard 第 74 行。

## 本地验证

~~~sh
uv run --locked --group ui pytest -q
~~~

AppTest 驱动真实应用代码，检查无 Key、离线冻结任务、STDIO、错误显示、归档指标一致性和只读账本。浏览器截图使用真实本地应用与实际生成的离线任务。

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

## 自行完成的练习

1. 为 Benchmark 页新增一个依赖明确分母的筛选器，写一个 AppTest 验证。
2. 用临时 SQLite 造一条未结算预留，检查 Token 面板的分桶。
3. 设计一个对比两个独立研究运行的视图，处理缺失指标和不同模式。
