# 上游审计与复用

上游 [TradingAgents](https://github.com/TauricResearch/TradingAgents/tree/9fa1196fffbcfd9de5146d2dd2869df84ab99838)，本机清洁检出。完整 Commit SHA 为 `9fa1196fffbcfd9de5146d2dd2869df84ab99838`。上游为 Apache-2.0，该提交无独立 NOTICE 文件。Lite 保留许可证文本并添加来源声明。Lite 的 Python、prompt、测试与评测机制独立实现。

| 上游路径及位置 | 审计观察 | Lite 第一期 |
|---|---|---|
| `graph/setup.py:53`、`:145`，`analyst_execution.py:49`、`conditional_logic.py:12` | 私有分析师子图、有界工具循环、并行分析、研究辩论与风险辩论路由 | 保留有界工具和分开的 Market / Trader / Risk。多分析师和一轮辩论接入规划见路线图 |
| `agents/analysts/market_analyst.py:15`、`agents/analysts/turn.py:13`、`agents/schemas.py:146`、`agents/trader/trader.py:20`、`agents/managers/portfolio_manager.py:23` | 技术数据工具、结构化 proposal、风险意见合成 | Pydantic 约束动作和引用，保留 Trader 提案，风险层用固定阈值和证据校验 |
| `graph/trading_graph.py:130`、`:309`、`memory/log.py:93`、`memory/reflection.py:36`、`memory/settlement.py:60` | 历史经验按结算日期过滤，未成熟收益不写入可用经验 | A0 无记忆。SQLite 保存本轮研究记录，未来 A1/A2 使用独立的可见性过滤与经验存储 |
| `backtest.py:143`、`:206` | 固定网格、隔离运行、错误记录，按成熟收益汇总 | 固定任务 JSONL，从下个交易日开盘到第 N 日收盘。上游使用下一交易日收盘进入，收益指标不可直接并表 |
| `dataflows/date_window.py:79`、`vendors/yahoo/ohlcv.py:229`、`vendors/sec_edgar.py:154`、`dataflows/router.py:200` | 工具访问日期由状态约束。历史 OHLC 截断，SEC 财报依 filing date 过滤，各类供应商路由 | 图只接收已经截断的市场视图，模型不能指定日期或外部数据源。财务文件与新闻留待引入可靠发布时点 |
| `default_config.py:77`、`tests/test_backtest.py`、`tests/test_tool_date_enforcement.py` | 统一配置入口，测试运行隔离和日期注入。还盘点了 `test_fundamentals_lookahead.py`、`test_news_lookahead.py`、`test_memory_pointintime.py` | 锁依赖、保存预算、模型和数据哈希。纯离线测试，增加未来行情变更不影响当前图的检查 |

本次以源码审计为主，上游自身的测试和评测指标未纳入 Lite 的成绩。
