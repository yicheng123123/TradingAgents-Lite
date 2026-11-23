# TradingAgents-Lite

一个独立的 Python 金融 Agent 研究项目。第一阶段实现 Market Analyst → Trader → Risk Manager → Backtest，使用 LangGraph、Pydantic 和 SQLite。普通运行使用冻结历史行情和可审查的五日动量规则，支持通过显式配置切换到现有 OpenAI-compatible Chat Completions。

上游为 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)，设计取舍和授权记录见 [上游审计](docs/upstream-audit.md) 和 [NOTICE](NOTICE)。项目处于第一阶段，后续模块见 [路线图](docs/roadmap.md)。

## 运行

```sh
cd ~/projects/tradingagents-lite
uv sync --locked
uv run pytest -q
uv run talite snapshot --tasks tasks/smoke.jsonl --output data/yahoo-2024-v1 \
  --start 2023-11-01 --end 2024-04-30
uv run talite run --backend offline --tasks tasks/smoke.jsonl \
  --data data/yahoo-2024-v1 --output results/smoke-new --seed 42
```

快照命令只读取得 AAPL、MSFT、SPY 日线，生成原始响应、规范化 JSONL 和 SHA-256 清单。已有快照可直接复用，输出目录需要新建。供应商重新下载的内容可能与先前哈希不同。原始行情只作为本地研究缓存。结算与工具隔离原理见 [架构和指标](docs/architecture.md)。

`run` 保存 `run.json`、`tasks.jsonl`、`summary.csv`、`REPORT.md` 与本地 SQLite 账本。方向正确率只计算有结算结果的非 Hold，覆盖率以全部任务为分母。对照为始终 Hold，其方向命中率无分母；配对 Alpha 差以 Hold 的零敞口为参照。失败样本单独计数。SQLite 和所有文件只写新结果目录。

使用 LLM 时，在当前 shell 设置 `TALITE_API_KEY`、`TALITE_BASE_URL` 和 `TALITE_MODEL`（参见 `.env.example`）。将命令的 `--backend offline` 改为 `--backend llm`，并改用新的输出路径。这会产生模型请求。默认 temperature=0，模型 seed 可通过 `--provider-seed` 传给支持它的供应商。离线规则运行不会访问模型 API。

这是历史行情快照研究。模型和数据的时点限制、不能实施的交易假设及收益口径均记录在报告中。请将这套研究结果与真实交易和上游评测分开。
