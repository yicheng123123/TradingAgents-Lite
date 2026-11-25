# Streamlit 离线验证

`uv run --locked --group ui pytest -q` 共通过 77 项，包含 7 项 Streamlit AppTest。完整结果见 [测试输出](tests.txt)。

AppTest 覆盖冻结任务执行、轨迹显示、MCP/RAG/Memory 集成、归档指标、只读 SQLite 账本、LLM 授权门禁、错误脱敏和页面重跑。

真实 Edge 浏览器打开本机 Streamlit 应用时，Research 和 Agent Trace 使用同一个离线运行。冻结题目为 `dev-20`，配置为 MCP、R1、M2、D1、Verifier 和 F1。该运行有 3 个证据 ID、2 次离线辩论角色调用和 3 条时点合规的历史 Memory。

- [Research 截图](../../docs/screenshots/research.png)
- [Agent Trace 截图](../../docs/screenshots/agent-trace.png)

本次验证的真实 GLM 请求和 Token 消耗均为 0。测试使用冻结数据及离线或 mock 后端。
