# 测试报告

- 本次在隔离 experiment/final-agent-pilot worktree 中运行全量 pytest：**241 passed, 3 skipped, 0 failed**（23.08 秒）。运行时移除 BigModel 环境凭证，测试不产生真实 GLM 请求。
- 首次测试启动时，3 个 namespace-isolated Counterfactual 用例因 worktree 缺少 .venv/bin/python 而未能启动。将已有离线环境临时链接到 worktree 后全量复测通过；测试后已移除临时链接。
- 3 个跳过项依赖额外的 A1 本地 RAG 或 V2 集成快照。已有 Yahoo、金融 Development 及 Counterfactual Development fixtures 由合法本地来源提供。
- 早期全量测试发现旧 Counterfactual 冻结协议校验 FinanceBench、Provider 与 Workflow 的源码哈希。Pilot 使用独立适配器，原文件保持与基线一致；没有放宽历史断言。
- Pilot 离线集成 Smoke 使用经校验的 V3 SEC/IR 独立索引、全 T0–T4、独立合成账本与 Mock 端点。Mock Token 与 BigModel 真实用量分开。
- 首轮真实技术 Smoke 的工具上限问题和修复后回归分别记录在 smoke-tool-limit-audit.json 与 budget-reallocation-2026-11-29.json；首轮用量继续保留在共享账本。
- 本次只提交预算与结果文档，没有修改 Agent、RAG、Memory、MCP、Verifier、风险门或 TokenBudget 协议代码；没有新增依赖。
