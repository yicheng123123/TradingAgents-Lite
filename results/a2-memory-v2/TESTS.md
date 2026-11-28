# 测试报告

- 新增不变量测试 11 项通过。覆盖 PIT、结算、ticker 与显式跨股票范围、分区与实验组、schema 和引用、旧 Episode 保全、重复写入和重启、索引差异检测及重建、UTF-8 精确预算边界、完整 lesson 和数值、日期和不确定性、空结果、Mock 格式恢复和失败后禁止自动重复发送。
- 相关回归 `tests/test_memory_v2.py tests/test_memory.py tests/test_rag.py tests/test_a3.py` 为 **38 通过，1 跳过**。
- 完整 `pytest -q --tb=short` 为 **255 通过，3 跳过**，38.61 秒。测试进程移除了真实 API 凭证。历史 private worker 测试使用 pytest 临时生成的合成 fixtures。
- 首次运行缺 Yahoo manifest。之后的全套尝试也暴露了模型文件、开发证据、Development 沙箱路径和本地 worker Python 需求。最终在本 worktree 挂接 `data/yahoo-2024-v1`、`data/financial-evidence-dev-003`、`data/local-models/models` 和已存在的 `.venv`；为通过 runner 的路径校验，将明确允许的 Counterfactual Development fixture 复制到本 worktree 忽略目录。原数据和缓存没有改写。
- 报告不包含真实 Private Test Gold 或 V3 Holdout。没有通过放松旧测试断言修复环境问题。
