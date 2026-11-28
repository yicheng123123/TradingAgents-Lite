# 测试报告

- 新增 6 个离线用例涵盖确定性分层抽样、未来数据拒绝、S3/C1 引用和原文位置、空检索、Flash Trader 历史经验送达、技术故障及结构修复、检查点与孤立账本记录拒绝。检索和旧冻结协议相关子集共 17 通过。
- 全套 pytest 最终 **250 通过、3 跳过**。跳过需要额外的 A1 本地 RAG 或 V2 集成快照。已有 Yahoo、金融 Development 及 Counterfactual Development fixtures 通过合法来源在独立 worktree 提供。真实凭证从测试环境移除。
- 早期全量测试发现旧 Counterfactual 冻结协议校验 FinanceBench、Provider 与 Workflow 的源码哈希。Pilot 改为独立适配器，原文件恢复到主线字节，完整测试复跑通过。未放宽历史断言。
- 实际语料离线 Smoke 使用经校验的 V3 SEC/IR 独立索引、全 T0–T4、独立的合成账本及 Mock 端点。Mock Token 与真实 BigModel 用量分开。
