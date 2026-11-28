# Phase 3.0 集成审计

基点 341c4414cb9e4cccc12dc2909e98183ef8479f94 已与 GitHub main 对齐。核验的历史提交分别为 A1-V3 ee7998438ffeee4d786eed46dcb925d1798afaf9、A1-V4 8b561a9d28b23aba73f2ecb900b1e757f89c8027、A1-V5.1 6d9e99f042e456b6bd5be0482c0a4a9b8db6dcdc、A1-V6 1b2eab0486e8faa114779f4e06febf04e832e8f0、A2-V2 554ee411455fe0b29d425419488da62fa27b5a0d。

| 部件 | 核验的主线 | 实验分支 | Pilot 集成 |
| --- | --- | --- | --- |
| RAG | R3 lexical/dense/RRF，scope 与版本控制 | V3 S3 全并集重排；V4 M3、V6 金融特征和结构索引 | 提取 S3 排序与模型入口，使用独立合法 V3 语料 |
| Context | FinanceBench 的 C1 query-aware 与专用投递；交易 RagView 使用前缀 | V5.1 校验 C1 并保留 | 复用通用 C1 片段，交易 evidence ID、位置、序列化预算与截断边界独立核验 |
| Memory | M3 事件过滤、SQLite、Qdrant、规则反思 | A2-V2 表示/排序/压缩探索保留 M3 | 双库存相同合法非 Pilot Warm-up，固定到 Pilot 之前 |
| Agent | D2、Verifier、风险、恢复与追踪；Flash V3 结构恢复 | A3 离线证据 | Flash Trader 可选 Memory 与 D2，同一 B2 Provider；按任务检查点与独立结算 |
| Data | 80 道 V3 Development 已有固定市场/SEC/XBRL，100 Holdout 仍封存 | 按公司划分的 FinanceBench 语料和标签 | 读取 Development 原哈希和截止预测包。当前结算另行密封 |
| Budget | 统一 SQLite 原子 TokenBudget、Flash 身份检查 | 原 Smoke 和 A1/A3 计量 | 复用实际项目账本，B2 500,000 子上限与请求对账 |
