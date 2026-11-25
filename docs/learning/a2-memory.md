# A2 长期 Episodic Memory

## 数据流

1. Runner 按纽约决策 cutoff 排序。每道题的 LangGraph State 保存当前行情、证据、工具轨迹和提案。
2. Trader 决策前，MemoryReplay 推进已经结算的事件，并用当前 ticker 和 MarketReport 查询 SQLite 与 Qdrant。短期 State 只接收本题选中的历史经验。
3. Trader 可以用旧经验提醒自己核查风险。历史 evidence ID 和旧市场结果放入独立上下文字段。引用校验仍使用当前任务实际观察到的 evidence。
4. 决策完成后，runner 调用已有独立结算接口，将结算输出放入 runner 的事件队列。结果可见时生成反思，写入 SQLite 版本记录，再更新 Qdrant。
5. 重启后 SQLite 和 Qdrant 都从磁盘加载。检测工具可找到索引缺失、孤立或不一致的记录，并从 SQLite 恢复。

## 模块输入输出

- `Episode` 保存任务 ID、ticker、决策时间、结算时间、反思产生时间、eligible 时间、action、决策依据 ID、结果、反思、实验分区和组别。逻辑 ID 由任务范围与决策时点生成，版本哈希由内容生成。
- `MemoryStore` 的 SQLite 保存权威记录，Qdrant 保存同一版本的查询向量。重复写入同一版本会去重，向量写入可重复。更新是新增版本，早期 cutoff 还能查询旧版本。
- `MemoryReplay` 使用离线结算事件时钟。确定性反思使用结算事件作为模拟产生时刻。供应商反思若接入实际历史流程，需要记录实际生成时刻。
- `deterministic_reflection` 同时考虑决策方向、证据、风险否决与实际结算。盈利或亏损单独不足以判断决策质量。
- `model_reflection` 通过相同 ModelBackend 和 TokenBudget 入口，输出要通过 Reflection schema。本阶段在 Mock 中验证 JSON 校验和 A2 记账，CLI 回放使用确定性反思。

## 时点安全和排序

可见条件包括同 ticker、同 dataset partition、同 experiment arm、`decision_time < cutoff`、`outcome_available_at <= cutoff`、`eligible_at <= cutoff`。版本按合规的最近 eligible 时间选择，再把可用版本的 ID 交给 Qdrant 候选过滤。相同决策时间的任务可以看见早先已经结算的经验，当前同时间的新决策结果仍在未来。

- M0 保持无记忆控制组。
- M1 按可用时间召回近期经验。
- M2 查询 BGE 的余弦相似度，转换为 `s=(cos+1)/2`。
- M3 在语义候选上按 `0.65×s + 0.25×exp(-age_days/30) + 0.10×q` 排序。证据质量 `q` 是观察引用的 PIT 分数均值，已认证为 1，历史代理为 0.5，冻结行情为 0.75，不可用为 0。没有观察引用时取 0。未携带质量字段的导入记录使用最多三个去重引用的计数代理。
- M3 压缩经验文字，保留原历史引用 ID 和时间。每题最多召回三条，默认注入上限 1,800 UTF-8 字节。Provider 另对 A2 整个请求设置 20,000 字节上限。

SQLite 优于单一向量索引的原因是它可以保存结构化结果与多个历史版本。两库写入跨进程中断时，SQLite 已提交的版本会留下可检测的缺失投影。选择本地 Qdrant 可以复用 A1 的模型、持久化和过滤能力。其他方案包括只按时间查 SQLite、用离线批处理生成投影，或使用支持向量的单库事务。不同方案需要比较修复复杂度和真实召回质量。

## 本地复现和评测

~~~sh
uv run --locked talite memory-benchmark --output data/a2-memory-local
uv run --locked talite memory-index --store data/a2-memory-local/store
~~~

Development 回放用固定 20 个任务和五日动量规则，每种模式单独运行。检索标签使用与当前已归档决策 action 相同的、同 ticker 的可用旧任务。标签由评测端事后计算，当前动作和未来收益不进入检索查询。Recall@K 按有相关记录的查询取宏平均，控制组使用与 M1 相同的分母。该标签用于检查跨任务复用，独立合成集另外验证一条明确的语义相关案例、时间和分区隔离、重启、幂等、历史版本和索引修复。检索结果、候选分数、引用、上下文记录、延迟和字节数写入逐题和汇总文件。实际 GLM 策略效果要由后续受控消融衡量。[本地 Memory 评测](../../results/a2-memory-local-001/REPORT.md)

## 常见故障

- 把结算结果预先写入可见索引会泄漏未来。事件只按 cutoff 推进。
- 把历史经验拼进当前 evidence 容器会误放行历史引用。Trader 的经验与当前 evidence 分别传递。
- 实验组和分区用同一个索引时容易混用记录。SQLite 与 Qdrant 都需要独立范围条件。
- 中断后向量缺失会阻止语义查询。运行 `memory-index --rebuild` 后再查询。
- 长记录可能塞不进字节上限。输出跟踪候选和实际注入量，保留能安全装入的记录。

## 源码导读

- [memory.py](../../tradingagents_lite/memory.py#L27) 的 `Episode` 在第 27 行，确定性与模型反思在第 67、78 行。
- [memory.py](../../tradingagents_lite/memory.py#L125) 的 `put` 在第 125 行，`rebuild`、`visible`、`recall` 在第 148、157、172 行。
- [memory.py](../../tradingagents_lite/memory.py#L255) 的 `MemoryReplay.advance` 在第 255 行。
- [runner.py](../../tradingagents_lite/runner.py#L185) 的 `run` 在第 185 行。窗口结算与反馈入队都在逐题循环内。
- [provider.py](../../tradingagents_lite/provider.py#L286) 的 `trader_with_memory` 在第 286 行。
- [memory_benchmark.py](../../tradingagents_lite/memory_benchmark.py#L58) 的离线评测入口在第 58 行。

## 面试追问

1. `Episode.ordered` 为什么要求结算时间早于或等于反思创建时间？
2. 同一决策时点的两个任务如何在 `MemoryReplay.advance` 中保持因果隔离？
3. `MemoryStore.visible` 为什么在截止时点之后再确定可见版本是不安全的？
4. 为什么 `MemoryStore.put` 必须先写 SQLite？
5. `MemoryStore.rebuild` 如何检查 orphan、missing 和 payload mismatch？
6. 版本哈希与逻辑 ID 为什么要同时保存？
7. M1 和 M2 分别会在哪类历史查询上表现差？
8. M3 中时间指数分数与语义分数为什么需要归一化到同一区间？
9. M3 压缩时为什么要保留历史引用 ID？
10. `trader_with_memory` 怎样防止历史结果被当成当前事实？
11. `model_reflection` 的字段校验和 TokenBudget 分别处理什么失败？
12. 为什么 Development 里的同动作标签需要一套独立合成相关性验证？

## 自行完成的练习

1. 用同一个 episode 写入时点不同的两个版本，增加一道处于版本切换边界的断言。
2. 增加一条与 query 高度相关但归属其它 arm 的合成经验，检查四种 M 模式的候选和注入记录。
3. 给三种语义、时间、证据权重配置写离线消融脚本。冻结相关性标签，报告 Recall、Evidence Preservation 和延迟。
