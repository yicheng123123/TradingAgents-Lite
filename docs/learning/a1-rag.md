# A1 Agentic RAG

## 数据流和模块

1. `rag-index` 接收冻结 EvidenceStore 与原始缓存，或明确许可依据、URL、原始哈希、可用时间与分区的本地 JSONL 清单。PDF 使用页索引，SEC HTML 用标题、段落和表格行，新闻取得标题时只索引标题。
2. `parse_document` 校验原始字节，生成有结构的文本块。 `chunk_document` 在章节和 PDF 页边界内滑动词窗，默认 220 词、重叠 32 词。固定窗跳过章节边界，保留 PDF 页便于对齐。UUID 基于文件版本、切分配置和首尾位置生成。
3. ONNX Runtime 在本机按批推理。缓存键包含权重指纹、文本、查询/文档角色。Qdrant 按余弦距离保存向量和元数据。Local mode 在过滤后的候选上做精确搜索，可跨进程顺序重启，适合此实验规模。服务端 HNSW 可单独验证 ANN 召回和延迟。
4. 检索先约束股票、分区、可用时间、PIT 等级，再使用 EvidenceView 同款版本仲裁。R1 BM25、R2 Dense、R3 RRF 融合、R4 本地 cross encoder 重排有限候选。每阶段记 ID、排名和 score。
5. 固定检索使用预设查询，Analyst 用 agentic 设置则能提出查询和补充检索。任务本身绑定 ticker 与 cutoff。RAG 回送有字节上限的片段，当前任务只允许引用已送达的 chunk。现有风险和结算节点继续使用确定性规则。

## 算法和取舍

BM25 对每个查询词计算 `IDF × tf × (k1+1)/(tf+k1×(1-b+b×len/avglen))`，使用 k1=1.2、b=0.75。余弦相似度对 L2 归一化向量等于点积。RRF 对每份列表的排名累加 `1/(60+rank)`，避开 BM25 和余弦分值范围不同的问题。Cross encoder 联合编码 query 和候选文本，默认只查看 24 个融合候选。可对照 SQLite FTS、BM25 单路、向量检索单路、服务端 Qdrant HNSW 或线性融合。不同融合公式应只用 calibration 分组比较。

## 本地复现

~~~sh
uv run --locked talite rag-index --output data/a1-development
git clone --filter=blob:none --sparse https://github.com/patronus-ai/financebench.git data/financebench
git -C data/financebench sparse-checkout set data pdfs
git -C data/financebench rev-parse HEAD
uv run --locked talite rag-benchmark --source data/financebench --output data/financebench-run
uv run --locked talite run --backend offline --tasks tasks/development.jsonl --data data/yahoo-2024-v1 --phase A1 --rag-index data/a1-development --rag-dispatch fixed --output data/a1-fixed-dev
~~~

公开题目与 PDF 使用结果清单中的 Git revision 和 SHA-256 核对。另存的 PDF 可用 `--pdf-root` 指定。默认用 pypdf。需要更快的文本提取时，可通过 `uv run --locked --extra pdfium talite rag-benchmark ... --pdf-engine pdfium` 选择 PDFium。`--modes R1` 使用仅保存元数据和文本的 Qdrant 集合，无需加载 embedding 权重。不同解析器会影响段落文本和切分，比较 R1–R4 时应统一解析器。[本地评测与逐题结果](../../results/a1-rag-local-001/REPORT.md)

`rag-index --manifest` 的每行含 Document 模型字段及清单目录内的相对 `path`。将核准的 PDF 或有全文使用权的新闻正文交给解析器。默认本地 CPU，`--device cuda` 检查 ONNX CUDA provider，缺失时走 CPU。固定 revision 的量化 BGE Small 与 MiniLM 权重缓存在本地，写入 tokenizer 和权重的 SHA-256。保留原始正文与权重于被 Git 忽略的 data 目录。

测试和本地规则可验证检索、引用与 Agent 控制流。未来使用 `--backend llm` 会归入现有 A1 TokenBudget。A1 可用独立的 16,000-byte 请求上限和 3,200-byte RAG 上下文上限。B1 Smoke 用实际 SQLite 记账数消耗阶段余额。运行 Full Development 时预检其静态上界与剩余余额，不能用 Smoke 的预算上界代替已经结算的用量。

## FinanceBench

固定公开数据版本，以公司哈希分 20% 桶作为 calibration，其他公司为 test。锁定检索参数后读取标注页，校验 evidence 文本与 PDF 零基页索引的对齐。输出记录每题、分母和排除原因。查询在已取得的同公司 PDF 中检索，使用该公开集合的固定评测 cutoff。该集合与历史交易任务分区隔离。

相关性有两层。文档层对 Top-10 chunk 返回的唯一文档计分。chunk 层采用页级代理，首次检中某个标注页的 chunk 算一个二元命中。Recall@K 除以标注相关对象数。MRR@10 为首次命中排名的倒数。nDCG@10 用 `1/log2(rank+1)` 折扣并除以理想排序 DCG。保留逐题阶段候选、延迟和索引大小，发布结果仅含 ID、指标、来源和哈希。Patronus 数据卡对公开题目标注 CC BY-NC 4.0。财报仅在本地研究流程使用，保留各自的来源 URL。[FinanceBench 数据卡](https://huggingface.co/datasets/PatronusAI/financebench)

## 常见故障

PDF 提取会丢失扫描图像或重排表格；不能对齐的标注应排除并报告。SEC 缓存的哈希变化要立即阻止索引。同一时点的冲突代理版本会被扣留。没有合规证据时应返回空结果。查询与重排模型需要一致的预处理和显式模型版本。Qdrant Local 数据目录同一时刻只由一个进程持有。工具超时会进入结构化错误路径。历史上下文过长时，Provider 的整体请求字节上限仍生效。

## 源码导读

- [documents.py](../../tradingagents_lite/documents.py#L68) 的 `parse_document` 在第 68 行，`chunk_document` 在第 129 行。
- [vectors.py](../../tradingagents_lite/vectors.py#L91) 的批量缓存入口在第 91 行，Qdrant 写入和查询在第 171、188 行。
- [rag.py](../../tradingagents_lite/rag.py#L78) 的候选门在第 78 行，`retrieve` 在第 107 行，`RagView.search` 在第 144 行。
- [workflow.py](../../tradingagents_lite/workflow.py#L167) 的 `build_workflow` 在第 167 行，工具执行和引用检查位于同一图内。
- [rag_benchmark.py](../../tradingagents_lite/rag_benchmark.py#L60) 的 `benchmark` 在第 60 行，`relevance` 在第 28 行。

## 面试追问

1. 为什么 `frozen_sources` 从缓存读取 SEC 并另检查正文哈希？
2. `parse_document` 如何处理 HTML 表格和 PDF 页码？
3. 改 overlap 会如何影响 `chunk_document` 的稳定 ID？
4. `VectorStore` 如何阻止混用两个 embedding 版本？
5. 为什么在 `Retriever._candidates` 同时约束 PIT 和分区？
6. `retain_versions` 如何处理相同可用时间的两个代理版本？
7. 为什么 RRF 使用排名而不用检索器给的原分？
8. `RagView.search` 预算耗尽时，哪些候选仍可被引用？
9. 工具失败怎样进入 `workflow.tools` 并允许补充查询？
10. 为什么 FinanceBench 以公司为单位划分？
11. 页级标签会怎样高估 chunk 级相关性？
12. Qdrant local 的延迟与远程 ANN 的性能有何区别？

## 自行完成的练习

1. 新增一个有重复表头的合成 HTML，比较两种切分的定位与重叠覆盖。
2. 在 calibration 上比较两组 RRF 常数，冻结后只运行一次 test。记录参数、检索分母与排序变化。
3. 编写一个固定调用三次检索的 mock Analyst，验证每次预算和最终可引用 ID。
