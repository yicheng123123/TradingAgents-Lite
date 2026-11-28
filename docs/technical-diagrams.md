# 架构与技术图解

本文按当前主线源码描述逻辑数据流。A1-V3 Cross-Encoder 是独立离线实验分支，不代表主线默认检索器。原有 B1 工作流、时点与结算说明保留在[架构历史说明](architecture.md)。

## Agentic RAG

Retriever 是确定性的混合检索组件；Agentic Retrieval Loop 是上层 Analyst 根据观察决定是否继续检索的有界控制流。固定检索组不拥有动态 Query 决策权。

```mermaid
flowchart LR
    Q["研究问题 + 当前观察"] --> LLM["Analyst LLM<br/>决定检索 Query 与是否继续"]
    LLM -->|Function Call: rag_search| R["Retriever"]
    subgraph RETRIEVER["确定性 Retriever"]
        direction TB
        BM25["BM25"]
        DENSE["BGE Dense / Qdrant"]
        RRF["Reciprocal Rank Fusion"]
        SELECT["Context Selection<br/>相关摘录 · 去重 · 预算"]
        BM25 --> RRF
        DENSE --> RRF
        RRF --> SELECT
    end
    R --> BM25
    SELECT --> OBS["受限 Observation<br/>Evidence ID · 来源 · 页码 · 时点"]
    OBS --> LLM
    LLM -->|继续：下一条 Query| R
    LLM -->|停止：最终 MarketReport| OUT["MarketReport<br/>只引用已送达的 Evidence IDs"]

    classDef query fill:#EAF1F8,stroke:#8CA4BA,color:#163047;
    classDef agent fill:#E7F0FA,stroke:#7799B8,color:#15334B;
    classDef retrieve fill:#E5F2F0,stroke:#79A99D,color:#173E38;
    classDef output fill:#EEF1F4,stroke:#8D99A6,color:#273746;
    class Q query;
    class LLM agent;
    class R,BM25,DENSE,RRF,SELECT retrieve;
    class OBS,OUT output;
```

Retriever 根据 ticker、分区、截止时点和 PIT 等范围先过滤候选，再计算 BM25 / Dense 排名及 RRF。LLM 只在 Agentic 配置中决定查询内容和是否继续；检索结果作为 Tool Observation 返回。FinanceBench V2 的 Context Selection 另外按 query-aware 规则选择完整片段，并对最终序列化消息施加预算。两层不要混为一个检索算法。

实现位置：[workflow.py](../tradingagents_lite/workflow.py)、[rag.py](../tradingagents_lite/rag.py)、[financebench.py](../tradingagents_lite/financebench.py)、[vectors.py](../tradingagents_lite/vectors.py)。

## Episodic Memory

Memory 仅在 episode 已结算且 eligible_at 不晚于任务 cutoff 后可见。SQLite 是权威记录；Qdrant 是可恢复的向量索引。默认最多取 Top-3，随后对记录做紧凑投影并受上下文字节上限控制。

```mermaid
flowchart LR
    DECISION["历史 Decision<br/>输入证据与执行轨迹"] --> SETTLE["延迟结算<br/>outcome_ts"]
    SETTLE --> REFLECT["Reflection<br/>规则逻辑或受预算约束的模型"]
    REFLECT --> STORE["SQLite + Qdrant<br/>版本化、幂等写入"]
    STORE --> FILTER["时点与范围过滤<br/>ticker · cutoff · partition · arm"]
    FILTER --> RANK["语义相似度 + 时间衰减<br/>+ Evidence Quality"]
    RANK --> TOP["Top-3"]
    TOP --> COMPRESS["Context Compression<br/>保留来源和历史 Evidence IDs"]
    COMPRESS --> TRADER["后续任务的 Trader<br/>历史经验只作参考"]

    classDef base fill:#EAF1F8,stroke:#8CA4BA,color:#163047;
    classDef memory fill:#E5F2F0,stroke:#79A99D,color:#173E38;
    classDef filter fill:#F0ECF8,stroke:#A597C5,color:#30294B;
    classDef trader fill:#EEF1F4,stroke:#8D99A6,color:#273746;
    class DECISION,SETTLE,REFLECT base;
    class STORE,RANK,TOP,COMPRESS memory;
    class FILTER filter;
    class TRADER trader;
```

M3 的实际排序公式：

~~~text
Score = 0.65 × Semantic + 0.25 × exp(-age_days / 30) + 0.10 × Evidence Quality
~~~

这是人工配置的加权规则，不是训练得到的参数。余弦相似度先映射到 [0, 1]；时间项以 30 天衰减；证据质量缺失时，以已观察 Evidence ID 数量做有界代理。M3 的上下文压缩不代表 Recall 提升：本地回放中 M3 Recall@3 为 0.565，低于 M2 的 0.631。

实现位置：[memory.py](../tradingagents_lite/memory.py)、[runner.py](../tradingagents_lite/runner.py)。

## Agent Harness

同一 Harness 管理状态、工具消息配对、请求预算和可追踪执行。Tool Calling 路径与最终结构化生成路径不同：参数先按工具 Schema 校验并执行；最终 MarketReport / Proposal 再进行 Pydantic 校验。启用时 Verifier 另外校验引用和范围；它不证明自由文本事实的语义蕴含关系。

```mermaid
flowchart TD
    LLM["LLM Response"] --> CHOICE{"finish_reason / tool_calls"}
    CHOICE -->|工具调用| CONTRACT["已注册工具 + 参数 Schema 校验"]
    CONTRACT -->|非法 / 超预算| STOP["记录错误并终止当前路径"]
    CONTRACT -->|合法| EXEC["Tool Registry 执行<br/>Local 或可选 MCP STDIO"]
    EXEC --> OBS["Tool Observation<br/>同 call_id 返回 LangGraph State"]
    OBS --> LLM
    CHOICE -->|最终文本 / JSON Mode| SCHEMA["Pydantic Schema Validation<br/>MarketReport / Proposal"]
    SCHEMA -->|失败| FAILURE["失败轨迹保留<br/>最多一次受配置限制的恢复"]
    SCHEMA -->|通过| CITATION["确定性引用检查<br/>已送达 ID · ticker · partition · eligible_at"]
    CITATION -->|违规| SAFE["安全降级 / Hold / 终止"]
    CITATION -->|通过| RISK["规则风险门<br/>如波动率阈值"]
    RISK --> DONE["Decision + Trace"]
    LLM -.-> BUDGET["TokenBudget：调用前原子预留<br/>失败 / 缺失 usage 保守记账"]
    EXEC -.-> TRACE["Trace：工具、错误、时延、证据 ID"]
    SCHEMA -.-> TRACE

    classDef model fill:#E7F0FA,stroke:#7799B8,color:#15334B;
    classDef contract fill:#F0ECF8,stroke:#A597C5,color:#30294B;
    classDef tool fill:#E5F2F0,stroke:#79A99D,color:#173E38;
    classDef stop fill:#F7EEEE,stroke:#C58C8C,color:#4A2929;
    classDef observe fill:#EEF1F4,stroke:#8D99A6,color:#273746;
    class LLM model;
    class CONTRACT,SCHEMA,CITATION contract;
    class EXEC,OBS tool;
    class STOP,FAILURE,SAFE stop;
    class CHOICE,RISK,DONE,BUDGET,TRACE observe;
```

**Verifier 当前实际边界：**结构合法、引用 ID 是否在已送达上下文、ticker/数据分区/版本/时点，以及确定性波动率规则。代码将 citation_entailment 明确标记为 not_assessed，因此不把它描述为完整事实核验。

实现位置：[workflow.py](../tradingagents_lite/workflow.py)、[provider.py](../tradingagents_lite/provider.py)、[tool_registry.py](../tradingagents_lite/tool_registry.py)、[verifier.py](../tradingagents_lite/verifier.py)、[token_budget.py](../tradingagents_lite/token_budget.py)、[agents.py](../tradingagents_lite/agents.py)。
