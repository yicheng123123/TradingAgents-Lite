# A3 MCP 与受限金融工具

In-process Function Calling 和 MCP 共用 ToolRegistry.execute。可信宿主绑定冻结任务、行情、证据和可选本地索引。模型能选工具和填写条数、证据类型、检索词。ticker、cutoff 和数据分区来自绑定。注册表中只有金融研究函数。

Function Calling 是模型向宿主提出调用建议。MCP 定义跨进程发现和调用。Python MCP SDK 负责 STDIO、initialize、tools/list 和 tools/call。宿主只向子进程传任务配置和 Python 输出配置，SDK 按白名单继承基础环境。协议使用 STDOUT，诊断走 STDERR。RAG 所需模型必须已在本地缓存。

同一任务延迟启动一个子进程，多个工具共用会话。超时或退出后关闭会话和 worker portal。F1 最多重试一次确定性工具。RAG 保有任务级字节余额，传输状态未知时不能直接重放。真实等价测试比较业务字段、错误和排名，各自记录会话初始化与调用延迟。

## 源码定位

tool_registry.py:49 统一工具授权执行，mcp_finance.py:47 Server，:80 初始化会话，:103 工具调用和超时。

## 面试问题

1. JSON Schema 和 Pydantic 校验分别在哪里完成？
2. 注册表为什么需要额外检查未知工具？
3. 为什么 ticker 参数会改变隔离边界？
4. 为什么 STDIO 日志必须走 STDERR？
5. 什么时候可以共用 ClientSession？
6. ClientSession 和 STDIO transport 分别清理哪些资源？
7. 为什么初始化和单次调用分别记时？
8. 如何区分业务错误、会话超时和进程退出？
9. 如何确保子进程不继承 API key？
10. RAG 超时后有什么状态需要保留？

## 自行完成的练习

1. 为本地和 STDIO 写一组相同的含额外字段请求，检查拒绝结果。
2. 在启动失败案例中加入第二次失败，核对有限重试和关闭行为。
