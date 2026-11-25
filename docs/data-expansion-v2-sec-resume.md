# Data Expansion V2 · SEC 补全

补全先校验已有 SEC 缓存，再写入独立目录。每个工作区使用一个采集锁。优先复用旧缓存，联网时串行访问官方 submissions、Company Facts 和 accession 文件，收到 403 或 429 后停止新的 SEC 请求。采集每进入一个公司或一份主文档就输出 JSON 进度，包含请求命中、实际下载、失败、剩余数量、阶段百分比和估计剩余时间。展品从原文或同一 accession 的官方索引发现。已发布的 SEC 快照和语料目录拒绝原地重建。[SEC 公平访问规则](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)

## 本轮缓存回放

本轮工具新启动的 WSL Ubuntu 进程未继承 `SEC_USER_AGENT`。Bash 登录与交互进程、WSL 用户服务环境和 Windows 进程环境均未找到该变量。已有 29 个有效缓存响应，其中 4 份为 Company Facts。没有启动联网采集。

- 11 份历史 SEC 文档来自 Apple 和 Microsoft，发行人覆盖 **2/20**。可用 Company Facts 覆盖 **4/20**，分别为 Apple、AMD、Microsoft 和 NVIDIA。Exhibit 99.1 正文仍缺失。
- 预览索引重新构建了 **65 份材料、362 个 chunk**。Offline Extended 的 RAG 候选覆盖为 **6/200**，Development V2 为 **6/80**。本轮没有达到 180/200 和 72/80 的门槛。可用季度 Revenue 字段分别覆盖 **33/200** 和 **12/80**。
- Offline Extended **200/200**、Development V2 **80/80** 均完成确定性基线，结果按分区单独记录。两组时点泄漏均为 0。原 400 题、历史归档及缓存保持原位。Holdout 结算 0 题，真实 GLM 请求与 Token 消耗均为 0。
- Frontend V2 发布后，使用最新提交、本轮 SEC 修改和版本一致的预览数据在隔离副本中完成 **109 项回归**。原工作区的历史数据目录仍保留，预览回放使用新的目录。[测试记录](../results/data-expansion-v2-sec-resume-001/tests.txt)

各门槛实测值、按题覆盖和 Development 结果见 [评测摘要](../results/data-expansion-v2-sec-resume-001/summary.json)；原 Offline Extended 的逐题结算完全一致，沿用[原结算文件](../results/data-expansion-v2-001/offline-cases.jsonl)。

## 用独立版本继续采集

在能够读取本机已配置 `SEC_USER_AGENT` 的 WSL Shell 执行。该变量只进入 SEC 请求头，采集日志和发布物没有保存联系人。在重新采集之前，可用 `python3 -c 'import os; print(bool(os.getenv("SEC_USER_AGENT")))'` 核对该 Shell 的继承状态。下列命令需要已下载的 V2 市场快照和已缓存的本地嵌入模型。

```bash
root=data/data-expansion-v2-002
mkdir -p "$root"
ln -s ../data-expansion-v2/market "$root/market"
uv run --locked --group data-v2 python -m tradingagents_lite.data_v2_sec \
  --root "$root" --reuse-cache data/data-expansion-v2/sec/cache --warmup --progress
uv run --locked --group data-v2 python -m tradingagents_lite.data_v2_corpus compile \
  --root "$root" --version data-expansion-v2-corpus-002
uv run --locked --group data-v2 python -m tradingagents_lite.data_v2_corpus index \
  --root "$root" --version rag-expansion-v2-002
uv run --locked --group data-v2 python -m tradingagents_lite.data_v2_evaluation \
  --root "$root" --output "$root/evaluation" --score-development
```

晚于 2020 年 8 月起的定期报告作为 2021 年初决策的财报预热，单独统计。若一个采集目录已经写出最终清单，需要用新目录和一个或多个 `--reuse-cache` 选项继续。评测只接收冻结的 Offline Extended 和 Development V2。验收字段在 `summary.json` 的 `acceptance` 中。
