# 离线复现

在 WSL 的独立 worktree 中运行。运行器使用原仓库数据中的 A2 Development 归档、截至 2026-08-07 的 V3 Development 行情和冻结的 CPU BGE-small 权重。运行器将缓存、SQLite 和 Qdrant 写到新路径，模型文件共用只读源。每次复现指定两个全新目录。

```bash
cd /home/gaoyi/projects/tradingagents-a2-v2-memory
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
a2_python=../tradingagents-lite/.venv/bin/python

"$a2_python" -m tradingagents_lite.memory_v2.benchmark \
  --source ../tradingagents-lite/data \
  --work data/a2-memory-v2/002 --output data/a2-memory-v2/002-results

cp results/a2-memory-v2/render_charts.py data/a2-memory-v2/002-results/
"$a2_python" data/a2-memory-v2/002-results/render_charts.py
```

发布的结果可直接重绘，图表代码只需要 Python 标准库。

```bash
python3 results/a2-memory-v2/render_charts.py
"$a2_python" -m pytest -q tests/test_memory_v2.py tests/test_memory.py tests/test_rag.py
```

全量回归的历史 fixture 及 worker 依赖见 [测试报告](TESTS.md)。数据顺序、文件指纹、模型 revision、配置和随机种子见 [预注册协议](protocol-freeze.json) 与 [配置冻结](freeze.json)。运行时和实际规则 Reflection 生成时间随环境变化。
