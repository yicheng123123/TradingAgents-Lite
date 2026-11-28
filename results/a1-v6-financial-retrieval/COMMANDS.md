# A1-V6 离线复现

在隔离 worktree 使用 A1-V3 的本地 CUDA 环境、已冻结的 A1-V3/V4 索引和模型文件。既有 SQLite 用只读连接查询，结构化证据、试验缓存和原始预测写入新的本地目录。原 83 份入库 PDF 由原始哈希校验。本轮使用 PyTorch FP16 与 RTX 4060 Laptop GPU。

```bash
cd /home/gaoyi/projects/tradingagents-a1-v6
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
v6_python=../tradingagents-a1-retrieval-v3/data/a1-retrieval-v3/benchmark-env/bin/python
v6_work=data/a1-v6/reproduction
v6_out=data/a1-v6/reproduction-results
mkdir -p "$v6_out"
cp results/a1-v6-financial-retrieval/protocol-freeze.json "$v6_out/"
for stage in calibrate build structure freeze; do
  "$v6_python" -m tradingagents_lite.retrieval_v6.experiment "$stage" \
    --work "$v6_work" --output "$v6_out"
done
```

复现 Calibration 不读取公开 Test 的标签。正式实验在 `freeze.json` 记录模型、输入、索引、代码和配置后，以相同目录参数调用 `test` 一次。该命令先保存预测，之后才加载评分标签。已生成正式评分归档时，直接使用归档绘图。

```bash
"$v6_python" results/a1-v6-financial-retrieval/render_artifacts.py
"$v6_python" -m pytest -q tests/test_retrieval_v6.py
```

如果本地进程在预测落盘后、评分开始前中断，可使用 `score-existing` 读取同一冻结预测。该命令不进行检索或模型调用。公开 Test 的启动标记阻止再次启动预测。

若旧 S0/S3 归档缺少显式 loss_attribution 字段，在预测完整落盘后使用同目录的 recover_archive_export.py，传入与原试验一致的 --work 和 --output。它从原归档的 Gold 页名次恢复三类计数，再评分冻结预测。
