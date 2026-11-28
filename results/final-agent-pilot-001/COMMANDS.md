# Phase 3 Pilot 复现

在独立 worktree 使用现有带 CUDA 的 Phase A1 Python 环境。行情、SEC 文件、冻结引用和模型文件均来自已有合法本地快照。prepare 创建新的本地预测包和索引；若编码中断，用 index 续写同一独立索引。预测与统一账本保存在本地。

```bash
cd /home/gaoyi/projects/tradingagents-final-pilot
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pilot_python=../tradingagents-a1-retrieval-v3/data/a1-retrieval-v3/benchmark-env/bin/python
"$pilot_python" -m tradingagents_lite.final_pilot prepare
"$pilot_python" -m tradingagents_lite.final_pilot offline
"$pilot_python" -m tradingagents_lite.final_pilot freeze
```

在运行模型请求前提交协议与冻结清单。使用现有官方凭证与共享账本，不打印凭证。Smoke 通过才运行 formal。进程异常且有未匹配的请求时，运行器拒绝续跑。正常终态会跳过。

```bash
set -a
source /home/gaoyi/.config/tradingagents-lite/glm.env
set +a
export TALITE_MODEL=glm-5.3-flash
"$pilot_python" -m tradingagents_lite.final_pilot smoke
"$pilot_python" -m tradingagents_lite.final_pilot formal
"$pilot_python" -m tradingagents_lite.final_pilot score
```

score 仅在 100 个预测完成且密封后读取结算。若真实 Smoke 或正式执行停止，使用 report 导出检查点与用量，保持结算封存。真实预算仍由原共享 SQLite 逐请求记账，凭证和预测原文留在本地。


## Phase 3.1 已冻结 Pilot 续跑与结算

额度迁移只登记独立审计事件；freeze.json 与 Pilot 源码没有修改。续跑当前 20 题 Pilot 时，在进程内显式应用迁移额度。已完成的单元会按检查点跳过；勿重新运行 prepare、freeze 或 smoke。环境变量从本机配置文件加载，不打印凭证。

```bash
cd /home/gaoyi/projects/tradingagents-final-pilot
set -a
source /home/gaoyi/.config/tradingagents-lite/glm.env
set +a
export PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
../tradingagents-a1-retrieval-v3/data/a1-retrieval-v3/benchmark-env/bin/python - <<'PY'
from tradingagents_lite.final_pilot import run
from tradingagents_lite import token_budget as tb
tb.PHASE_TOKEN_LIMITS = {**tb.PHASE_TOKEN_LIMITS, "B2": 2_500_000, "Reserve": 1_500_000}
run.LIMIT = 900_000
run.execute("formal")
PY
```

预测封存后，离线结算使用相同额度覆盖，以免汇总读取旧限额：

```bash
../tradingagents-a1-retrieval-v3/data/a1-retrieval-v3/benchmark-env/bin/python - <<'PY'
from tradingagents_lite.final_pilot import run
from tradingagents_lite import token_budget as tb
from tradingagents_lite.final_pilot.metrics import export
tb.PHASE_TOKEN_LIMITS = {**tb.PHASE_TOKEN_LIMITS, "B2": 2_500_000, "Reserve": 1_500_000}
run.LIMIT = 900_000
export(allow_settlement=True)
PY
```

续跑后的归档是本分支当前版本；本次已停止在 20 题 Pilot，不会自动启动剩余 60 题或解封 Holdout。
