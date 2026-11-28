# Reproduction

Run in the isolated worktree on Linux/WSL. Keep the original A1-V5 experiment, index and model files in their established locations. The diagnosis consumes only the three named Calibration files. It checks the original manifest hashes, the exact replayed deliveries, and the pinned BGE-M3 files. No Qdrant connection, data download, or chat-model request occurs.

```sh
cd /home/gaoyi/projects/tradingagents-a1-v5.1
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset TALITE_API_KEY OPENAI_API_KEY
PYTHON=/home/gaoyi/projects/tradingagents-a1-retrieval-v3/data/a1-retrieval-v3/benchmark-env/bin/python
"$PYTHON" -m tradingagents_lite.financebench_compression_diagnosis --device cuda
"$PYTHON" -m tradingagents_lite.financebench_compression_repair
"$PYTHON" results/a1-v5.1-compression-diagnosis/render_figures.py
"$PYTHON" -m pytest -q tests/test_financebench_compression_diagnosis.py
```

The first two commands default to the original A1-V5 local Calibration archives and this experiment's results directory. Use `--data` and `--out` on both if moving the input or rendering in another output directory. Each run regenerates the new derived diagnostics only. Timings can differ across machines. Archived C2/C3/C4 timings include encoder and accounting overhead. D0/D1 timings were remeasured interleaved under the same process, excluding proxy tokenizer accounting in both arms.

## Historical regression fixtures

The complete offline suite expects local snapshots under `data/yahoo-2024-v1`, `data/financial-evidence-dev-003`, the generated **Development** fixture at `data/counterfactual-development-v1-001`, and the pinned local embedding revision under `data/local-models/models`. Use approved copies of those four fixtures, not a link to the whole data tree. The isolated mock worker expects the local `.venv/bin/python` entry point. In this worktree, `.venv` points to the previously working A1-V3 environment. With the same authorized fixture set on fixed baseline and new code:

```sh
PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
```

The Counterfactual tests that exercise Private Test logic create fresh synthetic temporary fixtures. They do not open the real Private Test or V3 Holdout.
