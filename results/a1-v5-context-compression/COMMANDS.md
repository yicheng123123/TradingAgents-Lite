# A1-V5 Reproduction Commands

Run these commands from the isolated worktree. They use the A1-V3 benchmark Python environment and the locally pinned BGE-M3 files; Hugging Face downloads are disabled.

    cd /home/gaoyi/projects/tradingagents-a1-v5-context-compression
    export PYTHONPATH="$PWD"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    PYTHON=/home/gaoyi/projects/tradingagents-a1-retrieval-v3/data/a1-retrieval-v3/benchmark-env/bin/python

    "$PYTHON" -m tradingagents_lite.financebench_compression_v5 prepare
    "$PYTHON" -m tradingagents_lite.financebench_compression_v5 calibrate --device cuda
    "$PYTHON" -m tradingagents_lite.financebench_compression_v5 freeze
    "$PYTHON" -m tradingagents_lite.financebench_compression_v5 test
    "$PYTHON" results/a1-v5-context-compression/render_artifacts.py

The public Test command is one-shot: it writes a start marker before opening its 99 public questions. Do not rerun it after data/a1-v5-context-compression/001/test-started.json exists.

Run the isolated synthetic invariants with:

    PYTHONPATH="$PWD" "$PYTHON" -m pytest -q tests/test_financebench_compression_v5.py

Local run inputs, labels, and delivered excerpts live under the ignored data/a1-v5-context-compression/001/ directory and are not committed. Tracked result files contain de-identified per-question metrics, aggregate summaries, the freeze record, and SVG charts. No model training, network model download, or GLM call is part of these commands.

The evaluator writes per-question and aggregate result JSON before its optional matplotlib chart export. If matplotlib is unavailable in the reused benchmark environment, run the standard-library renderer above; it regenerates only REPORT.md and SVG figures from those archived JSON files. Never rerun the one-shot public Test.
