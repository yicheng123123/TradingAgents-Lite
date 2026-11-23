# TradingAgents-Lite historical smoke report

Backend: offline. Run is Lite A0.

These are OHLC snapshot research results. Historical provider revisions, corporate-action adjustments, hindsight universe and model pretraining are outside the cutoff guarantee. No transaction costs, dividend cash flows, borrow or executable portfolio are modelled. Intervals are descriptive. Four tasks do not establish a general effect. The day bootstrap keeps same-day instruments together but does not model serial dependence. Null rates indicate absent denominators or unavailable modules.

| Metric | Value |
|---|---|
| tasks | 4 |
| scored | 4 |
| failed | 0 |
| directional_n | 3 |
| directional_coverage | 0.75 |
| directional_hits | 2 |
| directional_hit_rate | 0.6666666666666666 |
| hit_rate_wilson_95 | [0.2076596008020477, 0.9385080552796037] |
| mean_alpha | -0.002681250237520444 |
| active_mean_alpha | -0.003575000316693925 |
| hold_reference_mean_alpha | 0 |
| delta_vs_hold | -0.002681250237520444 |
| alpha_date_bootstrap_95 | [-0.010898678826243152, 0.003484449853206506] |
| date_clusters | 4 |
| tool_calls | 4 |
| tool_successes | 4 |
| tool_success_rate | 1.0 |
| tool_failure_tasks | 0 |
| recovered_tasks | 0 |
| tool_recovery_rate | null |
| latency_mean_seconds | 0.014154451750073349 |
| token_usage | null |
| rag_recall_at_5 | null |
| rag_mrr | null |
| memory_retrieval_quality | null |
| context_evidence_preservation | null |
