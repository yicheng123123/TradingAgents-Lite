# TradingAgents-Lite experiment run

Backend: offline. Variant: B0.

These are OHLC snapshot research results. Historical provider revisions, corporate-action adjustments, hindsight universe and model pretraining are outside the cutoff guarantee. No transaction costs, dividend cash flows, borrow or executable portfolio are modelled. Intervals are descriptive; the date bootstrap keeps same-day instruments together and does not model serial dependence. Failed tasks are excluded from scored returns and remain visible in task trajectories.

| Metric | Value |
|---|---|
| tasks | 20 |
| completed_tasks | 20 |
| task_completion_rate | 1.0 |
| failed_tasks | 0 |
| failure_status_counts | {} |
| structured_output_count | 40 |
| structured_output_valid_count | 40 |
| structured_output_valid_rate | 1.0 |
| structured_output_valid_tasks | 20 |
| directional_n | 14 |
| non_hold_coverage | 0.7 |
| directional_coverage | 0.7 |
| directional_hits | 6 |
| directional_hit_rate | 0.42857142857142855 |
| hit_rate_wilson_95 | [0.21380798904474121, 0.6740935542034926] |
| mean_alpha | 0.0034757281640580505 |
| active_mean_alpha | 0.004965325948654358 |
| hold_reference_mean_alpha | 0.0 |
| delta_vs_hold | 0.0034757281640580505 |
| alpha_date_bootstrap_95 | [-0.0013285677542173302, 0.007854122946365876] |
| date_clusters | 10 |
| tool_calls | 20 |
| tool_successes | 20 |
| tool_success_rate | 1.0 |
| tool_failure_events | 0 |
| tool_recovery_events | 0 |
| tool_recovered_tasks | 0 |
| tool_recovery_rate | null |
| citation_references | 40 |
| valid_citation_references | 40 |
| invalid_citation_references | 0 |
| evidence_citation_valid_rate | 1.0 |
| invalid_citation_rate | 0.0 |
| financial_citation_references | 0 |
| eligible_financial_citation_task_rate | 0.0 |
| llm_calls | 0 |
| llm_logical_calls | 0 |
| llm_retries | 0 |
| llm_failed_calls | 0 |
| input_tokens | 0 |
| output_tokens | 0 |
| missing_usage_calls | 0 |
| api_cost_cny_estimate | 0.0 |
| actual_models_observed | [] |
| latency_mean_seconds | 0.006689424599926497 |
| llm_call_latency_mean_seconds | null |
| token_usage | null |
| rag_recall_at_5 | null |
| rag_mrr | null |
| memory_retrieval_quality | null |
| context_evidence_preservation | null |
| by_ticker | {"AAPL": {"tasks": 10, "completed_tasks": 10, "task_completion_rate": 1.0, "failed_tasks": 0, "failure_status_counts": {}, "structured_output_count": 20, "structured_output_valid_count": 20, "structured_output_valid_rate": 1.0, "structured_output_valid_tasks": 10, "directional_n": 7, "non_hold_coverage": 0.7, "directional_coverage": 0.7, "directional_hits": 3, "directional_hit_rate": 0.42857142857142855, "hit_rate_wilson_95": [0.15821985525146975, 0.7495416354723428], "mean_alpha": 0.0030709588397909494, "active_mean_alpha": 0.004387084056844214, "hold_reference_mean_alpha": 0.0, "delta_vs_hold": 0.0030709588397909494, "alpha_date_bootstrap_95": [-0.0061557835301489505, 0.011277570303057238], "date_clusters": 10, "tool_calls": 10, "tool_successes": 10, "tool_success_rate": 1.0, "tool_failure_events": 0, "tool_recovery_events": 0, "tool_recovered_tasks": 0, "tool_recovery_rate": null, "citation_references": 20, "valid_citation_references": 20, "invalid_citation_references": 0, "evidence_citation_valid_rate": 1.0, "invalid_citation_rate": 0.0, "financial_citation_references": 0, "eligible_financial_citation_task_rate": 0.0, "llm_calls": 0, "llm_logical_calls": 0, "llm_retries": 0, "llm_failed_calls": 0, "input_tokens": 0, "output_tokens": 0, "missing_usage_calls": 0, "api_cost_cny_estimate": 0.0, "actual_models_observed": [], "latency_mean_seconds": 0.006837079899923992, "llm_call_latency_mean_seconds": null, "token_usage": null, "rag_recall_at_5": null, "rag_mrr": null, "memory_retrieval_quality": null, "context_evidence_preservation": null}, "MSFT": {"tasks": 10, "completed_tasks": 10, "task_completion_rate": 1.0, "failed_tasks": 0, "failure_status_counts": {}, "structured_output_count": 20, "structured_output_valid_count": 20, "structured_output_valid_rate": 1.0, "structured_output_valid_tasks": 10, "directional_n": 7, "non_hold_coverage": 0.7, "directional_coverage": 0.7, "directional_hits": 3, "directional_hit_rate": 0.42857142857142855, "hit_rate_wilson_95": [0.15821985525146975, 0.7495416354723428], "mean_alpha": 0.003880497488325152, "active_mean_alpha": 0.005543567840464503, "hold_reference_mean_alpha": 0.0, "delta_vs_hold": 0.003880497488325152, "alpha_date_bootstrap_95": [-0.0001656755802049381, 0.00886425476313817], "date_clusters": 10, "tool_calls": 10, "tool_successes": 10, "tool_success_rate": 1.0, "tool_failure_events": 0, "tool_recovery_events": 0, "tool_recovered_tasks": 0, "tool_recovery_rate": null, "citation_references": 20, "valid_citation_references": 20, "invalid_citation_references": 0, "evidence_citation_valid_rate": 1.0, "invalid_citation_rate": 0.0, "financial_citation_references": 0, "eligible_financial_citation_task_rate": 0.0, "llm_calls": 0, "llm_logical_calls": 0, "llm_retries": 0, "llm_failed_calls": 0, "input_tokens": 0, "output_tokens": 0, "missing_usage_calls": 0, "api_cost_cny_estimate": 0.0, "actual_models_observed": [], "latency_mean_seconds": 0.006541769299929001, "llm_call_latency_mean_seconds": null, "token_usage": null, "rag_recall_at_5": null, "rag_mrr": null, "memory_retrieval_quality": null, "context_evidence_preservation": null}} |
