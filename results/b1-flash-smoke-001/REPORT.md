# B1 GLM-5.3-Flash Baseline V2

Status: **failed_smoke**. These results use only the frozen `glm-5.3-flash` model. Observed model identities: glm-5.3-flash.

| Group | Requested | Complete | Failed | Unattempted | Completion | Structured valid | Tool calls successful | Non-Hold coverage | Direction hits / N | Mean alpha; date-cluster 95% CI | Evidence refs valid / total | LLM calls | Input / output / reasoning / charged tokens | Avg charged / attempted task | Complete / hits per 100k tokens | Task P50/P95; call P50/P95 latency (s) | List-price estimate (CNY) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 2 | 2 | 0 | 0 | 1.0 | 4/4; rate 1.0 | 2/2; rate 1.0 | 0.5 | 1/1; rate 1.0 | 0.010858690289589468; CI - | 4/4; rate 1.0 | 0 | 0/0/0/0 | - | -/- | 0.008352345000275818/0.008931712800040258/-/- | 0.0 |
| B1-M Flash | 2 | 1 | 0 | 1 | 0.5 | 2/2; rate 1.0 | 2/2; rate 1.0 | 1.0 | 1/1; rate 1.0 | 0.021717380579178935; CI - | 4/4; rate 1.0 | 3 | 3769/542/9/4311 | 4311.0 | 23.19647413593134/23.19647413593134 | 25.46039435900093/25.46039435900093/7.239650800000163/11.90264713779925 | 0.0045328 |
| B1-E Flash | 2 | 0 | 1 | 1 | 0.0 | 0/1; rate 0.0 | 4/4; rate 1.0 | 0.0 | 0/0; rate - | -; CI - | 0/0; rate - | 3 | 4229/526/27/4755 | 4755.0 | 0.0/0.0 | 26.59271532600178/26.59271532600178/10.789573234998898/12.793717877900963 | 0.004856 |

Paired Flash tasks: 0; mean paired alpha delta E−M: None; date-cluster 95% interval: None.
Directional transition denominator: 0; wrong-to-right: 0; right-to-wrong: 0; action changed: 0.

Failure status counts by group: `{"B0": {}, "B1-E Flash": {"workflow_error": 1}, "B1-M Flash": {}}`.
B1-M calls: 3; B1-E calls: 3; retry count: 0.

Smoke source: `results/b1-flash-smoke-001`. Reused task instances are counted once: True.
Quality check: `{"complete": false, "failed_checks": ["structured_output_invalid", "task_not_completed"], "requires_no_unexpected_retry_or_truncation": true, "requires_successful_market_tool_call_per_task": true, "requires_two_valid_json_outputs_per_task": true, "requires_valid_evidence_references": true}`.
Project remaining: 14985877 Tokens; B1 remaining: 985877 Tokens.

Standard price estimates use the published Flash input/output rates, with cached input priced separately when usage detail is available. They are not provider invoices. The API does not expose an experiment-specific cash debit or package debit, so those amounts remain unavailable.

Flash Thinking is enabled with `reasoning_effort=low`; it cannot be disabled for this model. Reasoning is reported as a usage detail and is not double-counted when included in output tokens.

No RAG, Memory, Debate, MCP, or verifier extension was enabled. Holdout tasks and labels were not accessed.

## Failure and archive notes

dev-01 / B1-E stopped at Analyst MarketReport validation (ValidationError) after the API returned the frozen Flash identity and all four function-tool executions succeeded. The archive does not retain raw generated text or field-level Pydantic details, so the exact invalid field cannot be reconstructed offline. No retry or replay was made.

Full per-cell trajectories, task copies, and SQLite state are retained locally because their tool traces include cutoff-bound A0.5 evidence excerpts. The GitHub result commit contains aggregate metrics, manifests, hashes, and budget data only.
