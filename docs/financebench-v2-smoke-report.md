# FinanceBench V2 Context Engineering Smoke Report

**Decision:** GO for the bounded V2 engineering Smoke. This does not authorize or start the frozen 99-question evaluation. The two calibration questions have already informed diagnosis, so none of these comparisons is an independent estimate of model or retrieval quality.

## Scope and integrity

- One real Smoke only: the two frozen calibration cases across FB0–FB3, 8 planned instances.
- All 8 instances completed with `glm-5.3-flash`. The archive records 12 HTTP attempts, 0 failures, 0 retries, 0 denied requests, and 0 missing-usage responses.
- The largest recorded serialized request was 4,845 UTF-8 bytes, under the unchanged 8,000-byte limit. The existing 120,000-token Smoke cap and SQLite ledger remained in force.
- V1 Smoke, its score, the V1 input/index, and the frozen 99-question plan were not modified. The plan SHA-256 remains `592d48ac1e53d4154fc0e562826f86efb9117e54d70243ef1c85c60fd2efa3d2`.
- Predictions were finalized and hash-checked before the separate scorer opened Gold. Inference receives no Gold path or Gold field; FB3 receives only the current case's Oracle projection. Gold is absent from the prediction records. The tests cover Oracle schema isolation, inference API boundaries, and task/arm context isolation. **Leakage audit: PASS.**
- No formal 99-question tasks or V3 Holdout predictions were run.

## V1 and V2 comparison

“Complete required row” means that one delivered excerpt contains the requested metric label, all required numeric values, the relevant year, and the scale unit. There are three such rows per evidence-enabled arm across the two calibration cases (9 total); FB0 intentionally has no evidence.

| Metric | V1 | V2 |
| --- | --- | --- |
| Complete required rows delivered (FB1/FB2/FB3) | 0/9 | 5/9: FB1 1/3, FB2 1/3, FB3 3/3 |
| Oracle required rows delivered | 0/3 | 3/3 |
| Page-proxy evidence recall, FB0 / FB1 / FB2 / FB3 | 0.00 / 0.50 / 0.25 / 1.00 | 0.00 / 0.50 / 0.75 / 1.00 |
| FB2 second round: new supporting page reached the model | 0/2 cases | 2/2 cases; 2 new evidence IDs delivered per case |
| FB2 second round: complete required row delivered | 0/2 cases | 1/2 cases |
| Task completion | 8/8 | 8/8 |
| Answer scoring | 0/8 correct after V1's independent audit | 1 correct, 4 incorrect, 3 pending human review |
| Real model tokens | 11,194 | 11,053 |
| HTTP requests | 12 | 12 |
| Leakage audit | PASS | PASS |

Page recall and complete-row delivery are different measures. The V2 Oracle arm delivered the required table rows with their label, values, year, and scale; the page-level score alone would not establish that. Citation IDs were valid for all 5 citations, but citation validity does not prove that a cited excerpt supports the answer.

### V2 scoring and usage by arm

| Arm | Completed | Correct / incorrect / pending | Page-proxy recall | Complete rows delivered | Tokens (total; mean per task) | HTTP calls | Mean / P95 latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FB0 | 2/2 | 0 / 1 / 1 | 0.00 | N/A | 629; 314.5 | 2 | 2.99 / 4.06 |
| FB1 | 2/2 | 0 / 1 / 1 | 0.50 | 1/3 | 2,340; 1,170.0 | 2 | 3.62 / 4.07 |
| FB2 | 2/2 | 0 / 2 / 0 | 0.75 | 1/3 | 6,205; 3,102.5 | 6 | 18.90 / 23.43 |
| FB3 | 2/2 | 1 / 0 / 1 | 1.00 | 3/3 | 1,879; 939.5 | 2 | 1.98 / 2.32 |

Pending answers remain unresolved; no human verdicts were added. Full-denominator accuracy bounds are FB0 0–50%, FB1 0–50%, FB2 0%, and FB3 50–100%. These calibration results are not evidence of a general accuracy gain.

## Retrieval and delivery findings

- V1's fixed 720-character prefix was the confirmed Oracle delivery failure. V2's shared selector delivered all 3 required Oracle rows completely (3/3), including the far-offset rows that V1 excluded.
- FB1 used its two deterministic retrieval queries and kept both delivery events linked to the same final request hash and byte count. Its results still missed both required rows for one calibration case; the query did not return the necessary pages in its top four.
- FB2 made two model-selected searches for each case. Each second round delivered two new evidence IDs, and both cases reached a supporting page. That does not mean the full row reached the model: one of the two second-round cases delivered the complete required row.
- A local replay against the frozen index, with no model call, distinguished retrieval from delivery. For one FB2 case, the first-round target page ranked BM25 1 / Dense 9 / RRF 3 but was explicitly skipped to preserve the follow-up reserve. In the other inspected target-page candidates, the full row existed in the retrieved raw chunk but the selected excerpt did not preserve the required label/value/year/unit together. These are recorded as evidence-delivery limitations, not as model answer errors.
- The first-round reserve remained 1,120 of 3,200 serialized evidence bytes; the second round consumed remaining space. Budget-blocked candidates carry explicit `followup_reserve` reasons. No context or request ceiling was raised.

The remaining FB1/FB2 row gaps prevent using this Smoke to claim consistent evidence delivery across retrieval arms. They do not invalidate the Oracle-context repair gate, and the task does not authorize additional calibration calls or prompt tuning.

## Token ledger

| Ledger measure | Before | After | V2 increment |
| --- | ---: | ---: | ---: |
| Project-wide charged tokens | 51,564 | 62,617 | 11,053 |
| A1 charged tokens | 11,194 | 22,247 | 11,053 |
| A1 remaining | 2,488,806 | 2,477,753 | — |
| V2 experiment cap / charged | — | 120,000 / 11,053 | 108,947 remaining |

The provider-reported usage was 9,937 input plus 1,116 output tokens. The 693 reasoning tokens are a detail within output usage and are not added again. Cached tokens were 0. All 12 calls had known usage and were charged at 11,053 total.

## Frozen run identifiers

- Real run: `data/final-financebench-real-smoke-v2-001` (local, ignored data; not committed).
- Score output: `data/final-financebench-real-smoke-v2-score-001` (local, ignored data; not committed).
- Model identity: `glm-5.3-flash` only.
- Run manifest SHA-256: `d4819f8b454533c762de1e0a6aa1cc329a39ffa4b053c5750ee5e58ee5f045e3`.
- Predictions SHA-256: `4a8351c67f324f3189e8a304418cca0aff7544103f75bdb427cf708898763f40`.
- V2 configuration SHA-256: `1a86f701102015ec437aedbe1816c18e8da3ddea9ddecbd5e2067ef7d2478a1a`.
- V2 plan SHA-256: `1559f13dc5334a09c666f60e20b526ff67c3cca303dd7b6fca4a715034bdc52d`.
- Input manifest SHA-256: `2f9afab593a2c8422af5aa6270b7fd005065d07b2238bd599f714b6fab4414a6`.
- Question SHA-256: `6045385a14fc84c7b7e0fb298e0c51957142f9a339833f123bed2691089a8573`.
- Source index SHA-256: `b5d0685d9d0b9110b9105cc88e7614065ba6f9c455bbc20992337eb4b9081498`.
- Prompt SHA-256: `59be1a24cf13973e9aca13e4eb9d70fdf0f31aec263f7ad0eb2a30df03bc6c19`.
- Tool schema SHA-256: `bab8f8f11a0a530e2ca08dca81eec65461caf8ad0907124dce69fdc8d3bdfef0`.
- Answer schema SHA-256: `3dcbfa559b77b51db4c030350a4b0263f10851cbd3c6c310e4350572ac1773bd`.

## Tests and decision

The complete offline suite passed: **210 passed, 1 skipped**. The three former strict xfails now pass as ordinary tests. The V2 Mock dry run also completed all 8 instances; its only purpose was context/budget verification and it made no API requests.

**GO for the bounded V2 engineering Smoke criteria:** the fixed-prefix Oracle defect is corrected, multi-round budget behavior is bounded and observable, the numeric scorer regressions pass, Oracle rows are complete, no leakage was found, and the one real execution completed under the frozen model and budget.

This is not a GO to execute or report the formal 99-question evaluation. FB1/FB2 still show incomplete evidence-row delivery in this two-case diagnostic, and the user explicitly limited this phase to Smoke. No further real calls were made.
