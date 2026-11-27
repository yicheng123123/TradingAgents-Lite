# FinanceBench V2 Context Engineering Protocol

**Status:** frozen for one calibration Smoke; formal test execution remains paused.

V2 addresses the reproducible context delivery defect from the V1 root-cause audit. It is a separate protocol. V1 predictions, scores, and the frozen 99-question plan remain unchanged and cannot be merged with V2 results.

## Frozen inputs and model configuration

| Item | Frozen value |
| --- | --- |
| Model | `glm-5.3-flash` |
| Temperature / reasoning | `0` / `low` |
| Output cap / request retries | `512` tokens / `0` |
| Prompt ceiling | `8,000` UTF-8 bytes, enforced by the existing Provider |
| RAG mode | R3 BM25 + Dense + RRF; existing BGE Small and Qdrant Local index |
| Candidate / result limit | 24 candidates / up to 4 results per search |
| Evidence context ceiling | 3,200 UTF-8 bytes of serialized observation messages |
| Smoke tasks | Existing calibration IDs `financebench_id_04735` and `financebench_id_07507`, all FB0–FB3 arms |
| Formal V1 task-plan SHA-256 | `592d48ac1e53d4154fc0e562826f86efb9117e54d70243ef1c85c60fd2efa3d2` |

The existing V1 data bundle and index are reused without modification. The frozen input-manifest SHA-256 is `2f9afab593a2c8422af5aa6270b7fd005065d07b2238bd599f714b6fab4414a6`; question projection SHA-256 is `6045385a14fc84c7b7e0fb298e0c51957142f9a339833f123bed2691089a8573`; Oracle projection SHA-256 is `0d7dab08382ffc24db22ea0bfecd6244c9f885d7d8800e5165735d7ab0b3ff34`; source index SHA-256 is `b5d0685d9d0b9110b9105cc88e7614065ba6f9c455bbc20992337eb4b9081498`. The source dataset revision remains `cc39aeb4afdf33909ee1412188bf89035950c2eb`.

The system, Analyst, and final-answer instructions are unchanged; their combined prompt SHA-256 remains `59be1a24cf13973e9aca13e4eb9d70fdf0f31aec263f7ad0eb2a30df03bc6c19`. The tool schema SHA-256 remains `bab8f8f11a0a530e2ca08dca81eec65461caf8ad0907124dce69fdc8d3bdfef0`, and answer schema SHA-256 remains `3dcbfa559b77b51db4c030350a4b0263f10851cbd3c6c310e4350572ac1773bd`. The new Smoke manifest records the exact V2 code, test, lockfile, and protocol-document hashes together with its configuration hash and plan hash.

The inference process receives the question projection and, for FB3, only that case's Oracle evidence. `OracleCase` has no Gold field. The Gold file remains in the separate scoring path, and inference receives no Gold path. Formal questions, answer key, retrieval configuration, settlement rules, and V3 Holdout are not changed or opened by this protocol.

## Context selection and budget

V1 took a fixed 720-character prefix from each retrieved chunk. V2 replaces that with deterministic query-aware selection; no LLM compression or additional model call is used.

1. Combine the original question and the current retrieval query into a token set.
2. Rank verbatim token-boundary excerpts from each returned chunk by overlap with the non-generic question/query terms, with a modest preference for numeric values. The bounded local window keeps nearby labels, periods, and values together. When the same chunk identifies a period or currency-scale heading, include that header as a separate excerpt segment with an explicit omission marker for intervening text. If a dense-semantic candidate has no lexical overlap, use its complete chunk only if it fits; never substitute a fixed prefix.
3. Preserve the stable Evidence ID, Document ID, and Page. Do not fabricate, normalize, or rewrite source facts. If no query-relevant segment is available, record `no_query_relevant_excerpt`; if a complete excerpt cannot fit, record the specific context or prompt-byte budget reason and return `budget_insufficient`.
4. Deduplicate both Evidence IDs and normalized content across retrieval rounds. A duplicate does not consume evidence budget again.
5. For a first retrieval round that may be followed by another, reserve 35% of the configured evidence budget (1,120 of 3,200 bytes). The first round may use the remaining 2,080 bytes. The second round may use all bytes left. A one-round Oracle delivery does not reserve space.

Evidence use is accounted from the exact UTF-8 serialization of each observation message, including role, tool-call metadata, Evidence/Document/Page fields, status, and excerpt. Before accepting a candidate, the Runner also serializes the proposed full `messages` plus tool schema exactly as the Provider's existing prompt-byte guard does. That prospective body must remain within 8,000 bytes; the Provider independently checks and logs the actual request byte count and request hash before every HTTP request. Neither limit has been increased.

FB1, FB2, and FB3 use the same `CaseSearch._deliver` selector. FB1 continues to issue its fixed queries. FB2 continues to choose its own queries and has at most two retrieval rounds. FB3 continues to receive only the approved Oracle evidence for its current case.

## Trace and scoring changes

Each retrieval event records the original question, actual query, round, requested and effective limits, BM25/Dense/RRF candidate ranks and scores, selected chunk IDs, delivered Evidence IDs and excerpts, raw/delivered UTF-8 lengths, selection/skip/dedup reasons, remaining context budget, serialized message bytes, prospective/actual request bytes, and request hashes. Raw source chunks are not copied to the public report. The local Smoke output remains subject to the existing private-data and copyright handling rules.

The scorer is versioned as `financebench_numeric_v2`. It recognizes `%`, `percent`, `percents`, `percentage`, and `percentages` as the same percentage dimension. It applies a question's currency-scale hint to an unscaled scalar, but does not apply it a second time when the numeric magnitude is already at least one million base currency units. Existing numeric tolerance is unchanged. Multiple-value free text that cannot be unambiguously resolved remains pending for human review.

## V1 compatibility and execution boundary

V1 artifacts retain their original hashes and interpretation. V2's query-aware excerpts and V2 scoring results are not compatible with, resumable into, or statistically mergeable with V1. V1's 99-question plan remains frozen; no formal task will run in this phase. The one V2 Smoke is limited to eight instances, at most 12 HTTP requests, a 120,000-token experiment cap, the existing SQLite budget ledger, and no automatic retries. A valid response need not be correct for this engineering Smoke to pass; evidence delivery and execution integrity are evaluated separately from answer accuracy.

The V2 configuration hash is computed from this document, the implementation, tests, lockfile, model/tool/schema/prompt settings, and frozen data/index hashes. The run manifest records those hashes before the first model request. If any pinned input or model identity differs, the run must stop.
