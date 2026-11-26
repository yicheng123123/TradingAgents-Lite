# B1 Smoke Resume Supplement

This supplement adds archive reuse to the frozen protocol in `b1-protocol.md`. It does not change the task set, prompts, model settings, tools, risk gate, or settlement rules.

## Frozen sequence

1. Run the paired Smoke once on `dev-01` and `dev-02`, first B1-M and then B1-E. The run has a 120,000-token hard cap; the fixed request envelope is 98,432 tokens.
2. Stop if either group has a failed, incomplete, unstructured, unsettled, or identity-mismatched task. Failed requests stay in the source trajectory and budget ledger. They are not retried or converted to Hold.
3. A successful Smoke is reusable only when its task IDs, requested and returned model identity, prompt hashes, tool schema hashes, market and evidence hashes, reasoning and generation settings, settlement, trajectory bytes, and SQLite attempt records validate against this supplement.
4. Continue only the other 18 Development tasks in each group. The Smoke source is read-only and remains unchanged.
5. Merge the two archives into a new directory. The merge requires each task exactly once in each group, keeps failure rows, and refuses missing or duplicate trajectories. It recalculates all metrics from the merged trajectories.

## Identity and integrity checks

The run manifest records a canonical compatibility digest covering the protocol version, committed code SHA, 20-task set, frozen market and A0.5 evidence hashes, system prompt hashes, per-group tool schema hashes, settlement definition, and this document's SHA-256. Both source runs must have the same digest.

For each source run, the validator checks the original task input hash, both byte-identical trajectory files, model configuration, per-run metadata, actual model identities, and each request hash. Every logged HTTP attempt must match one and only one row in the existing SQLite budget ledger by task ID, logical call, attempt number, requested model, request hash, request size, status, reservation, usage details, and charged tokens. A missing ledger or mismatched record rejects reuse.

The final archive keeps both source archive paths and hashes. It counts each source ledger attempt once; merge itself does not write to or alter the budget ledger. An incomplete source can still be preserved, but it cannot be represented as a completed 20-task baseline.

## Budget envelope

With 5,000 prompt bytes, 128 protocol tokens, 512 output tokens and a conservative separate 512 reasoning-token allowance per request, the worst-case envelope is 98,432 tokens for the four Smoke task instances and 885,888 tokens for the remaining 36 instances. Their combined ceiling is 984,320 tokens, below the 1,000,000-token B1 phase cap. Smoke's separate hard cap remains 120,000 tokens. Each actual request still reserves against the persistent global and B1 phase ledgers before network I/O.

Reasoning tokens included in provider completion usage are not added a second time. If usage is missing or a request fails, the existing ledger keeps the full reservation. The envelope is deliberately conservative because UTF-8 bytes do not map exactly to provider tokens. Provider usage, cached-token billing, account pricing, and final invoice amounts may differ from local estimates.

This protocol uses only the frozen 20-task Development data. It does not access Holdout or enable RAG, Memory, Debate, MCP, or verifier extensions.
