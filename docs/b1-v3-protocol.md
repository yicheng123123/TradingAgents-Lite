# B1 Flash V3 Smoke protocol

V3 uses the original B1 Development tasks, frozen A0.5 market and evidence snapshots, B0 reference, and five-session settlement. Smoke runs dev-01 and dev-02 under B1-M and B1-E in that order for each task. Both arms share the same model, sampling, prompts, contracts, and repair limit. B1-E adds the cutoff-bound financial-evidence tool. The V2 successful instance stays with the V2 archive.

## Output contract

The API model is `glm-5.3-flash`, temperature 0, output limit 512 tokens, and thinking enabled with low reasoning effort. Requests retain the 8,000-byte message/tool cap, two tool rounds with at most four calls each, and zero SDK and transport retries.

Tool discovery uses ordinary Function Calling. Once tools finish or the model stops requesting them, a separate tool-free request generates MarketReport in JSON mode. Proposal generation and any repair also use a tool-free JSON-mode request. The [Flash model documentation](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash) lists structured output. The [structured-output guide](https://docs.bigmodel.cn/cn/guide/capabilities/struct-output) demonstrates `response_format.type=json_object` with client-side schema validation. V3 enforces the schema and available citation IDs locally. Both prompts include a minimal JSON example, exact allowed fields, evidence ID limits, and text length limits.

Local parsing can remove a single outer Markdown fence. A validation failure can trigger one fresh generation with field-level feedback and the same observations. The two stages share one repair allowance per instance. Repair calls cannot dispatch tools and are subject to all request limits. Validation tracks JSON syntax, required and extra fields, length and type errors, truncation, and citation errors. The public manifest records only known field locations and error types, counts, response bytes, finish reason, schema hashes, and repair charges. Extra keys are redacted. First and final rates use the same denominator of initial outputs that reach validation.

## Checkpoints and budget

The existing SQLite ledger atomically reserves each request, including repairs, under the 50,000-token Smoke cap, 1,000,000-token B1 cap, and 15,000,000-token project cap. A task output failure is recorded and the next instance can run. A system fault such as an API failure, unexpected model, unavailable settlement, or denied reservation stops execution.

Each instance persists an SQLite outcome checkpoint and local trajectory. Explicit resume verifies the source tree, protocol, original task plan, checkpoint hashes and corresponding ledger attempts. It skips all terminal instances, including failures. If a crash leaves attempts without a recoverable outcome, resume stops for reconciliation. Complete checkpoints can be recovered from SQLite if the JSONL export was interrupted. The manifest contains the public diagnostics and hashes. The local cell files contain financial evidence and stay in the local archive.

```bash
source ~/.config/tradingagents-lite/glm.env
export TALITE_MODEL=glm-5.3-flash
uv run --locked talite b1-v3 --output results/b1-flash-v3-smoke-001
```

After an interruption use the same output path with `--resume` once the system fault is resolved.
