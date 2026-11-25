# Project Token Budget

This guide follows the current implementation. It does not claim a real GLM result: the API key was absent during this recovery, so no network request was sent.

## Budget plan

The SQLite ledger applies a 15,000,000-token project ceiling and these non-transferable phase ceilings:

| Phase | Tokens |
|---|---:|
| B1 | 1,000,000 |
| A1 RAG | 2,500,000 |
| A2 Memory | 1,500,000 |
| A3 Agent | 2,000,000 |
| B2 Ablation | 1,000,000 |
| B3 Holdout | 4,000,000 |
| Reserve | 3,000,000 |

The phase ceilings sum to the project ceiling. `TokenBudget` uses a SQLite transaction (`BEGIN IMMEDIATE`) to reserve capacity before each HTTP request, so two local processes cannot both spend the same remaining balance. The default file is `data/token-budget.sqlite`; `/data/` is ignored by Git. Set `TALITE_TOKEN_BUDGET_DB` only when a different persistent ledger location is needed.

## Accumulating usage

`ModelBackend._call` in `tradingagents_lite/provider.py` creates a request hash and measures the UTF-8 size of messages and tool schemas. It asks `TokenBudget.reserve` in `tradingagents_lite/token_budget.py` for a pre-call upper bound containing:

- request bytes plus a 128-token protocol allowance;
- the configured output-token cap;
- an additional equal reasoning-token reserve, in case a provider meters reasoning separately.

On a response with usage, the ledger stores input, output, reasoning-detail and cached-detail counts. `charged_tokens` is input plus output; reasoning and cached counts are details and are not added again when they are already included in those provider totals. If reasoning is explicitly larger than reported output, only the excess is added. Cached input remains part of input usage.

Failed HTTP attempts remain charged at their entire reservation because the provider may have processed the request before a timeout or error. A successful response with missing or incomplete input/output usage is also charged at the full reservation and marked `usage_missing`; any token details that were present remain attached to that attempt. Aggregate summaries count successful calls where reasoning or cache detail fields were absent. The ledger stores the model, phase, experiment ID, task ID, attempt/retry number, request hash, byte size, status, token details and error code. It never stores prompts, responses, API keys or credentials.

Use `TokenBudget.snapshot` to see global, phase and experiment usage. `can_reserve` supports preflight checks; `reserve` repeats the check atomically immediately before each request. If an individual reservation is denied, `ModelBackend` raises `token_budget_exhausted`. The runner writes the current task trajectory and summary, marks the run `budget_exhausted`, then stops before starting another task.

## Why tool loops grow context

In `tradingagents_lite/workflow.py`, the analyst receives its prior messages on every round. A tool round appends the assistant's tool-call message and each tool result, then sends that longer history back to the model. With two rounds, the later analyst requests repeat earlier prompt and tool output. The configured workflow maximum is two rounds and at most two calls per round.

The workflow keeps full evidence in the local task state for validation and trajectory records, while `_llm_tool_result` sends a compact projection back to the model. It removes repeated bar availability metadata and caps a financial excerpt at 800 characters. `ModelBackend.trader` uses `_trader_context` to omit repeated transport metadata from its final prompt. Exact evidence IDs, relevant dates, source type, point-in-time level, market values and the excerpt remain available.

The B1 frozen protocol limits one prompt context to 5,000 bytes and one model output to 512 tokens. Its worst-case request reservation assumes four logical requests per task instance: two analyst tool rounds, the final analyst report, and the Trader proposal. B1 disables application retries, and the OpenAI SDK is configured with automatic retries disabled. Thus the preflight upper bounds are:

- smoke: 2 tasks × 2 groups × 4 requests × 6,152 reserved tokens = **98,432 tokens**, below its 120,000-token limit;
- full 20-task B1: 20 tasks × 2 groups × 4 requests × 6,152 = **984,320 tokens**, below its 1,000,000-token phase ceiling.

The actual prompt may exceed the 5,000-byte cap and be rejected before network I/O. The reservation treats one UTF-8 byte as at most one input token plus protocol allowance; tokenizer and provider accounting can differ. Per-request reservations and returned usage are recorded so observed totals can replace this estimate. No full benchmark starts automatically.

## SDK retry versus business retry

`client_from_env` creates the OpenAI-compatible client with `max_retries=0`. This prevents the SDK from replaying a request invisibly. `ModelBackend._call` owns application retries and reserves every HTTP attempt independently. Generic `talite run --backend llm` uses one application retry by default; frozen B1 uses zero retries to stay inside its preflight envelope. A timeout is still charged at its full reservation, and a later retry must pass the budget check again.

## Context engineering checklist

- Keep deterministic market calculations, evidence filtering, risk checks and settlement in local Python; they must not call GLM.
- Limit ReAct rounds and tool-call counts. Return only task-relevant rows and fields from tools.
- Avoid repeating dates, source manifests, transport metadata and full raw snapshots in each message.
- Keep prompt templates shared across B1-M and B1-E; change only the intended evidence tool and its observed evidence.
- Keep `max_tokens`, reasoning effort and retry policy frozen across experimental groups. [The official GLM-5.3 guide](https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3) says thinking cannot be disabled and supports `low`, `high` and `max`, so B1 fixes it at `low`.
- Review actual usage, missing-usage reservations, retries and cost after every smoke before deciding whether to run the next experiment.

## Next real smoke

After setting `TALITE_API_KEY` in the WSL environment without printing it, run:

```sh
cd /home/gaoyi/projects/tradingagents-lite
uv run --locked talite b1 --scope smoke --output results/b1-smoke-004
```

This uses frozen `dev-01` and `dev-02` in both B1-M and B1-E, checks the 98,432-token static bound and the current SQLite balance before sending requests, and hard-stops at 120,000 reserved tokens. It records actual model IDs and usage in each task trajectory plus the manifest's token-budget snapshot. It does not launch the 20-task runs.

## Code map

Line numbers below refer to the implementation accompanying this document and can shift as code changes.

- `tradingagents_lite/token_budget.py:12-24` defines project and phase caps; `reservation_size:120`, `capacity:137`, `can_reserve:168`, `reserve:176`, `settle_usage:235`, `settle_unknown:262`, and `snapshot:288` implement reservation, settlement and reporting.
- `tradingagents_lite/provider.py`: `_trader_context:38`, `ModelBackend.set_budget_context:119`, `ModelBackend._call:122`, and `client_from_env:294`. GLM fields are passed using SDK `extra_body`; SDK retries are disabled in the client constructor.
- `tradingagents_lite/runner.py:185`: `run` binds experiment/task identity, persists each completed task row, and stops after a denied request.
- `tradingagents_lite/b1.py`: `request_cost_ceiling:40` computes the frozen protocol bound; `_completion_check:72` requires all requested tasks, zero failures and all valid pairs before B1 can be complete.
- `tradingagents_lite/workflow.py`: `_llm_tool_result:120` compacts tool context; analyst loop `194`, tool execution `234`, and Trader preparation `259` show where context grows and is consumed.

## Interview follow-ups

1. Why is a reservation charged before the HTTP call, and why does a timeout keep that reservation?
2. How does `BEGIN IMMEDIATE` prevent two processes from overspending the same SQLite balance?
3. Why are cached tokens and reasoning tokens recorded without blindly adding them to provider input/output totals?
4. What is the difference between disabling SDK retries and setting the application's retry count to zero?
5. How does ReAct history growth affect prompt size, and which fields can a tool omit without weakening citation validation?
6. Which assumptions make the byte-based preflight an estimate rather than an invoice guarantee?
7. Why does a B1 run need successful completion counts and valid pairs in addition to a model identity check?
