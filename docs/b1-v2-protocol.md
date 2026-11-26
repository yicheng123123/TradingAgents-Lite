# B1 Flash V2 Frozen Protocol

This is a separately versioned protocol. The legacy GLM-5.3 protocol in
[b1-protocol.md](b1-protocol.md), its failed Smoke archive, and its 5,057-token
ledger records remain unchanged. GLM-5.3 trajectories are not eligible for a
Flash V2 result or a V2 resume.

## Frozen inputs and separation

V2 uses the original 20 Development task definitions, A0.5 market snapshot,
A0.5 financial-evidence snapshot, B0 archive, and the original settlement code.
The runner verifies the task, market, evidence manifest, evidence JSONL, and B0
hashes before creating an archive. B1-M exposes market tools. B1-E exposes the
same market tools plus the original cutoff-filtered A0.5 financial-evidence
tool. Both conditions use the same model and sampling settings. RAG, Memory,
Debate, MCP, and verifier extensions are disabled. Holdout tasks and labels are
not read.

Every run writes its protocol, prompt hashes, tool schema hashes, input hashes,
runtime dependencies, and code commit into its own manifest before any model
request. Smoke results can be reused only after checking the exact task IDs,
condition, requested and returned model, prompt and tool hashes, frozen inputs,
reasoning and sampling settings, settlement protocol, trajectory hashes, and
matching SQLite ledger attempts. Smoke tokens are referenced once when
Development results are assembled; old GLM-5.3 records are never imported.

## Model and request limits

- Unset `TALITE_MODEL` now defaults the application to `glm-5.3-flash`; the
  archived V1 runner explicitly pins `glm-5.3` and is unaffected. The Research
  page displays an explicit environment override and warns when it is not Flash.
- Requested and required returned model: `glm-5.3-flash`.
- Endpoint: `https://open.bigmodel.cn/api/paas/v4`.
- Temperature: `0`; output limit: `512` tokens; provider seed: unset.
- Thinking: enabled, because Flash Thinking cannot be disabled; reasoning
  effort: `low`.
- Analyst: at most 2 tool rounds and at most 4 calls in each round. Each tool
  response remains paired with its assistant tool-call message.
- Serialized UTF-8 `messages + tools`: at most 8,000 bytes per request. The
  previous 5,326-byte archived request is below this new limit.
- At most 4 model calls per task: two tool rounds, a final Analyst response,
  and one Trader response. SDK and application retries are both disabled.
- Tool results are compacted for the Flash protocol: at most four evidence
  items per result, excerpts capped at 320 characters, market bars capped at
  eight, and evidence IDs, dates, values, units, periods, and PIT fields kept.

The published Flash context window is larger than this experiment needs; the
8,000-byte limit is the experiment's own request guard. The provider's
`completion_tokens` includes reasoning when reported within output; the ledger
adds only reasoning tokens that exceed the reported output total.

## Budget admission and stopping

The project cap remains 15,000,000 tokens, the B1 phase cap remains 1,000,000,
and each new Smoke archive has a 120,000-token cap. The pre-existing 5,057
GLM-5.3 tokens remain charged against the B1 and project caps.

Before every HTTP request, the SQLite ledger atomically reserves the serialized
request size, protocol allowance, output limit, and conservative reasoning
reserve. Failed requests and missing usage remain charged at the full
reservation. Provider-reported reasoning and cached-input values are details;
they are not double-counted when included in output or input totals. A denied
reservation stops the run and leaves its trajectory and manifest in place.

At most four task instances form a Smoke: `dev-01` and `dev-02`, each run first
in B1-M then B1-E. Since the complete 8,000-byte static upper bound is larger
than the Smoke cap, admission is pair-by-pair. The first pair must fit its
worst-case reservation; after it completes, the next pair is checked against
twice the measured pair charge. Development estimates remaining pairs from
twice the largest Smoke task-pair charge and checks that forecast against the
remaining project and B1 capacity. Every actual request still gets its own
atomic reservation. If a pair fails or cannot fit, the run stops without
automatic retries and archives progress.

## Standard price and account resources

The official price page lists GLM-5.3-Flash at ¥0.80 per million uncached input
tokens, ¥0.23 per million cached input tokens, and ¥2.80 per million output
tokens. Cost estimates apply those rates when usage detail is available and
are not invoices. Thinking is part of output billing when the provider reports
it there. The account console showed a resource package explicitly suitable
for `glm-5.3-flash`; FlashX and generic token packages are excluded from that
classification. Package eligibility does not establish the exact debit order.
The console does not expose experiment-specific cash or resource-package
debits, so V2 records those as unavailable rather than claiming a cash charge
or a package deduction.

## Offline regression and execution

Offline regressions cover the archived 5,326-byte request, three and four tool
calls, rejection above four, exact Flash response identity, mixed-model archive
rejection, compact evidence context, and token reservation/settlement. The
runner never sends a request for a different configured model or a response
whose returned model identity differs from `glm-5.3-flash`.

Example commands:

```bash
source "$HOME/.config/tradingagents-lite/glm.env"
export TALITE_MODEL=glm-5.3-flash
uv run --locked talite b1-v2 --scope smoke \
  --output results/b1-flash-smoke-001
uv run --locked talite b1-v2 --scope development \
  --smoke results/b1-flash-smoke-001 \
  --output results/b1-flash-development-001
```

Use a new output directory for every new run. An existing directory is refused
to prevent repeating a successful task instance. Development requires a fully
validated successful Flash Smoke and reuses its four successful task
instances once.
