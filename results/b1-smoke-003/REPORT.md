# B1 Real LLM Development Baseline

Status: blocked_missing_api_key. Scope: smoke.

B0 is the deterministic five-session momentum rule. B1-M adds the real model with market-only tools; B1-E uses the same model and core prompts with the cutoff-bound A0.5 financial-evidence tool. No B1 task is scored unless the workflow, structured outputs, citations, and settlement all complete.

Frozen Development task SHA-256: 2d0e4f92d583d64ceba729b4e3ffa0004b3bd29a5a00d990434dcdab5fceedae
Selected task IDs: dev-01, dev-02
Market manifest SHA-256: f5e4c5095953ce737b7b99bba71b2ad26a9f30d2f34b9fba8320027e46e0f3cb
Evidence manifest SHA-256: b46f9ef2ebc1c08022500282147940985eb9d3ceabe8cdaf78eec6efa190e4dc
Evidence JSONL SHA-256: b0ffce595575257231432650495d756681a9f4ac10f5deee972110c7a7b7a532
Core prompt hashes: {"combined_core": "9b2181b5c519d5e434ec66882548132d5401be9ab60fc41023568588af75d956", "market_analyst_system": "b69b8d8a9ede3e7070f5c7abece218e8c0d0514802edce169bd5193cbb72e93d", "trader_system": "e8f6b4e6a21bf93628e8552037cd3fdeaf33a06e3e774e8856fd3ee72818140e"}

## Results

| Variant | Scope | Ticker | Completed / requested | Task completion | Structured valid | Non-Hold coverage | Hit rate (non-Hold) | Mean alpha | Tool success | Recovery events / rate | Citation valid / invalid | LLM calls | Input / output tokens | API cost (CNY) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | development-20 | AAPL | 10 / 10 | 1.0 | 1.0 | 0.7 | 0.42857142857142855 | 0.0030709588397909494 | 1.0 | 0 / None | 1.0 / 0.0 | 0 | 0 / 0 | 0.0 |
| B0 | development-20 | MSFT | 10 / 10 | 1.0 | 1.0 | 0.7 | 0.42857142857142855 | 0.003880497488325152 | 1.0 | 0 / None | 1.0 / 0.0 | 0 | 0 / 0 | 0.0 |
| B0 | development-20 | ALL | 20 / 20 | 1.0 | 1.0 | 0.7 | 0.42857142857142855 | 0.0034757281640580505 | 1.0 | 0 / None | 1.0 / 0.0 | 0 | 0 / 0 | 0.0 |
| B0 | paired-smoke-2 | AAPL | 1 / 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.021717380579178935 | 1.0 | 0 / None | 1.0 / 0.0 | 0 | 0 / 0 | 0.0 |
| B0 | paired-smoke-2 | MSFT | 1 / 1 | 1.0 | 1.0 | 0.0 | - | 0.0 | 1.0 | 0 / None | 1.0 / 0.0 | 0 | 0 / 0 | 0.0 |
| B0 | paired-smoke-2 | ALL | 2 / 2 | 1.0 | 1.0 | 0.5 | 1.0 | 0.010858690289589468 | 1.0 | 0 / None | 1.0 / 0.0 | 0 | 0 / 0 | 0.0 |
| B1-M | smoke | AAPL | - / 1 | - | - | - | - | - | - | 0 / None | - / - | - | - | - |
| B1-M | smoke | MSFT | - / 1 | - | - | - | - | - | - | 0 / None | - / - | - | - | - |
| B1-M | smoke | ALL | - / 2 | - | - | - | - | - | - | 0 / None | - / - | - | - | - |
| B1-E | smoke | AAPL | - / 1 | - | - | - | - | - | - | 0 / None | - / - | - | - | - |
| B1-E | smoke | MSFT | - / 1 | - | - | - | - | - | - | 0 / None | - / - | - | - | - |
| B1-E | smoke | ALL | - / 2 | - | - | - | - | - | - | 0 / None | - / - | - | - | - |

B0 full Development metrics:

- Tasks: 20; completed: 20; failures: 0.
- Non-Hold: 14; coverage: 0.7; directional hit rate: 0.42857142857142855; Wilson 95%: [0.21380798904474121, 0.6740935542034926].
- Mean alpha: 0.0034757281640580505; date-cluster interval: [-0.0013285677542173302, 0.007854122946365876].
- B0 paired-scope subset contains 2 tasks.

## Cost and execution limits

The conservative standard-list-price ceiling for this scope is 9.59488 CNY. It assumes every request reaches the 32768-byte prompt cap plus 512 protocol-token allowance, all bounded retries are charged, and each response uses the full 1200-token cap. This is a ceiling, not an observed bill.
The rough smoke expectation before API usage is approximately 0.2–0.5 CNY, assuming 12 model requests, 1–3k input tokens and 200–500 output tokens per request. This is an estimate only. GLM-5.3 standard list prices were captured from the official pricing page; account discounts, cache treatment and billing may differ.

## Interpretation limits

The smoke pair uses two Development tasks (one AAPL and one MSFT) on one decision date. It is an integration check, not a performance estimate; no meaningful confidence interval can be computed from one date cluster. Directional accuracy uses the non-Hold denominator; failures are reported separately. Alpha is per-task price-return excess over SPY, without fees, dividends, borrow or portfolio execution. The underlying Yahoo snapshot and A0.5 issuer evidence are historical proxies, not certified point-in-time data.

This is not an official TradingAgents benchmark and B0, B1-M and B1-E are Lite internal comparisons.

## Blocker

TALITE_API_KEY is absent in the current WSL environment. Set it in the shell without printing or committing it, then rerun with a new --output directory. The configured model and endpoint were taken from the checked official GLM documentation, but account-level authentication and actual response model identity remain untested.

No real model request was sent. No B1 accuracy, token usage, latency or cost is claimed.
