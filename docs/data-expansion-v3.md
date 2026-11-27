# Data Expansion V3

## Frozen historical coverage

V3 market data spans July 1, 2025 through the last complete usable US session at collection, November 25, 2026. The 20 stocks, SPY and 8 sector ETFs all cover every expected session. The November 26 US holiday falls between that session and the November 27 shortened session, which had yet to start at collection. Yahoo OHLC uses one retrieved split scale per ticker for entry and exit. Adjusted Close and corporate actions are retained separately. Input bars use the share scale at the decision date. Returns use open and close on the same snapshot scale and exclude dividends. [NYSE session calendar](https://www.nyse.com/trade/hours-calendars)

SEC refreshed submissions and Company Facts for all 20 issuers. The collection stored 241 documents, including 20 10-K, 57 10-Q, two 10-K amendments, 84 earnings 8-K and 78 Exhibit 99.1 releases. Six exhibit links could not be verified. SEC-mirrored releases covered 19 issuers. Direct issuer collections included 119 Apple Newsroom records, four Microsoft IR releases and one Pfizer release. The combined direct and SEC-mirrored sources cover all 20 issuers with at least one release, while direct IR coverage is a selected 3 of 20 issuers.

SEC observations retain source provenance and accession versions. Later data for a prior period remains attached to the later accession. The query admits a filing only when its conservative eligibility timestamp passes the task cutoff. Both the SEC eligibility rule and publisher-date rule are historical proxies. Current archive responses do not establish that the exact retrieved bytes were historically obtainable at that time. The immutable metadata receipts, dates, download hashes and missing records are in [the task bundle](../tasks/data-expansion-v3/). Full publisher bodies, raw SEC documents, the market cache and Company Facts bundles live locally under `data/data-expansion-v3`.

## Task freeze and isolation

Development has 80 completed-window tasks covering 80 decision dates, January 2 through July 30, 2026. Holdout has 100 completed-window tasks covering 56 dates, September 1 through November 18, 2026. Both cover the original 20 stocks and 8 sectors. Development comes from the requested January–July window and is intended for protocol development. The posterior-release Holdout uses the preregistered September start because the probe audit did not identify a reliable statistical boundary.

Deterministic per-stock session strata favor the least-used dates inside each stratum. The sampler requires 21 warmup bars and the next five complete sessions for the stock, SPY and sector ETF. It excludes overlapping same-stock settlement windows, earlier task windows, and identified return-probe windows. V3 has zero same-stock outcome overlaps. The task manifests contain task IDs, decision dates, sampling and settlement definitions, source and partition hashes and coverage counts. No Holdout outcomes were evaluated or exported. The runner rejects V3 Holdout task IDs and seals outcomes from September onward on this market snapshot. Public historical prices remain independently reconstructable.

In the 80 Development views the audit checked 62,090 eligible XBRL fact instances, 280 visible documents or headlines, and 15,987 visible bars. There were zero future records in these views. All 80 had eligible XBRL, 67 had SEC filing evidence, and 25 had news. The audit verified the full Development settlement calendar separately from model inputs.

## Reproduction

Use a fresh output root for a new collection. The archived manifest SHA-256 values identify the original receipts. Later downloads can differ.

```bash
uv run --locked python -m tradingagents_lite.data_v3 market --through 2026-11-25
source ~/.config/tradingagents-lite/sec.env
uv run --locked python -m tradingagents_lite.data_v2_sec --root data/data-expansion-v3 --through 2026-11-25 --progress
uv run --locked python -m tradingagents_lite.data_v3 ir --through 2026-11-25
uv run --locked python -m tradingagents_lite.data_v3 ir-releases --through 2026-11-25
uv run --locked python -m tradingagents_lite.data_v3 macro --through 2026-11-25
uv run --locked python -m tradingagents_lite.data_v3 probe-sec
uv run --locked python -m tradingagents_lite.temporal_audit freeze
source ~/.config/tradingagents-lite/glm.env
uv run --locked python -m tradingagents_lite.temporal_audit run
uv run --locked python -m tradingagents_lite.data_v3 freeze
uv run --locked python -m tradingagents_lite.temporal_controls --out results/temporal-knowledge-audit-001/controls.json
uv run --locked pytest -q
```

The audit validates a preexisting budget ledger and refuses a duplicate experiment ID. To reproduce scoring without spending model Tokens, use the stored normalized responses in the audit results and the frozen probe answer key.
