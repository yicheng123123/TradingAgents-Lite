# Development evidence coverage

Coverage means at least one eligible version in the 180-day SEC or 30-day news window measured from the proxy eligibility timestamp. Results describe source coverage, not model accuracy.

| Source | Available | Proxy | Verified exact version | Unavailable |
|---|---:|---:|---:|---:|
| sec_filing | 20/20 | 20 | 0 | 0 |
| news | 10/20 | 10 | 0 | 10 |

| Task | Ticker | Decision date | Filing count | News count |
|---|---|---|---:|---:|
| dev-01 | AAPL | 2024-02-13 | 4 | 12 |
| dev-02 | MSFT | 2024-02-13 | 5 | 0 |
| dev-03 | AAPL | 2024-02-20 | 4 | 10 |
| dev-04 | MSFT | 2024-02-20 | 5 | 0 |
| dev-05 | AAPL | 2024-02-27 | 4 | 9 |
| dev-06 | MSFT | 2024-02-27 | 5 | 0 |
| dev-07 | AAPL | 2024-03-05 | 5 | 8 |
| dev-08 | MSFT | 2024-03-05 | 5 | 0 |
| dev-09 | AAPL | 2024-03-12 | 5 | 10 |
| dev-10 | MSFT | 2024-03-12 | 6 | 0 |
| dev-11 | AAPL | 2024-03-19 | 5 | 13 |
| dev-12 | MSFT | 2024-03-19 | 6 | 0 |
| dev-13 | AAPL | 2024-03-26 | 5 | 12 |
| dev-14 | MSFT | 2024-03-26 | 6 | 0 |
| dev-15 | AAPL | 2024-04-02 | 5 | 12 |
| dev-16 | MSFT | 2024-04-02 | 6 | 0 |
| dev-17 | AAPL | 2024-04-09 | 5 | 8 |
| dev-18 | MSFT | 2024-04-09 | 6 | 0 |
| dev-19 | AAPL | 2024-04-16 | 5 | 6 |
| dev-20 | MSFT | 2024-04-16 | 6 | 0 |

The successful news sources are issuer-controlled and sampled. Apple has archive dates only. Microsoft has a curated Security Blog subset. Missing rows do not prove an absence of company news. EDGAR uses primary accession documents only and keeps report period, filing day, acceptance and unverified public availability separate. See source_manifest.json for per-request errors and hashes.
