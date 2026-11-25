# Frontend V2 · Financial Research Terminal

## Layout

The application uses a 216 px navigation rail and a shared status strip. Research has task controls, a market chart and execution flow, and a report. Agent Trace pairs an event timeline and duration waterfall with a node inspector. Evidence has a filtered library, a reading pane, and provenance. Memory, Benchmark, Usage, and Run History each have their own workspace.

A bookmarked `?page=Evidence&run=<id>` opens the same saved run. The only research execution action is **Run frozen task**. A paid run still requires a configured key and a one-run authorization, with the existing TokenBudget guard. Comparison and history use saved results. BigModel account refresh is an explicit, read-only action with its existing session cache.

## Design tokens

- Background `#0B101A`, panel `#111925`, elevated control `#182233`, border `#273244`
- Primary text `#E6EDF7`, secondary text `#94A3B8`, interactive accent `#38BDF8`
- Positive `#34D399`, negative `#F87171`, warning `#FBBF24`
- Page headings use 28 px, section headings 15 px, and captions 12–13 px. Tables inherit tabular figures. Keys, timestamps, and headline numbers use a system monospace fallback.
- Panels use 8 px radii, 8–14 px internal gaps, and 16–34 px responsive outer padding. Long identifiers wrap in detail panels. Budget cards show the abbreviated figure and the exact token count together. Wide data tables scroll within their own frame.
- Empty states explain which archive or selection supplies the missing data. Blocked runs use an error state and preserve the previously open archive. Unknown amounts and metrics use `Unavailable`.

Theme configuration lives in `.streamlit/config.toml` and `tradingagents_lite/terminal/theme.css`. `design.py` supplies shared headings, definition lists, numeric formats, badges, timeline markers, and card markup. Vega-Lite charts use Altair and pandas, both present in the Streamlit UI dependency group through Streamlit.

## Data provenance and display boundaries

- OHLCV uses hash-checked frozen Yahoo Finance bars via `HistoricalData.view`. The graph shows at most 55 sessions, through the archived decision cutoff. The last candle represents the decision session when available at the cutoff. The chart normalizes SPY and the asset to their first shared displayed close. Settlement appears separately in an evaluation expander.
- The report displays recorded decisions, cited evidence, verifier, risk, and measured usage. Available Development V2 XBRL values use eligible SEC facts. The trend reader groups comparable quarterly facts with the same decision cutoff.
- The trace waterfall accumulates recorded top-level node durations in order. Its origin is relative because the trace lacks per-event start timestamps. Nested tool measurements are listed separately, and archived JSON can be opened in developer details.
- Memory requires valid timezone-aware decision, eligibility, and settlement times, all at or before the selected cutoff. The timeline shows episodes retained by compaction. Its size comparison uses archived serialized byte counts.
- Benchmark names its input archives and denominators. A1 covers local retrieval, A2 covers deterministic replay and embeddings, and A3 covers rules, real MCP transport, mocks, and injected faults. GLM benchmark metrics require a real scored model archive.
- Usage shows the provider account view and the independently read local SQLite ledger. Its stacked bars use recorded settled and pending charges plus each stage's remaining budget. The list-price estimate uses ledger model usage with the existing pricing snapshot. The invoice remains unavailable.
- Run History opens single-task UI archives and per-task records in the A0/B0 replay archives. Older archives may omit agent events, retrieval scores, or detailed usage.

## Browser acceptance

The V2 screenshots are Chrome renders of the local Streamlit app. The reference run is Legacy task `dev-20` (MSFT, 2024-04-16), Offline, with the Full Multi-Agent preset. Chrome verified that reload and workspace navigation preserved the run identity and created no extra run archive. The baseline screenshots remain in the parent directory. Open the 1440 × 900 reference images below. The V2 folder also records the 1280 × 800 and 1920 × 1080 desktop checks.

- [Research](screenshots/v2/research.png)
- [Agent Trace](screenshots/v2/agent-trace.png)
- [Evidence](screenshots/v2/evidence.png)
- [Memory](screenshots/v2/memory.png)
- [Benchmark](screenshots/v2/benchmark.png)
- [Token Dashboard](screenshots/v2/token-dashboard.png)
- [Run History](screenshots/v2/run-history.png)
- [Unavailable run](screenshots/v2/unavailable-run.png)
