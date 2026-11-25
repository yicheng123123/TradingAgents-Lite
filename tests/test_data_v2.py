from datetime import date, timedelta
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from tradingagents_lite.data import DataError, HistoricalData, close_available, read_tasks, sha256
from tradingagents_lite.data_v2_market import ISSUERS, SYMBOLS, UNIVERSE, write_json
from tradingagents_lite.data_v2_sec import SecCache, dated_filing, exhibit_reference
from tradingagents_lite.data_v2_tasks import frozen_tasks
from tradingagents_lite.data_v2_evaluation import score_offline
from tradingagents_lite.evidence_http import SourceError
from tradingagents_lite.fundamentals import FundamentalsView, inline_company_facts, normalize
from tradingagents_lite.mcp_finance import MCPTransport
from tradingagents_lite.rag import Retriever
from tradingagents_lite.schema import Bar, Task
from tradingagents_lite.tool_registry import ToolRegistry
from tradingagents_lite.verifier import verify_citations
from tradingagents_lite.vectors import VectorStore

from test_rag import SmallEncoder, document
from tradingagents_lite.documents import chunk_document, parse_document


TASKS = Path("tasks/data-expansion-v2")
LEGACY_MARKET = Path("data/yahoo-2024-v1")
ACC = "0000320193-24-000001"
AMEND = "0000320193-24-000002"
END = "2023-12-31"
START = "2023-10-01"
SEC = "https://www.sec.gov"


def test_historical_cik_and_benchmarks():
    assert len(ISSUERS) == 20 and len(UNIVERSE) == 8 and len(SYMBOLS) == 29
    assert ISSUERS["XOM"] == ("Energy", "XLE", 34088)
    assert ISSUERS["META"] == ("Communication Services", "XLC", 1326801)
    assert ISSUERS["GOOGL"][2] == 1652044
    assert all(ticker in SYMBOLS for ticker in ("SPY", "XLI", "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLC"))


def test_edgar_acceptance_waits_for_next_session_and_99_1_stays_in_accession():
    row = {"accessionNumber": ACC, "filingDate": "2024-02-08", "acceptanceDateTime": "2024-02-09T23:30:00Z",
           "reportDate": END, "form": "10-Q", "primaryDocument": "filing.htm", "items": ""}
    days = (date(2024, 2, 8), date(2024, 2, 9), date(2024, 2, 12))
    filing = dated_filing("AAPL", 320193, row, days)
    assert filing["acceptance_datetime"] != filing["eligible_at"]
    assert filing["eligible_at"] == close_available(days[-1]).isoformat()
    assert filing["pit_level"] == "historical_proxy" and filing["first_seen_at"] is None
    assert exhibit_reference(b'<tr><td>99.1</td><td><a href="release.htm">Results</a></td></tr>', filing).endswith(
        "/000032019324000001/release.htm")
    assert exhibit_reference(b'<tr><td>99.1</td><td><a href="../release.htm">Results</a></td></tr>', filing) is None


@pytest.mark.parametrize("status", [403, 429])
def test_sec_access_limit_stops_new_network_requests(tmp_path, monkeypatch, status):
    monkeypatch.setenv("SEC_USER_AGENT", "Integration Test contact@example.test")
    cache = SecCache(tmp_path / "sec", tmp_path / "legacy", offline=False)
    calls = []

    def refuse(req, **kwargs):
        calls.append(req.full_url)
        raise HTTPError(req.full_url, status, "test", {}, None)

    cache.current.opener = refuse
    cache.current.sleep = lambda delay: None
    with pytest.raises(SourceError, match=f"http_{status}"):
        cache.get(SEC + "/first")
    with pytest.raises(SourceError, match="sec_network_stopped"):
        cache.get(SEC + "/second")
    assert len(calls) == 1


def fixture():
    filings = {}
    for accession, form, filed, eligible in ((ACC, "10-Q", "2024-02-02", "2024-02-05"),
                                             (AMEND, "10-Q/A", "2024-08-01", "2024-08-02")):
        filings[accession] = {"ticker": "AAPL", "accession": accession, "filing_date": filed, "form": form,
            "report_date": END, "eligible_at": close_available(date.fromisoformat(eligible)).isoformat(),
            "source_url": SEC + "/" + accession, "pit_level": "historical_proxy"}
    def fact(value, start=START, end=END, accession=ACC):
        f = filings[accession]
        return {"val": value, "start": start, "end": end, "accn": accession,
                "filed": f["filing_date"], "form": f["form"]}
    facts = {"cik": 320193, "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [fact(80, "2022-10-02", "2022-12-31"), fact(100), fact(90, accession=AMEND)],
                               "EUR": [fact(99999)]}},
        "NetIncomeLoss": {"units": {"USD": [fact(30), fact(999, end="2024-01-01")]}},
        "OperatingIncomeLoss": {"units": {"USD": [fact(35)]}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [fact(50, "2023-04-01")]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [fact(20, "2023-04-01"), fact(5)]}},
        "Assets": {"units": {"USD": [fact(200, None)]}},
        "DebtCurrentAndNoncurrent": {"units": {"USD": [fact(40, None)]}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": [fact(2)], "USD": [fact(100)]}},
    }}}
    return filings, normalize(facts, 320193, filings)


def test_xbrl_units_period_and_missing_values():
    filings, facts = fixture()
    task = read_tasks(Path("tasks/development.jsonl"))[0]
    view = FundamentalsView(task, "a" * 64, facts, filings, "expansion-v2")
    query = view.query(("Revenue", "Net Income", "Revenue YoY Growth", "Operating Margin", "Net Profit Margin",
                        "EPS", "Free Cash Flow", "Operating Cash Flow", "Debt / Assets", "Total Liabilities"))
    items = {row["metric"]: row for row in query["items"]}
    assert items["Revenue"]["value"] == 100 and items["Revenue"]["inputs"][0]["accession"] == ACC
    assert items["Net Income"]["value"] == 30
    assert items["Revenue YoY Growth"]["value"] == pytest.approx(.25)
    assert items["Operating Margin"]["value"] == pytest.approx(.35)
    assert items["Net Profit Margin"]["value"] == pytest.approx(.30)
    assert items["EPS"]["value"] == 2 and items["EPS"]["unit"] == "USD/shares"
    assert items["Debt / Assets"]["value"] == pytest.approx(.2) and items["Debt / Assets"]["period"] == "instant"
    assert set(query["unavailable"]) == {"Free Cash Flow", "Operating Cash Flow", "Total Liabilities"}
    cash = view.query(("Free Cash Flow",), "ytd")["items"][0]
    assert cash["value"] == 30 and {f["period_start"] for f in cash["inputs"]} == {"2023-04-01"}
    assert view.query(("Revenue",), "annual")["status"] == "unavailable"


def test_revision_before_and_after_cutoff_and_citation_scope():
    filings, facts = fixture()
    before = read_tasks(Path("tasks/development.jsonl"))[0]
    later = before.model_copy(update={"decision_date": date(2024, 8, 2)})
    def result(task):
        return FundamentalsView(task, "b" * 64, facts, filings, "expansion-v2").query(("Revenue",))["items"][0]
    first, revision = result(before), result(later)
    assert first["value"] == 100 and revision["value"] == 90
    assert revision["inputs"][0]["accession"] == AMEND
    assert first["evidence_id"] != revision["evidence_id"]
    key = revision["evidence_id"]
    assert verify_citations([key], {key: revision}, {key}, later, partition="expansion-v2")["passed"]
    future = verify_citations([key], {key: revision}, {key}, before, partition="expansion-v2")
    assert "future_evidence" in future["violations"] and not future["repairable"]


def test_inline_xbrl_requires_consolidated_context_currency_scale_and_same_cik():
    filing = {**fixture()[0][ACC], "cik": "0000320193"}
    raw = b"""
    <xbrli:context id="valid"><xbrli:entity><xbrli:identifier>320193</xbrli:identifier></xbrli:entity>
      <xbrli:period><xbrli:startDate>2023-10-01</xbrli:startDate>
      <xbrli:endDate>2023-12-31</xbrli:endDate></xbrli:period></xbrli:context>
    <xbrli:context id="segment"><xbrli:entity><xbrli:identifier>320193</xbrli:identifier>
      <xbrli:segment>Product only</xbrli:segment></xbrli:entity><xbrli:period>
      <xbrli:startDate>2023-10-01</xbrli:startDate><xbrli:endDate>2023-12-31</xbrli:endDate>
      </xbrli:period></xbrli:context>
    <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
    <xbrli:unit id="eur"><xbrli:measure>iso4217:EUR</xbrli:measure></xbrli:unit>
    <ix:nonFraction name="us-gaap:Revenues" contextRef="valid" unitRef="usd" scale="6"
      format="ixt:num-dot-decimal">1,234</ix:nonFraction>
    <ix:nonFraction name="us-gaap:Revenues" contextRef="valid" unitRef="eur" scale="6">99</ix:nonFraction>
    <ix:nonFraction name="us-gaap:Revenues" contextRef="segment" unitRef="usd">999</ix:nonFraction>
    """
    facts = inline_company_facts(raw, filing)
    assert len(facts) == 1 and facts[0]["value"] == 1_234_000_000
    assert inline_company_facts(raw, {**filing, "cik": "789019"}) == []


def test_real_mcp_fundamentals_equivalence_is_opt_in(tmp_path):
    filings, facts = fixture()
    task = read_tasks(Path("tasks/development.jsonl"))[0]
    path = tmp_path / "AAPL.json"
    write_json(path, {"facts": facts, "filings": filings})
    write_json(tmp_path / "manifest.json", {"issuers": {
        "AAPL": {"sha256": sha256(path), "company_facts_sha256": "a" * 64}
    }})
    market = HistoricalData(LEGACY_MARKET).view(task)
    assert "xbrl_fundamentals" not in ToolRegistry(market).definitions
    local = ToolRegistry(market, fundamentals=FundamentalsView.load(tmp_path, task, "expansion-v2"))
    args = json.dumps({"metrics": ["Revenue", "EPS"], "period": "quarter"})
    with MCPTransport({"task": task.model_dump(mode="json"), "market": str(LEGACY_MARKET.resolve()),
                       "fundamentals": {"root": str(tmp_path), "partition": "expansion-v2"}}) as bridge:
        assert local.execute("xbrl_fundamentals", args) == bridge.call("xbrl_fundamentals", args)
        assert bridge.call("xbrl_fundamentals", '{"ticker":"MSFT"}') == {"error": "invalid_arguments"}
        assert bridge.call("xbrl_fundamentals", '{"cutoff":"2025-12-31"}') == {"error": "invalid_arguments"}


def test_frozen_counts_and_holdout_seal():
    sets = {name: frozen_tasks(TASKS, name) for name in ("offline", "development", "holdout")}
    assert list(map(len, sets.values())) == [200, 80, 100]
    mixed = [task for tasks in sets.values() for task in tasks]
    assert len({(t.symbol, t.decision_date) for t in mixed}) == 380
    manifest = json.loads((TASKS / "manifest.json").read_text())
    assert manifest["v2_window_overlaps"] == 0
    with pytest.raises(ValueError, match="Offline Extended"):
        score_offline(sets["holdout"], None)
    market = HistoricalData.__new__(HistoricalData)
    market.manifest = {"version": "yahoo-us-v2-001"}
    for task in (sets["holdout"][0], sets["holdout"][0].model_copy(update={"task_id": "renamed"})):
        with pytest.raises(DataError, match="sealed"):
            market.outcome(task)


def test_market_view_keeps_future_rows_out_and_preserves_shared_settlement_scale():
    task = Task(task_id="v2-offline-aapl-01", symbol="AAPL", decision_date=date(2022, 2, 21), horizon_sessions=5)
    bars = []
    for i in range(-20, 6):
        day = task.decision_date + timedelta(days=i)
        price = 100 + i
        bars.append(Bar(day=day, available_at=close_available(day), open=price, high=price + 1,
                        low=price - 1, close=price, volume=1))
    market = HistoricalData.__new__(HistoricalData)
    market.manifest = {"version": "yahoo-us-v2-001", "benchmark": "SPY"}
    market.series = {"AAPL": tuple(bars), "SPY": tuple(bars)}
    market.split_events = {"AAPL": ((date(2022, 2, 22), 4),)}
    view = market.view(task)
    assert len(view.history) == 21 and view.history[-1].day == task.decision_date
    assert view.history[-1].close == 400 and market.series["AAPL"][20].close == 100
    assert all(row["day"] <= "2022-02-21" for row in view.get_bars(20)["bars"])
    assert market.outcome(task).asset_return == pytest.approx(105 / 101 - 1)


def test_rag_partition_and_index_version_are_independent(tmp_path):
    raw = b"<h1>Revenue</h1><p>Quarterly net sales increased.</p>"
    doc = document(raw)
    v2 = doc.model_copy(update={"dataset_partition": "expansion-v2"})
    rows = chunk_document(v2, parse_document(v2, raw))
    legacy_rows = chunk_document(doc, parse_document(doc, raw))
    assert rows[0]["version_id"] != legacy_rows[0]["version_id"]
    with VectorStore(tmp_path / "legacy", SmallEncoder(), "chunks") as legacy:
        legacy.put(legacy_rows, id_key="chunk_id", text_key="text")
        cutoff = max(doc.eligible_at, v2.eligible_at)
        args = {"ticker": v2.ticker, "cutoff": cutoff, "mode": "R1", "top_k": 2, "candidate_k": 5}
        assert Retriever(legacy).retrieve("revenue", partition="expansion-v2", **args)["eligible_chunks"] == 0
        with VectorStore(tmp_path / "v2", SmallEncoder(), "chunks") as expansion:
            expansion.put(rows, id_key="chunk_id", text_key="text")
            retriever = Retriever(expansion)
            assert retriever.retrieve("revenue", partition="expansion-v2", **args)["eligible_chunks"] == len(rows)
            assert retriever.retrieve("revenue", partition="holdout", **args)["eligible_chunks"] == 0
