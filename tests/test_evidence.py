from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from test_stage1 import TASK, bars
from tradingagents_lite.data import MarketView, close_available
from tradingagents_lite.edgar import collect_filings
from tradingagents_lite.evidence import Evidence, EvidenceStore, next_session_proxy
from tradingagents_lite.evidence_http import CachedHTTP, SourceError
from tradingagents_lite.news import collect_apple, collect_microsoft, MS_SECURITY_ARTICLES
from tradingagents_lite.schema import Budget, MarketReport, Proposal, Task
from tradingagents_lite.workflow import MomentumBackend, build_workflow, call_message

FIXTURES = Path(__file__).parent / "fixtures/evidence"
DATES = tuple(date(2024, 3, day) for day in (7, 8, 11, 12, 13, 14))
TASK_MARCH = Task(task_id="pit", symbol="AAPL", decision_date=date(2024, 3, 11), horizon_sessions=3)


def utc(stamp):
    return datetime.fromisoformat(stamp)


def evidence(kind, stamp, version, *, ticker="AAPL", document_id="same-id"):
    return Evidence(
        ticker=ticker, source_type=kind, document_id=document_id,
        source_url="https://example.test/source",
        eligible_at=utc(stamp), ingested_at=utc("2026-11-23T12:00:00Z"),
        content_hash=hashlib.sha256(version.encode()).hexdigest(), excerpt=version,
        availability_basis="synthetic_fixture_proxy", pit_level="historical_proxy",
    )


class Response:
    def __init__(self, url, data):
        self.url, self.data = url, data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def geturl(self):
        return self.url

    def read(self, count):
        return self.data[:count]


def test_new_york_dst_and_conservative_filing_gate():
    assert close_available(date(2024, 3, 8)) == utc("2024-03-08T21:15:00Z")
    assert close_available(date(2024, 3, 11)) == utc("2024-03-11T20:15:00Z")
    assert close_available(date(2024, 11, 4)) == utc("2024-11-04T21:15:00Z")
    assert next_session_proxy(date(2024, 3, 8), DATES) == utc("2024-03-11T20:15:00Z")
    holiday_grid = (date(2024, 3, 28), date(2024, 4, 1))
    assert next_session_proxy(holiday_grid[0], holiday_grid) == utc("2024-04-01T20:15:00Z")


@pytest.mark.parametrize("kind", ["sec_filing", "news"])
def test_future_docs_and_other_ticker_remain_invisible(kind):
    original = evidence(kind, "2024-03-11T20:15:00Z", "earlier")
    future = evidence(kind, "2024-03-11T20:15:01Z", "future")
    other = evidence(kind, "2024-03-11T20:00:00Z", "other ticker", ticker="MSFT")
    base = EvidenceStore([original]).view(TASK_MARCH).search(kind)
    revised = EvidenceStore([original, future, other]).view(TASK_MARCH).search(kind)
    assert base == revised
    assert len(revised["items"]) == 1
    assert "ingested_at" not in revised["items"][0]
    assert revised["items"][0]["decision_cutoff"] == "2024-03-11T20:15:00+00:00"


def test_same_id_versions_preserved_and_unattested_collisions_withheld():
    original = evidence("news", "2024-03-08T21:15:00Z", "original")
    revised = evidence("news", "2024-03-11T20:15:00Z", "newer version")
    store = EvidenceStore([original, revised])
    assert len({v.evidence_id for v in store.records}) == 2
    assert len(store.view(TASK_MARCH).visible("news")) == 2
    unknown_revision = evidence("news", "2024-03-08T21:15:00Z", "ambiguous version")
    store = EvidenceStore([original, unknown_revision])
    assert len(store.records) == 2
    assert store.view(TASK_MARCH).search("news")["status"] == "unavailable"


def test_verified_requires_exact_version_attestation_and_timezone():
    data = evidence("news", "2024-03-08T21:15:00Z", "title").model_dump()
    with pytest.raises(ValueError, match="attestation"):
        Evidence(**(data | {"pit_level": "verified_known_at"}))
    with pytest.raises(ValueError):
        Evidence(**(data | {"published_at": datetime(2024, 3, 1)}))
    with pytest.raises(ValueError, match="precedes"):
        Evidence(**(data | {"published_at": utc("2024-03-12T00:00:00Z")}))
    verified = Evidence(**(data | {"pit_level": "verified_known_at",
        "first_seen_at": utc("2024-03-08T16:14:00-05:00"),
        "source_metadata": {"version_attestation_sha256": data["content_hash"],
                            "version_attestation_source": "synthetic timestamped archive"}}))
    assert verified.first_seen_at == utc("2024-03-08T21:14:00Z")
    assert verified.first_seen_at.tzinfo == UTC


def test_sec_accession_and_amendment_cached_offline(tmp_path):
    requests, delays = [], []
    def fetch(request, timeout):
        requests.append(request)
        name = "edgar_submissions.json" if "submissions/" in request.full_url else "edgar_filing.html"
        return Response(request.full_url, (FIXTURES / name).read_bytes())
    ua = "Test fixture contact@example.test"
    live = CachedHTTP(tmp_path, sec_user_agent=ua, opener=fetch, sleep=delays.append)
    dates = (DATES[0], DATES[-1])
    rows, info = collect_filings(live, "AAPL", *dates, DATES)
    assert info["status"] == "available" and len(rows) == 2
    assert rows[0].source_metadata["form"] == "10-Q"
    assert rows[1].source_metadata["form"] == "10-Q/A"
    assert rows[0].source_metadata["report_date"] == "2023-12-30"
    assert rows[0].source_metadata["filing_date"] == "2024-03-08"
    assert rows[0].source_metadata["accepted_at"].startswith("2024-03-08T23:40")
    assert rows[0].source_metadata["public_available_at"] is None
    assert rows[0].published_at is None
    assert rows[0].eligible_at == close_available(date(2024, 3, 11))
    assert rows[1].eligible_at == close_available(date(2024, 3, 13))
    assert rows[0].document_id != rows[1].document_id
    assert "hidden" not in rows[0].excerpt
    assert all(r.get_header("User-agent") == ua for r in requests)
    assert delays == [1.25] * 3
    def no_network(*args, **kwargs):
        raise AssertionError("attempted network in offline mode")
    replay = CachedHTTP(tmp_path, offline=True, sec_user_agent=ua, opener=no_network)
    assert collect_filings(replay, "AAPL", *dates, DATES) == (rows, info)
    body = next(tmp_path.glob("*.body"))
    body.write_text("altered")
    envelope = json.loads(body.with_suffix(".json").read_text())
    with pytest.raises(SourceError, match="checksum mismatch"):
        replay.get(envelope["url"])


def test_network_failure_reported_and_retried(tmp_path):
    delays = []
    def fail(request, timeout):
        raise URLError("fixture network failure")
    adapter = CachedHTTP(tmp_path, sec_user_agent="Fixture", opener=fail, sleep=delays.append)
    rows, outcome = collect_filings(adapter, "AAPL", DATES[0], DATES[-1], DATES)
    assert rows == [] and outcome["status"] == "unavailable"
    assert outcome["failures"] == [{"stage": "submissions", "code": "network_unavailable"}]
    assert delays == [1.25, 1, 1.25, 2, 1.25]


def test_sec_403_is_not_retried(tmp_path):
    interactions = []
    def forbid(request, timeout):
        interactions.append(request.full_url)
        raise HTTPError(request.full_url, 403, "Forbidden", {}, None)
    http = CachedHTTP(tmp_path, sec_user_agent="Fixture", opener=forbid, sleep=lambda _: None)
    with pytest.raises(SourceError, match="http_403"):
        http.get("https://data.sec.gov/submissions/CIK0000320193.json")
    assert len(interactions) == 1


def test_apple_metadata_only_cache_and_offline_replay(tmp_path):
    def fetch(request, timeout):
        return Response(request.full_url, (FIXTURES / "apple_archive.html").read_bytes())
    http = CachedHTTP(tmp_path, sec_user_agent="Fixture", opener=fetch, sleep=lambda _: None)
    rows, info = collect_apple(http, DATES[0], DATES[-1], DATES, [1])
    assert len(rows) == 2
    assert info["requested"][0]["retention"] == "metadata_only"
    assert not list(tmp_path.glob("*.body"))
    assert len(EvidenceStore(rows).view(TASK_MARCH).visible("news")) == 1
    offline = CachedHTTP(tmp_path, offline=True, sec_user_agent="Fixture")
    assert collect_apple(offline, DATES[0], DATES[-1], DATES, [1]) == (rows, info)


def test_microsoft_withholds_version_modified_after_historical_window(tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents_lite.news.MS_SECURITY_ARTICLES", MS_SECURITY_ARTICLES[2:3])
    def fetch(request, timeout):
        raw = b"""<meta property="og:title" content="Synthetic product headline">
<meta property="article:published_time" content="2024-03-13T16:00:00Z">
<meta property="article:modified_time" content="2025-05-01T00:00:00Z">"""
        return Response(request.full_url, raw)
    http = CachedHTTP(tmp_path, sec_user_agent="Fixture", opener=fetch, sleep=lambda _: None)
    rows, info = collect_microsoft(http, DATES[0], DATES[-1], DATES)
    assert len(rows) == 1 and rows[0].pit_level == "unavailable"
    assert rows[0].eligible_at is None and info["status"] == "unavailable"


def test_graph_offers_and_cites_cutoff_filtered_evidence():
    class ScriptedAnalyst(MomentumBackend):
        def analyst(self, messages, tools):
            if messages[-1]["role"] != "tool":
                assert any(t["function"]["name"] == "financial_evidence" for t in tools)
                msg = call_message("financial_evidence")
                msg["tool_calls"][0]["function"]["arguments"] = '{"source_type":"news"}'
                return msg
            result = json.loads(messages[-1]["content"])
            item = result["items"][0]
            assert result["status"] == "available" and item["excerpt"] == "visible"
            return {"content": MarketReport(view="Read source title.", evidence_ids=[item["evidence_id"]]).model_dump_json()}

        def trader(self, report, evidence):
            assert len(evidence) == 1
            return Proposal(action="hold", rationale="Test citation transfer.", evidence_ids=report.evidence_ids)

    doc = evidence("news", "2024-01-19T21:15:00Z", "visible", ticker=TASK.symbol)
    future = evidence("news", "2024-01-22T21:15:00Z", "never in graph", ticker=TASK.symbol)
    budget = Budget()
    market = MarketView(TASK, bars())
    state = build_workflow(market, ScriptedAnalyst(budget), budget,
                           financial=EvidenceStore([doc, future]).view(TASK)).invoke({})
    assert state["decision"]["evidence_ids"] == [doc.evidence_id]
    assert state["evidence"][doc.evidence_id]["source_url"] == "https://example.test/source"
    assert future.evidence_id not in state["evidence"]
    assert "never in graph" not in json.dumps(state)
