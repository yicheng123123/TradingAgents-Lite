import pytest

from tradingagents_lite.data_v2_evaluation import acceptance_gates
from tradingagents_lite.data_v2_corpus import compile_corpus
from tradingagents_lite.data_v2_market import write_json
from tradingagents_lite.data_v2_sec import (
    SecCache, build_sec, choose_documents, collector_lock, exhibit_reference, progress_event,
)
from tradingagents_lite.evidence_http import SourceError


def test_checked_cache_reused_with_fresh_output_and_counted_without_network(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    url = "https://data.sec.gov/submissions/CIK0000320193.json"
    key = hashlib.sha256(url.encode()).hexdigest()
    source = tmp_path / "history"
    source.mkdir()
    raw = b'{"cik":320193}'
    (source / f"{key}.body").write_bytes(raw)
    original = {"url": url, "ingested_at": "2026-01-01T20:00:00+00:00",
                "content_hash": hashlib.sha256(raw).hexdigest()}
    write_json(source / f"{key}.json", original)
    cache = SecCache(tmp_path / "new", tmp_path / "legacy", offline=True, reuse=(source,))
    cache.current.opener = lambda *a, **k: pytest.fail("Offline mode must not open a connection")
    assert cache.get(url).body == raw
    assert cache.get(url).body == raw
    assert (source / f"{key}.body").read_bytes() == raw
    assert cache.downloads == 0 and cache.cache_hits == 2
    event = progress_event("metadata", "AAPL", 1, 2, 0, cache, [], [])
    assert event["company"] == "AAPL" and event["remaining"] == 1 and event["percent"] == 50
    assert event["cache_hits"] == 2 and event["failures"] == 0


def test_collector_lock_excludes_second_output_and_releases_on_exit(tmp_path):
    lock = tmp_path / "collector.lock"
    with collector_lock(lock):
        with pytest.raises(SourceError, match="sec_collector_already_running"):
            with collector_lock(lock):
                pytest.fail("Exclusive lock should prevent overlapping collection")
    with collector_lock(lock):
        assert lock.is_file()


def test_published_snapshots_are_immutable(tmp_path):
    for directory in ("sec", "evidence"):
        (tmp_path / directory).mkdir()
        write_json(tmp_path / directory / "source_manifest.json", {"version": "published"})
    with pytest.raises(FileExistsError, match="SEC snapshot already published"):
        build_sec(tmp_path, tmp_path / "old_cache", offline=True)
    with pytest.raises(FileExistsError, match="Research corpus already published"):
        compile_corpus(tmp_path, tmp_path / "news.json")


def test_earnings_exhibit_index_must_point_to_same_accession():
    origin = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000001/"
    filing = {"source_url": origin + "filing.htm"}
    row = b"""
      <tr><td>2</td><td>News release</td><td><a href="/ix?doc=/Archives/edgar/data/320193/000032019324000001/release.htm">
        release.htm</a></td><td>EX-99.1</td></tr>
    """
    assert exhibit_reference(row, filing, index=True) == origin + "release.htm"
    foreign = row.replace(b"000032019324000001", b"000032019324000002")
    assert exhibit_reference(foreign, filing, index=True) is None


def test_warmup_periodic_filings_do_not_select_later_filings():
    def item(day, form="10-Q"):
        return {"filing_date": day, "accession": day, "form": form, "ticker": "AAPL", "items": ""}

    filings = [item("2020-07-31"), item("2020-08-03"), item("2021-02-01"), item("2026-01-01")]
    assert choose_documents(filings) == [filings[2]]
    assert choose_documents(filings, warmup=True) == filings[1:3]


def test_coverage_gates_keep_company_facts_distinct_from_cached_sec_documents():
    sec = {"documents_by_ticker": {"AAPL": 5}, "company_facts": {"AAPL": {"rows": 10}, "AMD": {"rows": 0}}}
    result = acceptance_gates(sec, {"offline": {"rag_candidates": 180, "pit_violations": 0},
                                    "development": {"rag_candidates": 71, "pit_violations": 0}})
    assert result["sec_issuers"] == {"actual": 1, "minimum": 20, "passed": False}
    assert result["company_facts_issuers"]["actual"] == 1
    assert result["offline_rag_tasks"]["passed"]
    assert not result["development_rag_tasks"]["passed"]
    assert result["pit_violations"]["passed"]
