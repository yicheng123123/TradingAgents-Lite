from collections import Counter

import pytest

from tradingagents_lite.reranker_v3 import batches
from tradingagents_lite.retrieval_v3 import (
    evaluate, financial_boost, oracle_at_five, ordered_scores, rank_once, select,
)
from tradingagents_lite.retrieval_v3_candidates import pool, rank_ids


def chunk(i):
    return {"document_id": "a", "page": i, "text": f"text {i}", "title": "FY 2024", "section": ""}


class RecordingReranker:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def score(self, query, texts):
        self.calls.append((query, texts))
        if self.fail:
            raise ValueError("bad scores")
        return [float(i) for i in range(len(texts))], {
            "milliseconds": 1, "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048,
            "oom_batch_splits": 0,
        }


def test_rerank_uses_the_complete_deduplicated_union_and_same_model_inputs():
    lex = [(str(i), -i) for i in range(100)]
    dense = [(str(i), -i) for i in range(40, 140)]
    ranking = {"bm25": lex, "dense": dense}
    rows = {str(i): chunk(i) for i in range(140)}
    query = {"id": "id", "query": "Cash flow?", "gold": "must not enter", "answer": "also private"}
    small, big = RecordingReranker(), RecordingReranker()
    models = {"mini": small, "bge": big}
    s2 = rank_once("S2", query, ranking, rows, models)
    s3 = rank_once("S3", query, ranking, rows, models)
    s4 = rank_once("S4", query, ranking, rows, models)
    assert len(pool(lex, dense, 50)) == len(s2["scores"]) == len(s3["scores"]) == 90
    assert len(s4["scores"]) == 140
    assert small.calls[0] == big.calls[0]
    assert set(big.calls[0][1]) == {rows[k]["text"] for k in s2["pool"]}
    assert small.calls[0][0] == query["query"]
    assert s2["ranked"] == s3["ranked"]
    assert all(len(set(x["ranked"])) == len(x["pool"]) for x in (s2, s3, s4))


def test_oracle_counts_five_distinct_relevant_pages_not_the_whole_pool():
    rows = {str(i): chunk(i) for i in range(7)}
    rows["dup"] = chunk(1)
    gold = {("a", i) for i in range(7)}
    assert oracle_at_five(rows, rows, gold) == 5 / 7
    assert oracle_at_five(["1", "dup", "2"], rows, gold) == 2 / 7


def test_reranker_failure_keeps_gold_denominator_and_candidate_coverage():
    model = RecordingReranker(fail=True)
    q = {"id": "q1", "company": "synthetic", "query": "cash flow"}
    rows = {"1": chunk(1), "2": chunk(2)}
    candidate_ranks = {"bm25": [("1", 2)], "dense": [("2", .5)]}
    result = rank_once("S2", q, candidate_ranks, rows, {"mini": model})
    evaluated = evaluate(q, {"rank": result, "latency": {}},
                         {"gold": {("a", 1), ("a", 2)}, "lines": [{"1"}]}, rows, candidate_ranks)
    assert evaluated["failure"] == "ValueError"
    assert evaluated["candidate_count"] == 2 and evaluated["candidate_page_recall"] == 1
    assert evaluated["page"]["recall@5"] == 0
    assert all(page["after"] is None for page in evaluated["gold_page_ranks"])
    assert evaluated["line_proxy"]["top5_complete"] == 0


def test_financial_metafeature_does_not_equate_filing_year_to_report_period():
    q = "What was operating cash flow in 2023?"
    wrong = {"text": "net cash provided by operating activities 2022 194", "title": "filing 2023",
             "section": "Cash flows"}
    right = {**wrong, "title": "filing 2024", "text": "net cash provided by operating activities 2023 269"}
    assert financial_boost(q, right) > financial_boost(q, wrong) > 0
    assert financial_boost(q, wrong) == financial_boost(q, {**wrong, "title": "filing 2025"})


def test_dynamic_batches_restore_every_pair_and_respect_padding_budget():
    lengths = [12, 512, 92, 121, 300, 47, 512, 256, 199, 128, 333, 211]
    result = list(batches(lengths, batch_limit=4, token_limit=900))
    assert Counter(i for group in result for i in group) == Counter(range(len(lengths)))
    assert all(len(group) <= 4 and len(group) * max(lengths[i] for i in group) <= 900 for group in result)


def test_tie_breaking_and_timing_compare_identity_independent_of_score_noise():
    assert ordered_scores({"b": .5, "a": .5, "c": .6}) == ["c", "a", "b"]
    with pytest.raises(ValueError):
        ordered_scores({"a": float("nan")})
    assert rank_ids({"dense": [("a", .123), ("b", .121)]}) == rank_ids(
        {"dense": [("a", .123001), ("b", .121001)]})


def test_quality_winner_and_deployment_candidate_have_separate_decisions():
    def stats(r5, delay, failures=0):
        return {"page": {"recall@1": .2, "recall@5": r5, "recall@10": .8, "mrr@10": .3},
                "line_coverage_at_5": .4, "failures": failures,
                "latency": {key: {"total_ms": {"mean": delay}} for key in ("cold", "hot")}}
    assert select({"S0": stats(.4, 100), "S1": stats(.42, 101), "S4": stats(.6, 230)}) == ("S4", "S1")
    assert select({"S0": stats(.4, 100), "S1": stats(.42, 101, failures=1), "S4": stats(.6, 230)}) == ("S4", "S0")




@pytest.mark.parametrize("update", ({"ticker": "other"}, {"eligible_ts": 2**40},
                                   {"dataset_partition": "other"}, {"pit_level": "unsupported"}))
def test_rank_input_scope_rejects_cross_company_future_and_wrong_partition(update):
    from tradingagents_lite.retrieval_v3 import check_candidate_scope

    query = {"company": "own"}
    entry = {"rankings": {"50": {"bm25": [("a", 1)], "dense": []}}}
    legal = {"ticker": "own", "eligible_ts": 0, "dataset_partition": "financebench_public",
             "pit_level": "historical_proxy"}
    check_candidate_scope(query, entry, {"a": legal})
    with pytest.raises(ValueError, match="outside the frozen"):
        check_candidate_scope(query, entry, {"a": {**legal, **update}})



def test_formal_fingerprint_drift_fails_before_test_retrieval(tmp_path, monkeypatch):
    from tradingagents_lite import retrieval_v3

    retrieval_v3.write_json(tmp_path / "freeze.json", {"inputs": {"index": "sealed"}})
    monkeypatch.setattr(retrieval_v3, "local_fingerprints", lambda *args: {"index": "changed"})
    monkeypatch.setattr(retrieval_v3, "test_candidates", lambda *args: pytest.fail("test accessed before validation"))
    with pytest.raises(ValueError, match="no longer matches freeze"):
        retrieval_v3.formal(tmp_path, tmp_path, tmp_path, "cuda")
    assert not (tmp_path / "test-candidates.json").exists()
