from tradingagents_lite.embedding_v4 import DIMENSION, length_batches
from tradingagents_lite.retrieval_v4 import (
    candidate_oracle, choose_calibration, dense_top_k, paired_comparison, rank_candidates, source_fingerprint,
)


def test_length_batches_keep_every_input_and_bound_padding():
    lengths = [300, 12, 240, 10, 128, 72]
    batches = length_batches(lengths, batch_limit=3, token_limit=600)
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    assert all(len(batch) <= 3 for batch in batches)
    assert all(max(lengths[index] for index in batch) * len(batch) <= 600 for batch in batches)
    assert DIMENSION == 1024


def test_source_fingerprint_is_stable_and_ordered_by_point_id():
    rows = [
        {"point_id": "b", "chunk_id": "b", "text": "second"},
        {"point_id": "a", "chunk_id": "a", "text": "first"},
    ]
    first = source_fingerprint(sorted(rows, key=lambda row: row["point_id"]))
    second = source_fingerprint(sorted(rows, key=lambda row: row["point_id"]))
    assert first == second


def test_candidate_oracle_counts_distinct_pages():
    chunks = {
        "a": {"document_id": "doc", "page": 1},
        "b": {"document_id": "doc", "page": 1},
        "c": {"document_id": "doc", "page": 2},
    }
    assert candidate_oracle(["a", "b", "c"], chunks, {("doc", 1), ("doc", 2)}) == 1.0


def test_dense_depth_is_unavailable_when_archive_does_not_contain_it():
    rankings = {
        "24": {"dense": [(f"c{i}", 1.0 - i / 100) for i in range(24)]},
        "50": {"dense": [(f"c{i}", 1.0 - i / 100) for i in range(50)]},
    }
    assert dense_top_k(rankings, 5) == rankings["50"]["dense"][:5]
    assert dense_top_k(rankings, 50) == rankings["50"]["dense"]
    assert dense_top_k(rankings, 100) is None


class _FakeReranker:
    def score(self, query, texts):
        score = [float(index) for index in range(len(texts))]
        return score, {"milliseconds": 1.0, "peak_allocated_bytes": 10,
                       "peak_reserved_bytes": 20, "oom_batch_splits": 0}


def _measurement(keys):
    lexical = [(key, float(len(keys) - index)) for index, key in enumerate(keys)]
    dense_keys = keys[25:75]
    dense = [(key, float(len(dense_keys) - index)) for index, key in enumerate(dense_keys)]
    timing = {temperature: {str(depth): {"scope_ms": 1.0, "bm25_ms": 2.0,
                "encode_ms": 3.0, "dense_ms": 4.0, "peak_allocated_bytes": 5,
                "peak_reserved_bytes": 6, "oom_batch_splits": 0}
            for depth in (24, 50, 100)} for temperature in ("cold", "hot")}
    rankings = {str(depth): {"bm25": lexical[:depth], "dense": dense[:depth]}
                for depth in (24, 50, 100)}
    return {"rankings": rankings, "timing": timing}


def test_reranker_receives_entire_deduplicated_hybrid_union():
    keys = [f"chunk-{index:03d}" for index in range(100)]
    chunks = {key: {"text": f"body {key}", "document_id": "doc", "page": index}
              for index, key in enumerate(keys)}
    q = {"id": "q1", "company": "ABC", "query": "revenue fiscal 2023?"}
    measurement = _measurement(keys)
    computed, _ = rank_candidates([q], {"q1": measurement}, {"q1": measurement}, chunks,
                                  _FakeReranker(), run_e4=False)
    e3 = computed["q1"]["arms"]["E3-m3"]
    assert len(e3["candidate"]) == 75
    assert len(e3["ranked"]) == 75


def test_calibration_selection_prefers_recall_at_5_then_uses_deployment_gate():
    def arm(r5, latency):
        return {"page": {"recall@1": .2, "recall@5": r5, "recall@10": .4,
                         "mrr@10": .3, "ndcg@10": .3},
                "line_proxy": {"coverage_at_5": .5},
                "latency": {temperature: {"total_ms": {"mean": latency}}
                            for temperature in ("cold", "hot")}}
    summary = {"arms": {"E0-S3": arm(.4, 100), "E2-small": arm(.5, 150),
                         "E2-m3": arm(.55, 250), "E3-m3": arm(.6, 350)}}
    choice = choose_calibration(summary)
    assert choice["best_hybrid_no_reranker"] == "E2-m3"
    assert choice["quality_best"] == "E3-m3"
    assert choice["deployment_candidate"] == "E2-small"


def test_paired_comparison_reports_cluster_bootstrap_and_all_three_outcomes():
    def row(question, company, value):
        return {"question_sha256": question, "company_group": company,
                "page": {"recall@5": value}}
    result = paired_comparison(
        [row("q1", "co1", .5), row("q2", "co2", .25), row("q3", "co2", .5)],
        [row("q1", "co1", .25), row("q2", "co2", .5), row("q3", "co2", .5)],
    )
    assert result["improved"] == 1
    assert result["regressed"] == 1
    assert result["tied"] == 1
    assert result["companies"] == 2
    assert len(result["bootstrap_95_ci"]) == 2
