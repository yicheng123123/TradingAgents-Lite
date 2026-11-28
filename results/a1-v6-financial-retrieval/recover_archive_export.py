"""Resume deterministic scoring from frozen predictions with a V3 schema adapter.

V3 archived gold-page ranks but did not store an explicit loss_attribution object.
Keep the frozen retrieval/scoring module unchanged and adapt only the old exports.
"""
from argparse import ArgumentParser

from tradingagents_lite import retrieval_v4
from tradingagents_lite.retrieval_v6.experiment import OUT, WORK
from tradingagents_lite.retrieval_v6.final import score_existing

summarize = retrieval_v4.summarize_rows


def summarize_with_v3_schema(rows):
    adapted = []
    for row in rows:
        if "loss_attribution" not in row:
            rank = [entry["after"] for entry in row["gold_page_ranks"]]
            row = {**row, "loss_attribution": {
                "A_top5": sum(value is not None and value <= 5 for value in rank),
                "B_in_pool_not_top5": sum(value is not None and value > 5 for value in rank),
                "C_not_in_pool": rank.count(None),
            }}
        adapted.append(row)
    return summarize(adapted)


if __name__ == "__main__":
    retrieval_v4.summarize_rows = summarize_with_v3_schema
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUT)
    parser.add_argument("--work", default=WORK)
    score_existing(parser.parse_args())
