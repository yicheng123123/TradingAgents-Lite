"""Synthetic reproductions for defects found during the offline Smoke audit."""
from decimal import Decimal

import pytest

from tradingagents_lite.financebench import CaseSearch
from tradingagents_lite.financebench_score import assess, numeric


def test_single_bounded_oracle_excerpt_keeps_periods_units_and_relevant_row():
    # Synthetic table: the header and intended row fit together without the intervening rows.
    header = "Statements of income (USD thousands) | 2022 | 2021"
    filler = "\n".join(f"Expense line {row} | 35 | 20" for row in range(40))
    target = "Operating income | 480 | 320"
    case = type("Case", (), {"question": "What was the 2022 operating income?"})()
    item = {"evidence_id": "fixture", "document_id": "synthetic", "page": 0,
            "excerpt": "\n".join((header, filler, target))}
    context = CaseSearch(case, None, {})
    excerpt, = [entry["excerpt"] for entry in context._deliver([item])["items"]]
    assert header in excerpt and target in excerpt


def test_plural_percent_cue_applies_to_unqualified_scalar():
    assert numeric("42", "Give the change in percents.") == (Decimal("42"), "%", Decimal(".01"))


def test_explicit_currency_amount_is_not_scaled_twice():
    question = "Give total revenue in USD millions."
    assert numeric("$250,000,000", question)[0] == numeric("250 million USD", question)[0]


@pytest.mark.parametrize("answer, gold, verdict", [
    ("0.81", "0.80", "correct"),
    ("0.82", "0.80", "incorrect"),
    ("(125 bps)", "-1.25%", "correct"),
    ("-USD 250 million", "(250 million USD)", "correct"),
    ("40% (from USD 10 million to USD 14 million)", "40%", "review"),
])
def test_numeric_controls_keep_existing_tolerance_and_review_policy(answer, gold, verdict):
    prediction = {"status": "ok", "answer_status": "answered", "answer": answer}
    assert assess(prediction, gold, "What was the reported figure?")[0] == verdict


@pytest.mark.parametrize("answer, question", [
    ("42", "Give the change in percent."),
    ("42", "Give the change in percents."),
    ("42", "Give the change in percentage."),
    ("42", "Give the change as a percentage."),
    ("42", "Give the change in %."),
    ("42 percents", "What change was reported?"),
    ("42 percentages", "What change was reported?"),
    ("42 percentage points", "What change was reported?"),
    ("42%", "What change was reported?"),
])
def test_percent_spellings_share_percentage_dimension(answer, question):
    parsed = numeric(answer, question)
    assert parsed is not None and parsed[1] == "%" and parsed[0] == Decimal("42")


@pytest.mark.parametrize("answer, question, expected", [
    ("250", "Give revenue in USD millions.", Decimal("250000000")),
    ("250 million USD", "Give revenue in USD millions.", Decimal("250000000")),
    ("$250,000,000", "Give revenue in USD millions.", Decimal("250000000")),
    ("2.5", "Give revenue in USD billions.", Decimal("2500000000.0")),
    ("$2,500,000,000", "Give revenue in USD billions.", Decimal("2500000000")),
    ("-250", "Give revenue in USD millions.", Decimal("-250000000")),
])
def test_millions_billions_and_explicit_absolute_amounts(answer, question, expected):
    assert numeric(answer, question)[0] == expected
