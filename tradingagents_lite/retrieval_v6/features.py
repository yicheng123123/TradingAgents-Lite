"""Financial query features that can be calculated without labels."""
from collections import Counter
from dataclasses import dataclass
import re

# Separate metric keys distinguish related accounting concepts.
METRICS = {
    "revenue": ("income", "revenue", "revenues", "net sales", "total sales"),
    "gross profit": ("income", "gross profit", "gross income"),
    "gross margin": ("income", "gross margin", "gross profit margin"),
    "operating income": ("income", "operating income", "income from operations", "operating profit"),
    "ebit": ("income", "ebit", "earnings before interest and taxes"),
    "ebitda": ("income", "ebitda", "earnings before interest taxes depreciation and amortization"),
    "net income": ("income", "net income", "net earnings", "net profit", "net loss"),
    "operating cash flow": ("cash flow", "operating cash flow", "cash flow from operations",
                           "net cash provided by operating activities", "cash from operating activities",
                           "net cash used in operating activities"),
    "capital expenditure": ("cash flow", "capital expenditure", "capital expenditures", "capex",
                            "purchases of property plant and equipment", "additions to property plant and equipment",
                            "purchases of property and equipment"),
    "free cash flow": ("cash flow", "free cash flow", "fcf"),
    "total assets": ("balance", "total assets"),
    "total liabilities": ("balance", "total liabilities"),
    "equity": ("balance", "shareholders equity", "stockholders equity", "total equity", "shareowners equity"),
    "cash and cash equivalents": ("balance", "cash and cash equivalents", "cash equivalents"),
    "debt": ("balance", "debt", "borrowings", "long term debt", "short term debt"),
    "accounts receivable": ("balance", "accounts receivable", "trade receivables", "receivables"),
    "inventory": ("balance", "inventory", "inventories"),
    "eps": ("income", "eps", "earnings per share", "net income per share", "net earnings per share"),
    "research and development": ("income", "research and development", "r d", "research development"),
    "selling general and administrative": ("income", "selling general and administrative", "sg a", "s g a",
                                            "selling general administrative"),
}
STATEMENTS = {
    "income": ("statement of operations", "statements of operations", "statement of income",
               "statements of income", "statement of earnings", "statements of earnings", "income statement"),
    "cash flow": ("statement of cash flows", "statements of cash flows", "cash flow statement"),
    "balance": ("balance sheet", "balance sheets", "statement of financial position", "statements of financial position"),
}
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
QTR = re.compile(r"\bq([1-4])(?=\D|$)|\b(first|second|third|fourth) quarter\b", re.I)
NUMBER = re.compile(r"(?<!\w)(?:[$€£]\s*)?(?:\([+\-−]?\s*\d[\d,]*(?:\.\d+)?\s*%?\)|[+\-−]?\d[\d,]*(?:\.\d+)?%?)(?!\w)")
WEIGHTS = {"M": .17, "Y": .09, "F": .06, "H": .04, "N": .10, "D": .04}
STAGES = ((), ("M",), ("M","Y"), ("M","Y","F","H"), ("M","Y","F","H","N"), tuple(WEIGHTS))


def clean(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("’", "").replace("'", "")))


def find_metrics(text):
    normalized = clean(text)
    matches = []
    for metric, (_, *aliases) in METRICS.items():
        for alias in aliases:
            for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", normalized):
                matches.append((metric, *match.span()))
    accepted = []
    for metric, start, end in sorted(matches, key=lambda m: (m[1], -(m[2]-m[1]), m[0])):
        if not any(start < b and end > a for _, a, b in accepted):
            accepted.append((metric, start, end))
    return tuple(dict.fromkeys(metric for metric, _, _ in accepted))


def statements(text):
    normal = " " + clean(text) + " "
    return {key for key, names in STATEMENTS.items() if any(" " + name + " " in normal for name in names)}


def quarters(text):
    names = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
    return {number or names[name.casefold()] for number, name in QTR.findall(text)}


@dataclass(frozen=True)
class Intent:
    metrics: tuple[str, ...]
    years: tuple[str, ...]
    quarters: tuple[str, ...]
    statements: tuple[str, ...]
    period: str
    kind: str
    comparison: str

    @property
    def calculation(self):
        return len(self.metrics) > 1 or self.comparison != "none" or self.kind != "amount"


def parse(query: str) -> Intent:
    normal = clean(query)
    metrics = find_metrics(query)
    years = tuple(dict.fromkeys(YEAR.findall(query)))
    q = tuple(sorted(quarters(query)))
    target = statements(query) or {METRICS[m][0] for m in metrics}
    period = ("quarter" if q or "three months" in normal or "quarter" in normal else
              "year_to_date" if re.search(r"\b(?:six|nine) months|year to date\b", normal) else
              "point_in_time" if target == {"balance"} or "as of" in normal else
              "annual" if years or "fiscal year" in normal else "unknown")
    kind = ("growth" if re.search(r"\b(?:growth|change|increase|decrease)\b", normal) else
            "ratio" if re.search(r"\b(?:percentage|percent|ratio|margin|yield)\b", normal) else "amount")
    comparison = ("yoy" if "year over year" in normal or "yoy" in normal else
                  "qoq" if "quarter over quarter" in normal or "qoq" in normal else
                  "cross_year" if len(years) > 1 else "none")
    return Intent(metrics, years, q, tuple(sorted(target)), period, kind, comparison)


def expansions(query: str, intent: Intent) -> list[str]:
    if not intent.metrics:
        return [query]
    period = " ".join([*("FY" + year for year in intent.years), *("Q" + quarter for quarter in intent.quarters)])
    first_aliases = [METRICS[m][1] for m in intent.metrics]
    second_aliases = [METRICS[m][-1] for m in intent.metrics]
    values = [
        query,
        " ".join([*first_aliases, *second_aliases, period]).strip(),
        " ".join([*(STATEMENTS[s][0] for s in intent.statements), *first_aliases, period]).strip(),
    ]
    return list(dict.fromkeys(values))


def document_features(row):
    text = row["text"]
    metrics = set(find_metrics(text))
    numeric = set()
    # Keep signs and delimiters and require a non-year number near a financial alias.
    for metric in metrics:
        for alias in METRICS[metric][1:]:
            pattern = r"(?<!\w)" + r"[\W_]+".join(alias.split()) + r"(?!\w)"
            for match in re.finditer(pattern, text, re.I):
                span = text[match.end():match.end()+160].split("\n", 1)[0]
                if any(not YEAR.fullmatch(number.group()) for number in NUMBER.finditer(span)):
                    numeric.add(metric)
                    break
            if metric in numeric:
                break
    return {"metrics": metrics, "years": set(YEAR.findall(text)), "quarters": quarters(text),
            "statements": statements(text[:3000]), "head": set(find_metrics(row.get("section", ""))),
            "numeric": numeric}


def fraction(targets, found):
    return sum(target in found for target in targets) / len(targets) if targets else 0.


def features(intent, document, logit, bm25_rank, dense_rank):
    score = sum(1 / (60 + rank) for rank in (bm25_rank, dense_rank) if rank is not None) / (2/61)
    year = fraction(intent.years, document["years"])
    quarter = fraction(intent.quarters, document["quarters"])
    period = (year + quarter)/2 if intent.years and intent.quarters else year or quarter
    return {"S": logit, "M": fraction(intent.metrics, document["metrics"]), "Y": period,
            "F": fraction(intent.statements, document["statements"]),
            "H": fraction(intent.metrics, document["head"]), "N": fraction(intent.metrics, document["numeric"]),
            "D": score}


def page_key(row):
    return row["document_id"], row["page"]


def rank(feature_map, rows, parsed, stage=0, fusion="residual", pages="none"):
    raw = sorted(feature_map, key=lambda key: (-feature_map[key]["S"], key))
    if not parsed.metrics:
        return raw
    active = STAGES[stage]
    normalized = {key: 1 - i/max(1, len(raw)-1) for i, key in enumerate(raw)}
    def score(key):
        values = feature_map[key]
        boost = sum(WEIGHTS[name] * values[name] for name in active)
        if not stage:
            return values["S"] if pages == "none" else normalized[key]
        return (values["S"] + boost) if fusion == "residual" and pages == "none" else (.8*normalized[key] + .4*boost)
    ordered = sorted(raw, key=lambda key: (-score(key), key))
    if pages == "none":
        return ordered
    remaining = ordered.copy()
    head, counts = [], Counter()
    while remaining and len(head) < 5:
        if pages == "distinct":
            key = next((key for key in remaining if not counts[page_key(rows[key])]), remaining[0])
        else:
            key = max(remaining, key=lambda key: score(key) - .10*counts[page_key(rows[key])])
        remaining.remove(key)
        head.append(key)
        counts[page_key(rows[key])] += 1
    return head + remaining


def top5(ranked):
    return list(ranked[:5])
