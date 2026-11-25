"""Consolidated SEC XBRL facts restricted to an accession and task cutoff."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Literal

from .data import close_available, sha256 as file_hash
from .schema import Task


TAGS = {
    "Revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet", "RevenuesNetOfInterestExpense", "TotalRevenuesAndOtherIncome"),
    "Net Income": ("NetIncomeLoss", "ProfitLoss"),
    "Operating Income": ("OperatingIncomeLoss",),
    "Operating Cash Flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "Capital Expenditure": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "Total Assets": ("Assets",),
    "Total Liabilities": ("Liabilities",),
    "Total Debt": ("DebtCurrentAndNoncurrent", "LongTermDebtAndShortTermBorrowings"),
    "EPS": ("EarningsPerShareDiluted", "EarningsPerShareBasic"),
}
DERIVED = ("Revenue YoY Growth", "Net Profit Margin", "Operating Margin", "Free Cash Flow", "Debt / Assets")
METRICS = (*TAGS, *DERIVED)
INSTANT = {"Total Assets", "Total Liabilities", "Total Debt"}
Period = Literal["quarter", "ytd", "annual"]


@dataclass(frozen=True)
class Fact:
    tag: str
    unit: str
    value: float | int
    start: date | None
    end: date
    filing: dict

    def provenance(self):
        return {"tag": self.tag, "unit": self.unit, "value": self.value,
                "period_start": self.start.isoformat() if self.start else None, "period_end": str(self.end),
                **{key: self.filing[key] for key in ("accession", "filing_date", "eligible_at", "source_url")},
                "source_sha256": self.filing.get("content_hash")}


def normalize(company_facts: dict, cik: int, filings: dict[str, dict]):
    if int(company_facts["cik"]) != cik:
        raise ValueError("Company Facts CIK mismatch")
    concepts = company_facts["facts"].get("us-gaap", {})
    data = []
    for metric, tags in TAGS.items():
        unit = "USD/shares" if metric == "EPS" else "USD"
        for tag in tags:
            for row in concepts.get(tag, {}).get("units", {}).get(unit, []):
                filing = filings.get(row.get("accn"))
                if (not filing or not filing["eligible_at"] or row.get("filed") != filing["filing_date"]
                        or row.get("form") != filing["form"] or type(row.get("val")) not in (int, float)
                        or not filing.get("report_date") or row["end"] > filing["report_date"]):
                    continue
                start, end = row.get("start"), row["end"]
                if (metric in INSTANT) != (start is None) or (start is not None and start > end):
                    continue
                data.append({"metric": metric, "tag": tag, "unit": unit, "value": row["val"],
                             "start": start, "end": end, "accession": filing["accession"]})
    return data


def inline_company_facts(raw: bytes, filing: dict) -> list[dict]:
    """Extract known, undimensioned US-GAAP numeric facts from a cached SEC accession."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "html.parser")
    unit_map = {}
    for unit in soup.find_all("xbrli:unit", id=True):
        measures = [node.get_text(strip=True).lower() for node in unit.find_all("xbrli:measure")]
        if measures == ["iso4217:usd"] and not unit.find("xbrli:divide"):
            unit_map[unit["id"]] = "USD"
        if measures == ["iso4217:usd", "xbrli:shares"] and unit.find("xbrli:divide"):
            unit_map[unit["id"]] = "USD/shares"
    contexts = {}
    for node in soup.find_all("xbrli:context", id=True):
        entity = node.find("xbrli:identifier")
        if (entity is None or not entity.get_text(strip=True).isdigit()
                or int(entity.get_text(strip=True)) != int(filing["cik"])
                or node.find(["xbrli:segment", "xbrli:scenario"])):
            continue
        start = node.find("xbrli:startdate")
        end = node.find("xbrli:enddate") or node.find("xbrli:instant")
        if end is not None:
            contexts[node["id"]] = (start.get_text(strip=True) if start else None, end.get_text(strip=True))
    facts = defaultdict(lambda: defaultdict(list))
    known = {tag for tags in TAGS.values() for tag in tags}
    for node in soup.find_all("ix:nonfraction"):
        tag = node.get("name", "").removeprefix("us-gaap:")
        context = contexts.get(node.get("contextref"))
        unit = unit_map.get(node.get("unitref"))
        formatter = node.get("format", "").split(":")[-1].casefold().replace("-", "")
        if (tag not in known or context is None or unit is None or node.get("xsi:nil") == "true"
                or formatter not in {"", "numdotdecimal", "numdotdecimalin", "numcommadot"}):
            continue
        text = node.get_text(strip=True)
        if not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", text):
            continue
        try:
            scale = int(node.get("scale", "0"))
            if abs(scale) > 12:
                continue
            number = Decimal(text.replace(",", "")).scaleb(scale)
        except (InvalidOperation, ValueError):
            continue
        if node.get("sign") == "-":
            number = -number
        amount = int(number) if number == number.to_integral_value() else float(number)
        start, end = context
        facts[tag][unit].append({"start": start, "end": end, "val": amount, "accn": filing["accession"],
                                 "filed": filing["filing_date"], "form": filing["form"]})
    payload = {"cik": filing["cik"], "facts": {"us-gaap": {tag: {"units": units} for tag, units in facts.items()}}}
    return normalize(payload, int(filing["cik"]), {filing["accession"]: filing})


class FundamentalsView:
    def __init__(self, task: Task, content_hash: str, facts: list[dict], filings: dict[str, dict],
                 partition: str = "development-v2"):
        self.task, self.content_hash, self.partition = task, content_hash, partition
        cutoff = close_available(task.decision_date)
        eligible = {
            key: filing for key, filing in filings.items()
            if filing.get("eligible_at") and filing.get("report_date")
            and filing["ticker"] == task.symbol and filing["pit_level"] in {"historical_proxy", "verified_known_at"}
            and date.fromisoformat(filing["report_date"]) <= task.decision_date
            and date.fromisoformat(filing["filing_date"]) <= task.decision_date
            and datetime.fromisoformat(filing["eligible_at"]) <= cutoff
        }
        self.facts = defaultdict(list)
        self.report_periods = defaultdict(set)
        for filing in eligible.values():
            self.report_periods[date.fromisoformat(filing["report_date"])].add(filing["form"].removesuffix("/A"))
        for row in facts:
            if filing := eligible.get(row["accession"]):
                self.facts[row["metric"]].append(Fact(row["tag"], row["unit"], row["value"],
                    date.fromisoformat(row["start"]) if row["start"] else None, date.fromisoformat(row["end"]), filing))

    @classmethod
    def load(cls, root: Path, task: Task, partition: str):
        path = root / f"{task.symbol}.json"
        manifest = json.loads((root / "manifest.json").read_text())
        entry = manifest["issuers"].get(task.symbol)
        if not entry:
            return cls(task, "", [], {}, partition)
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Fundamentals snapshot hash mismatch")
        bundle = json.loads(path.read_text())
        return cls(task, entry.get("company_facts_sha256") or entry["sha256"],
                   bundle["facts"], bundle["filings"], partition)

    @staticmethod
    def _period(fact: Fact, period: Period, instant: bool) -> bool:
        if instant:
            return fact.start is None
        if fact.start is None:
            return False
        days = (fact.end - fact.start).days + 1
        return {"quarter": 70 <= days <= 115, "ytd": 70 <= days <= 380,
                "annual": 345 <= days <= 380}[period]

    def _find(self, name, end, period, *, start=None):
        for tag in TAGS[name]:
            choices = [fact for fact in self.facts[name] if fact.tag == tag and fact.end == end
                       and self._period(fact, period, name in INSTANT) and (start is None or start == fact.start)]
            if not choices:
                continue
            if period == "ytd" and name not in INSTANT:
                earliest = min(fact.start for fact in choices)
                choices = [fact for fact in choices if fact.start == earliest]
            newest = max((fact.filing["eligible_at"], fact.filing["accession"]) for fact in choices)
            latest = [fact for fact in choices if (fact.filing["eligible_at"], fact.filing["accession"]) == newest]
            if len({(fact.value, fact.start, fact.end) for fact in latest}) == 1:
                return latest[0]
        return None

    def query(self, metrics: tuple[str, ...], period: Period = "quarter") -> dict:
        ends = [end for end, forms in self.report_periods.items() if period != "annual" or "10-K" in forms]
        end = max(ends, default=None)
        items, unavailable = [], []
        if end is not None:
            base = {key: self._find(key, end, period) for key in TAGS}
            for metric in metrics:
                result = self._metric(metric, base, period)
                if result is None:
                    unavailable.append(metric)
                else:
                    value, unit, inputs = result
                    release = max(f.filing["eligible_at"] for f in inputs)
                    sources = [fact.provenance() for fact in inputs]
                    row_period = "instant" if all(fact.start is None for fact in inputs) else period
                    document_id = sha256(json.dumps([metric, row_period, sources], sort_keys=True).encode()).hexdigest()
                    identity = [self.task.symbol, "sec_filing", document_id, self.content_hash]
                    items.append({
                        "metric": metric, "value": value, "unit": unit, "period": row_period, "period_end": str(end),
                        "document_id": document_id, "content_hash": self.content_hash, "ticker": self.task.symbol,
                        "source_type": "sec_filing", "dataset_partition": self.partition,
                        "eligible_at": release, "pit_level": ("historical_proxy"
                            if any(f.filing["pit_level"] == "historical_proxy" for f in inputs) else "verified_known_at"),
                        "source_url": inputs[0].filing["source_url"], "inputs": sources,
                        "evidence_id": "fin:" + sha256(json.dumps(identity).encode()).hexdigest(),
                    })
        else:
            unavailable = list(metrics)
        return {"status": "ok" if items else "unavailable", "source_type": "sec_xbrl",
                "items": items, "unavailable": unavailable}

    def _metric(self, name, base, period):
        if name in TAGS:
            fact = base[name]
            return None if fact is None else (fact.value, fact.unit, (fact,))
        if name == "Revenue YoY Growth":
            current = base["Revenue"]
            if current is None:
                return None
            offsets = range(359, 372)
            prior_ends = {f.end for f in self.facts["Revenue"]
                          if f.tag == current.tag and (current.end - f.end).days in offsets}
            candidates = [self._find("Revenue", end, period) for end in sorted(prior_ends)]
            for prior in candidates:
                if (prior and prior.tag == current.tag and prior.value > 0
                        and current.start and prior.start and (current.start - prior.start).days in offsets
                        and abs((current.end - current.start).days - (prior.end - prior.start).days) <= 8):
                    return current.value / prior.value - 1, "ratio", (current, prior)
            return None
        lhs, rhs = {
            "Net Profit Margin": ("Net Income", "Revenue"),
            "Operating Margin": ("Operating Income", "Revenue"),
            "Free Cash Flow": ("Operating Cash Flow", "Capital Expenditure"),
            "Debt / Assets": ("Total Debt", "Total Assets"),
        }[name]
        a, b = base[lhs], base[rhs]
        if not a or not b or (a.start, a.end, a.unit) != (b.start, b.end, b.unit):
            return None
        if name == "Free Cash Flow":
            return (a.value - b.value, "USD", (a, b)) if b.value >= 0 else None
        return (a.value / b.value, "ratio", (a, b)) if b.value > 0 else None
