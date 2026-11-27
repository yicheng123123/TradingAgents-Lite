"""Optional, allowlisted model projection for historical research contexts."""
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import hmac
import math

from .data import close_available
from .fundamentals import METRICS
from .tool_registry import FINANCIAL_SPEC, FUNDAMENTALS_SPEC, RAG_SPEC, SPECS


PRICE = {"open", "high", "low", "close", "sma_20", "price", "eps"}
RATIOS = {"return_5", "daily_volatility_20", "margin", "growth", "confidence"}
DATES = {"day", "date", "decision_date", "period_start", "period_end", "published_at", "eligible_at", "available_at"}
ENUMS = {
    "source_type": {"sec_filing", "sec_xbrl", "news", "market", "synthetic"},
    "action": {"buy", "sell", "hold"},
    "stance": {"bull", "bear", "neutral"},
    "role": {"analyst", "trader", "bull", "bear", "verifier"},
    "event": {"earnings", "guidance_withdrawn", "guidance_raised", "unavailable"},
    "period": {"quarter", "ytd", "annual"},
}
TOPICS = {"earnings", "risks", "guidance", "liquidity", "operations"}
TOOL_ARGS = {
    "market_bars": {"limit"},
    "market_indicators": set(),
    "financial_evidence": {"limit", "source_type"},
    "xbrl_fundamentals": {"metrics", "period"},
    "rag_search": {"query", "limit"},
}
TOOL_SCHEMAS = {**SPECS, "financial_evidence": FINANCIAL_SPEC,
                "xbrl_fundamentals": FUNDAMENTALS_SPEC, "rag_search": RAG_SPEC}


def _day(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value[:10])


def _number(value):
    return type(value) in {int, float} and math.isfinite(value)


@dataclass(frozen=True)
class Profile:
    anonymous: bool
    external_evidence: bool


PROFILES = {"P0": Profile(False, False), "P1": Profile(False, True),
            "P2": Profile(True, False), "P3": Profile(True, True)}


class Projection:
    """Bind to a private case identity and keep all original metadata on the backend."""

    def __init__(self, *, secret: bytes, case: str, ticker: str, decision: date, close: float):
        if len(secret) < 16 or not _number(close) or close <= 0:
            raise ValueError("Projection requires a private key and positive anchor close")
        self._key = hmac.digest(secret, case.encode(), "sha256")
        self._ticker = ticker
        self._decision = decision
        self._price_scale = 100 / close
        self._money_scale = .5 + int.from_bytes(self._key[:4], "big") / 2**32
        self._labels = {}

    def _alias(self, kind, value):
        digest = hmac.new(self._key, f"{kind}:{value}".encode(), sha256).hexdigest()[:24]
        return f"{kind}_{digest}"

    @property
    def entity(self):
        return self._alias("entity", self._ticker)

    def _evidence_id(self, real):
        if not isinstance(real, str) or not real:
            return None
        alias = self._alias("ref", real)
        previous = self._labels.setdefault(alias, real)
        if previous != real:
            raise ValueError("Citation alias collision")
        return alias

    def resolve_citations(self, aliases):
        if any(key not in self._labels for key in aliases):
            raise ValueError("Unknown evidence alias")
        return [self._labels[key] for key in aliases]

    def record(self, raw, *, require_availability=False):
        """Accept only recognized typed fields. Free prose and transport metadata stay local."""
        cutoff = close_available(self._decision)
        visibility = [raw[key] for key in ("available_at", "eligible_at") if raw.get(key)]
        try:
            if require_availability and not visibility:
                return {}
            for stamp in visibility:
                when = stamp if isinstance(stamp, datetime) else datetime.fromisoformat(stamp)
                if when.tzinfo is None or when > cutoff:
                    return {}
            for key in DATES:
                if raw.get(key) and _day(raw[key]) > self._decision:
                    return {}
        except (ValueError, TypeError, AttributeError):
            return {}
        result = {}
        for key, value in raw.items():
            if key in DATES and value:
                result[f"{key}_offset"] = (_day(value) - self._decision).days
            elif key in PRICE and _number(value):
                result[key] = value * self._price_scale
            elif key in RATIOS and _number(value):
                result[key] = value
            elif key == "volume" and _number(value):
                result[key] = value * (self._money_scale / self._price_scale)
            elif key in ENUMS and isinstance(value, str) and value in ENUMS[key]:
                result[key] = value
            elif key == "evidence_id" and isinstance(value, str) and value:
                result[key] = self._evidence_id(value)
            elif key == "evidence_ids" and isinstance(value, list):
                aliases_by_original = {original: alias for alias, original in self._labels.items()}
                result[key] = [item if item in self._labels else aliases_by_original[item]
                               for item in value if isinstance(item, str)
                               and (item in self._labels or item in aliases_by_original)]
        if raw.get("metric") in METRICS and _number(raw.get("value")):
            unit = raw.get("unit")
            if isinstance(unit, str) and unit in {"USD", "USD/shares", "ratio", "shares"}:
                value, unit = self._metric(raw["value"], unit)
                result.update(metric=raw["metric"], value=value, unit=unit)
        if "bars" in raw and isinstance(raw["bars"], list):
            result["bars"] = [record for item in raw["bars"] if isinstance(item, dict)
                              if (record := self.record(item))]
        if "items" in raw and isinstance(raw["items"], list):
            result["items"] = [record for item in raw["items"] if isinstance(item, dict)
                               if (record := self.record(item, require_availability=True))]
        return result

    def _metric(self, value, unit):
        return {
            "USD": (value * self._money_scale, "currency_units"),
            "USD/shares": (value * self._price_scale, "currency_units/adjusted_share"),
            "ratio": (value, "ratio"),
            "shares": (value * self._money_scale / self._price_scale, "adjusted_shares"),
        }[unit]

    def tool_arguments(self, tool, args):
        """Allow bounded intent arguments. The backend supplies actual ticker and cutoff."""
        if tool not in TOOL_ARGS or not isinstance(args, dict) or args.keys() - TOOL_ARGS[tool]:
            raise ValueError("Unsupported anonymous tool arguments")
        for name, value in args.items():
            if name == "query" and (not isinstance(value, str) or value not in TOPICS):
                raise ValueError("Expected a controlled retrieval topic")
        try:
            return TOOL_SCHEMAS[tool][0].model_validate(args).model_dump(mode="json")
        except ValueError:
            raise ValueError("Invalid anonymous tool arguments") from None

    def tool_result(self, tool, result):
        if tool not in TOOL_ARGS or not isinstance(result, dict) or result.get("error"):
            return {"status": "unavailable"}
        return self.record(result) or {"status": "unavailable"}


def model_context(profile, *, ticker, company, decision, market, external=(), rag=(), memory=(), agents=(),
                  projection=None):
    """Build a fresh context for an explicit P0–P3 arm, without mixing evidence arms."""
    arm = PROFILES[profile]
    if arm.anonymous:
        if projection is None:
            raise ValueError("Anonymous profiles require a case projection")
        result = {"entity": projection.entity, "decision_offset": 0,
                  "market": projection.record(market)}
        if arm.external_evidence:
            for name, records in (("evidence", external), ("rag", rag), ("memory", memory)):
                result[name] = [projected for item in records
                                if (projected := projection.record(item, require_availability=True))]
        result["agents"] = [projection.record(item) for item in agents]
        return result
    result = {"ticker": ticker, "company": company, "decision": str(decision), "market": market,
              "agents": list(agents)}
    if arm.external_evidence:
        result.update(evidence=list(external), rag=list(rag), memory=list(memory))
    return result
