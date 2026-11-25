"""Task-scoped finance tools shared by in-process calls and the MCP adapter."""
from dataclasses import dataclass

from pydantic import Field, ValidationError

from .evidence import SourceType
from .schema import Contract


class BarsArgs(Contract):
    limit: int = Field(default=10, ge=1, le=20, strict=True)


class EmptyArgs(Contract):
    pass


class FinancialArgs(Contract):
    source_type: SourceType
    limit: int = Field(default=3, ge=1, le=4, strict=True)


class RagArgs(Contract):
    query: str = Field(min_length=1, max_length=400)
    limit: int = Field(default=4, ge=1, le=10, strict=True)


SPECS = {
    "market_indicators": (EmptyArgs, "Get past-only daily momentum, moving average and volatility."),
    "market_bars": (BarsArgs, "Get up to 20 daily bars known at the decision cutoff."),
}
FINANCIAL_SPEC = (FinancialArgs,
    "Retrieve dated financial evidence for this task only; returns exact citation IDs and PIT levels.")
RAG_SPEC = (RagArgs, "Search task-bound filings and news for relevant passages. Cite returned chunk IDs.")


@dataclass
class ToolRegistry:
    view: object
    financial: object = None
    rag: object = None

    @property
    def definitions(self):
        return {**SPECS,
                **({"financial_evidence": FINANCIAL_SPEC} if self.financial is not None else {}),
                **({"rag_search": RAG_SPEC} if self.rag is not None else {})}

    def execute(self, name: str, raw: str | dict) -> dict:
        spec = self.definitions.get(name)
        if spec is None:
            return {"error": "unknown_tool"}
        try:
            if isinstance(raw, dict):
                args = spec[0].model_validate(raw)
            else:
                if not isinstance(raw, str) or len(raw.encode()) > 4096:
                    return {"error": "invalid_arguments"}
                args = spec[0].model_validate_json(raw)
        except (TypeError, ValidationError):
            return {"error": "invalid_arguments"}
        try:
            if name == "financial_evidence":
                return self.financial.search(args.source_type, args.limit)
            if name == "rag_search":
                return self.rag.search(args.query, args.limit)
            if name == "market_bars":
                return self.view.get_bars(args.limit)
            return self.view.indicators()
        except TimeoutError:
            return {"error": "tool_timeout"}
        except Exception as exc:
            return {"error": "tool_execution_error", "error_type": type(exc).__name__}
