"""Data and decision contracts shared by the graph and backtest."""
from datetime import date
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PositiveFloat, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Task(Contract):
    task_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")
    symbol: str = Field(pattern=r"^[A-Z][A-Z0-9.-]{0,9}$")
    decision_date: date
    horizon_sessions: int = Field(ge=1, le=20)


class Bar(Contract):
    day: date
    available_at: AwareDatetime
    open: PositiveFloat
    high: PositiveFloat
    low: PositiveFloat
    close: PositiveFloat
    volume: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("Inconsistent OHLC bounds")
        return self


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

    @property
    def direction(self) -> int:
        return {Action.BUY: 1, Action.SELL: -1, Action.HOLD: 0}[self]


class MarketReport(Contract):
    view: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=4)


class Proposal(Contract):
    action: Action
    rationale: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(max_length=4)


class Decision(Contract):
    action: Action
    rationale: str
    evidence_ids: list[str]
    risk_veto: str | None = None


class Budget(Contract):
    max_tool_rounds: int = Field(default=2, ge=1, le=4)
    max_tokens: int = Field(default=1200, ge=64, le=4096)
    momentum_threshold: float = Field(default=0.006, ge=0, le=1)
    volatility_limit: float = Field(default=0.04, gt=0, le=1)


class Outcome(Contract):
    entry_day: date
    exit_day: date
    available_at: AwareDatetime
    asset_return: float
    benchmark_return: float
