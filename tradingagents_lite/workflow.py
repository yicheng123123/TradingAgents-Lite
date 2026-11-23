"""Bounded analyst tools, separate Trader proposal, deterministic risk gate."""
import json
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError

from .data import MarketView
from .schema import Action, Budget, Contract, Decision, MarketReport, Proposal


class BarsArgs(Contract):
    limit: int = Field(default=10, ge=1, le=20, strict=True)


class EmptyArgs(Contract):
    pass


SPECS = {
    "market_indicators": (EmptyArgs, "Get past-only daily momentum, moving average and volatility."),
    "market_bars": (BarsArgs, "Get up to 20 daily bars known at the decision cutoff."),
}


def specs() -> list[dict]:
    return [{"type": "function", "function": {"name": name, "description": text,
             "parameters": model.model_json_schema()}} for name, (model, text) in SPECS.items()]


def call_message(name: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "offline-1", "type": "function", "function": {"name": name, "arguments": "{}"}}]}


class Backend(Protocol):
    usage: dict | None

    def analyst(self, messages: list[dict], tools: list[dict]) -> dict: ...

    def trader(self, report: MarketReport, evidence: dict) -> Proposal: ...


class MomentumBackend:
    """Actual offline trading rule, with no LLM calls or simulated model scores."""

    usage = None

    def __init__(self, budget: Budget):
        self.budget = budget

    def analyst(self, messages: list[dict], tools: list[dict]) -> dict:
        observed = None
        for message in reversed(messages[2:]):
            if message["role"] not in {"tool", "user"}:
                continue
            try:
                payload = json.loads(message["content"])
            except ValueError:
                continue
            if isinstance(payload, dict) and "return_5" in payload:
                observed = payload
                break
        if observed is None:
            return call_message("market_indicators") if tools else {"content": ""}
        report = MarketReport(view=f"Five-session momentum = {observed['return_5']:.6f}.",
                              evidence_ids=[observed["evidence_id"]])
        return {"content": report.model_dump_json()}

    def trader(self, report: MarketReport, evidence: dict) -> Proposal:
        momentum = evidence[report.evidence_ids[0]]["return_5"]
        threshold = self.budget.momentum_threshold
        action = Action.BUY if momentum > threshold else Action.SELL if momentum < -threshold else Action.HOLD
        return Proposal(action=action,
                        rationale="Compare five-session close momentum with the fixed two-sided threshold.",
                        evidence_ids=report.evidence_ids)


class State(TypedDict, total=False):
    messages: list[dict]
    pending: list[dict]
    evidence: dict
    tool_trace: list[dict]
    rounds: int
    report: dict
    proposal: dict
    decision: dict
    errors: list[dict]


def build_workflow(view: MarketView, backend: Backend, budget: Budget, react: bool = True):
    """Bind tools to a prefiltered single-task view, never to the outcome provider."""
    def execute(name: str, raw: str) -> dict:
        if name not in SPECS:
            return {"error": "unknown_tool"}
        try:
            args = SPECS[name][0].model_validate_json(raw)
        except (TypeError, ValidationError):
            return {"error": "invalid_arguments"}
        return view.indicators() if name == "market_indicators" else view.get_bars(args.limit)

    def start(_: State) -> dict:
        instructions = (
            "Market Analyst for a historical research task. Use supplied tools and only the observed evidence. "
            "On your final turn emit precisely a JSON MarketReport. Cite exact evidence_id values. "
            f"MarketReport schema: {json.dumps(MarketReport.model_json_schema())}"
        )
        messages = [{"role": "system", "content": instructions}, {"role": "user", "content":
            f"Analyze {view.task.symbol} after the {view.task.decision_date} session. "
            f"Target horizon: {view.task.horizon_sessions} sessions."}]
        evidence, trace = {}, []
        if not react:
            result = execute("market_indicators", "{}")
            evidence[result["evidence_id"]] = result
            trace.append({"name": "market_indicators", "status": "ok", "dispatch": "prefetch"})
            messages.append({"role": "user", "content": json.dumps(result)})
        return {"messages": messages, "evidence": evidence, "rounds": 0, "pending": [],
                "errors": [], "tool_trace": trace}

    def analyst(state: State) -> dict:
        allowed = react and state["rounds"] < budget.max_tool_rounds
        messages = state["messages"]
        if not allowed:
            messages = [*messages, {"role": "user", "content": "Return the final JSON MarketReport now."}]
        try:
            reply = backend.analyst(messages, specs() if allowed else [])
            calls = reply.get("tool_calls") or []
            if calls:
                if (not allowed or len(calls) > 2
                        or len({c["id"] for c in calls}) != len(calls)
                        or any(not isinstance(c["function"]["name"], str)
                               or not isinstance(c["function"]["arguments"], str) for c in calls)):
                    return {"pending": [], "errors": [*state["errors"], {"stage": "analyst", "code": "tool_budget"}]}
                return {"messages": [*state["messages"], reply], "pending": calls,
                        "rounds": state["rounds"] + 1}
            report = MarketReport.model_validate_json(reply.get("content") or "")
            if not set(report.evidence_ids) <= state["evidence"].keys():
                raise ValueError("Unobserved evidence IDs")
            return {"report": report.model_dump(mode="json"), "pending": []}
        except Exception as exc:
            return {"pending": [], "errors": [*state["errors"], {"stage": "analyst", "code": type(exc).__name__}]}

    def tools(state: State) -> dict:
        messages, evidence = list(state["messages"]), dict(state["evidence"])
        trace = list(state["tool_trace"])
        for call in state["pending"]:
            fn = call["function"]
            result = execute(fn["name"], fn["arguments"])
            error = result.get("error")
            if not error:
                evidence[result["evidence_id"]] = result
            trace.append({"name": fn["name"] if fn["name"] in SPECS else "unregistered",
                          "status": "error" if error else "ok", "code": error,
                          "evidence_id": result.get("evidence_id"), "dispatch": "react"})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
        return {"messages": messages, "evidence": evidence, "tool_trace": trace, "pending": []}

    def trader(state: State) -> dict:
        if "report" not in state:
            return {}
        try:
            report = MarketReport.model_validate(state["report"])
            evidence = {i: state["evidence"][i] for i in report.evidence_ids}
            proposal = backend.trader(report, evidence)
            if (not set(proposal.evidence_ids) <= evidence.keys()
                    or (proposal.action != Action.HOLD and not proposal.evidence_ids)):
                raise ValueError("A directional proposal requires observed evidence")
            return {"proposal": proposal.model_dump(mode="json")}
        except Exception as exc:
            return {"errors": [*state["errors"], {"stage": "trader", "code": type(exc).__name__}]}

    def risk(state: State) -> dict:
        if "proposal" not in state:
            decision = Decision(action=Action.HOLD, rationale="No valid evidence-backed proposal.",
                                evidence_ids=[], risk_veto="workflow_failure")
        else:
            proposal = Proposal.model_validate(state["proposal"])
            too_volatile = view.indicators()["daily_volatility_20"] > budget.volatility_limit
            if proposal.action != Action.HOLD and too_volatile:
                decision = Decision(action=Action.HOLD, rationale="Volatility exceeds the rule limit.",
                                    evidence_ids=proposal.evidence_ids, risk_veto="volatility_limit")
            else:
                decision = Decision(**proposal.model_dump())
        return {"decision": decision.model_dump(mode="json")}

    graph = StateGraph(State)
    for name, fn in {"prepare": start, "analyst": analyst, "tools": tools, "trader": trader, "risk": risk}.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "analyst")
    graph.add_conditional_edges("analyst", lambda s: "tools" if s["pending"] else "trader",
                                {"tools": "tools", "trader": "trader"})
    graph.add_edge("tools", "analyst")
    graph.add_edge("trader", "risk")
    graph.add_edge("risk", END)
    return graph.compile()
