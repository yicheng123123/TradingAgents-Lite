"""Bounded analyst tools, separate Trader proposal, deterministic risk gate."""
import json
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError

from .data import MarketView
from .evidence import EvidenceView, SourceType
from .prompts import ANALYST_SYSTEM_PROMPT
from .provider import ModelCallError
from .schema import Action, Budget, Contract, Decision, MarketReport, Proposal


class BarsArgs(Contract):
    limit: int = Field(default=10, ge=1, le=20, strict=True)


class EmptyArgs(Contract):
    pass


SPECS = {
    "market_indicators": (EmptyArgs, "Get past-only daily momentum, moving average and volatility."),
    "market_bars": (BarsArgs, "Get up to 20 daily bars known at the decision cutoff."),
}


class FinancialArgs(Contract):
    source_type: SourceType
    limit: int = Field(default=3, ge=1, le=4, strict=True)


FINANCIAL_SPEC = (FinancialArgs,
    "Retrieve dated financial evidence for this task only; returns exact citation IDs and PIT levels.")


class RagArgs(Contract):
    query: str = Field(min_length=1, max_length=400)
    limit: int = Field(default=4, ge=1, le=10, strict=True)


RAG_SPEC = (RagArgs, "Search task-bound filings and news for relevant passages. Cite returned chunk IDs.")


def specs(registry: dict = SPECS) -> list[dict]:
    return [{"type": "function", "function": {"name": name, "description": text,
            "parameters": model.model_json_schema()}} for name, (model, text) in registry.items()]


def call_message(name: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "offline-1", "type": "function", "function": {"name": name, "arguments": "{}"}}]}


class Backend(Protocol):
    usage: dict | None
    call_log: list[dict]
    def analyst(self, messages: list[dict], tools: list[dict]) -> dict: ...
    def trader(self, report: MarketReport, evidence: dict) -> Proposal: ...


class MomentumBackend:
    """Actual offline trading rule, with no LLM calls or simulated model scores."""
    usage = None

    def __init__(self, budget: Budget):
        self.budget = budget
        self.call_log = []

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
    structured_output_trace: list[dict]
    citation_trace: list[dict]


def _citation_event(stage: str, cited_ids: list[str], evidence: dict) -> dict:
    invalid = [item for item in cited_ids if item not in evidence]
    financial = [item for item in cited_ids
                 if item in evidence and evidence[item].get("source_type") in {"sec_filing", "news"}]
    return {"stage": stage, "cited_ids": cited_ids,
            "valid_ids": [item for item in cited_ids if item in evidence],
            "invalid_ids": invalid, "financial_ids": financial}


def _structured_event(stage: str, valid: bool, error_type: str | None = None) -> dict:
    event = {"stage": stage, "valid": valid}
    if error_type is not None:
        event["error_type"] = error_type
    return event


def _llm_tool_result(result: dict) -> dict:
    """Keep only model-useful fields in repeated conversation context."""
    if "retrieval_trace" in result:
        return {key: result[key] for key in ("status", "items")}
    if "bars" in result:
        return {
            "evidence_id": result.get("evidence_id"),
            "bars": [
                {key: bar[key] for key in ("day", "open", "high", "low", "close", "volume") if key in bar}
                for bar in result["bars"]
            ],
        }
    if "items" in result:
        return {
            "status": result.get("status"),
            "source_type": result.get("source_type"),
            "reason": result.get("reason"),
            "items": [
                {
                    key: item[key]
                    for key in ("evidence_id", "source_type", "published_at", "eligible_at", "pit_level")
                    if key in item
                }
                | {
                    "title": item.get("source_metadata", {}).get("title"),
                    "publisher": item.get("source_metadata", {}).get("publisher"),
                    "excerpt": (item.get("excerpt") or "")[:800],
                }
                for item in result["items"]
            ],
        }
    return result


def _tool_content(result: dict) -> str:
    compact = (",", ":") if "retrieval_trace" in result else None
    return json.dumps(_llm_tool_result(result), ensure_ascii=False, separators=compact)


def build_workflow(view: MarketView, backend: Backend, budget: Budget, react: bool = True,
                   financial: EvidenceView | None = None, *, rag=None):
    """Bind tools to a prefiltered single-task view, never to the outcome provider.

    The invariant analyst prompt is identical in B1-M and B1-E. B1-E differs only
    in the additional, cutoff-bound financial_evidence tool and its observed output.
    """
    registry = dict(SPECS)
    if financial is not None:
        registry["financial_evidence"] = FINANCIAL_SPEC
    if rag is not None:
        registry["rag_search"] = RAG_SPEC

    def execute(name: str, raw: str) -> dict:
        if name not in registry:
            return {"error": "unknown_tool"}
        try:
            args = registry[name][0].model_validate_json(raw)
        except (TypeError, ValidationError):
            return {"error": "invalid_arguments"}
        try:
            if name == "financial_evidence":
                return financial.search(args.source_type, args.limit)
            if name == "rag_search":
                return rag.search(args.query, args.limit)
            return view.indicators() if name == "market_indicators" else view.get_bars(args.limit)
        except TimeoutError:
            return {"error": "tool_timeout"}
        except Exception as exc:
            return {"error": "tool_execution_error", "error_type": type(exc).__name__}

    def start(_: State) -> dict:
        messages = [{"role": "system", "content": ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze {view.task.symbol} after the {view.task.decision_date} session. "
             f"Target horizon: {view.task.horizon_sessions} sessions."}]
        evidence, trace = {}, []
        if rag is not None and rag.config.dispatch == "fixed":
            result = execute("rag_search", json.dumps({"query": "Recent financial results, risks and company outlook"}))
            evidence.update({item["evidence_id"]: item for item in result.get("items", [])})
            trace.append({"name": "rag_search", "dispatch": "fixed",
                          "status": "error" if "error" in result else "ok",
                          "code": result.get("error"), "retrieval_trace": result.get("retrieval_trace")})
            messages.append({"role": "user", "content": _tool_content(result)})
        if not react:
            result = execute("market_indicators", "{}")
            if "error" not in result:
                evidence[result["evidence_id"]] = result
            trace.append({"name": "market_indicators", "status": "error" if "error" in result else "ok",
                          "dispatch": "prefetch", "code": result.get("error")})
            messages.append({"role": "user", "content": json.dumps(result)})
        return {"messages": messages, "evidence": evidence, "rounds": 0, "pending": [],
                "errors": [], "tool_trace": trace, "structured_output_trace": [], "citation_trace": []}

    def analyst(state: State) -> dict:
        allowed = react and state["rounds"] < budget.max_tool_rounds
        messages = state["messages"]
        if not allowed:
            messages = [*messages, {"role": "user", "content": "Return the final JSON MarketReport now."}]
        try:
            active = list(registry)
            if rag is not None and rag.config.dispatch == "fixed":
                active.remove("rag_search")
            reply = backend.analyst(messages, specs({name: registry[name] for name in active}) if allowed else [])
        except ModelCallError as exc:
            return {"pending": [], "errors": [*state["errors"],
                {"stage": "model", "agent_stage": "analyst", "code": exc.code, "attempts": exc.attempts}]}
        except Exception as exc:
            return {"pending": [], "errors": [*state["errors"],
                {"stage": "analyst", "code": type(exc).__name__}]}

        try:
            calls = reply.get("tool_calls") or []
            if calls:
                if (not allowed or len(calls) > 2 or len({c["id"] for c in calls}) != len(calls)
                        or any(not isinstance(c["function"]["name"], str)
                               or not isinstance(c["function"]["arguments"], str) for c in calls)):
                    return {"pending": [], "errors": [*state["errors"],
                        {"stage": "analyst", "code": "tool_budget"}]}
                return {"messages": [*state["messages"], reply], "pending": calls,
                        "rounds": state["rounds"] + 1}
            report = MarketReport.model_validate_json(reply.get("content") or "")
        except Exception as exc:
            trace = [*state.get("structured_output_trace", []),
                     _structured_event("market_report", False, type(exc).__name__)]
            return {"pending": [], "structured_output_trace": trace,
                    "errors": [*state["errors"], {"stage": "analyst", "code": type(exc).__name__}]}

        output_trace = [*state.get("structured_output_trace", []), _structured_event("market_report", True)]
        citation = _citation_event("market_report", report.evidence_ids, state["evidence"])
        citation_trace = [*state.get("citation_trace", []), citation]
        if citation["invalid_ids"]:
            return {"pending": [], "structured_output_trace": output_trace, "citation_trace": citation_trace,
                    "errors": [*state["errors"], {"stage": "analyst", "code": "invalid_citation"}]}
        return {"report": report.model_dump(mode="json"), "pending": [],
                "structured_output_trace": output_trace, "citation_trace": citation_trace}

    def tools(state: State) -> dict:
        messages, evidence = list(state["messages"]), dict(state["evidence"])
        trace = list(state["tool_trace"])
        for call in state["pending"]:
            fn = call["function"]
            if fn["name"] == "rag_search" and rag is not None and rag.config.dispatch == "fixed":
                result = {"error": "unknown_tool"}
            else:
                result = execute(fn["name"], fn["arguments"])
            error = result.get("error")
            if "items" in result:
                evidence.update({item["evidence_id"]: item for item in result["items"]})
            elif not error:
                evidence[result["evidence_id"]] = result
            try:
                arg_record = registry[fn["name"]][0].model_validate_json(fn["arguments"]).model_dump(mode="json")
            except (KeyError, TypeError, ValidationError):
                arg_record = None
            trace.append({"name": fn["name"] if fn["name"] in registry else "unregistered",
                "status": "error" if error else "ok", "code": error, "arguments": arg_record,
                "evidence_id": result.get("evidence_id"), "dispatch": "react",
                **({"retrieval_trace": result["retrieval_trace"]} if "retrieval_trace" in result else {}),
                **({"availability": result["status"],
                    "evidence_ids": [item["evidence_id"] for item in result["items"]]}
                   if "items" in result else {})})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": _tool_content(result)})
        return {"messages": messages, "evidence": evidence, "tool_trace": trace, "pending": []}

    def trader(state: State) -> dict:
        if "report" not in state:
            return {}
        try:
            report = MarketReport.model_validate(state["report"])
            evidence = {i: state["evidence"][i] for i in report.evidence_ids}
            proposal = Proposal.model_validate(backend.trader(report, evidence))
        except ModelCallError as exc:
            return {"errors": [*state["errors"],
                {"stage": "model", "agent_stage": "trader", "code": exc.code, "attempts": exc.attempts}]}
        except Exception as exc:
            return {"structured_output_trace": [*state.get("structured_output_trace", []),
                        _structured_event("proposal", False, type(exc).__name__)],
                    "errors": [*state["errors"], {"stage": "trader", "code": type(exc).__name__}]}

        output_trace = [*state.get("structured_output_trace", []), _structured_event("proposal", True)]
        citation = _citation_event("proposal", proposal.evidence_ids, evidence)
        citation_trace = [*state.get("citation_trace", []), citation]
        if citation["invalid_ids"] or (proposal.action != Action.HOLD and not proposal.evidence_ids):
            return {"structured_output_trace": output_trace, "citation_trace": citation_trace,
                    "errors": [*state["errors"], {"stage": "trader", "code": "invalid_citation"}]}
        return {"proposal": proposal.model_dump(mode="json"),
                "structured_output_trace": output_trace, "citation_trace": citation_trace}

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
