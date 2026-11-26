"""Bounded analyst tools, separate Trader proposal, deterministic risk gate."""
import json
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .data import MarketView
from .evidence import EvidenceView
from .prompts import (ANALYST_SYSTEM_PROMPT, ANALYST_SYSTEM_PROMPT_V2, ANALYST_SYSTEM_PROMPT_V3,
                      FINAL_JSON_V3, TRADER_SYSTEM_PROMPT_V3)
from .provider import ModelCallError
from .schema import Action, Budget, Decision, MarketReport, Proposal
from .structured_output import StructuredOutput


from .tool_registry import FINANCIAL_SPEC, RAG_SPEC, SPECS, ToolRegistry


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
    context_ids: list[str]
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


def _llm_tool_result(result: dict, *, compact: bool = False) -> dict:
    """Keep only model-useful fields in repeated conversation context."""
    if compact:
        if result.get("source_type") == "sec_xbrl":
            fields = ("metric", "value", "unit", "period", "period_end", "evidence_id", "eligible_at", "pit_level")
            return {"source_type": "sec_xbrl", "status": result.get("status"),
                    "items": [{key: item[key] for key in fields if key in item}
                              for item in result.get("items", [])[:4]]}
        if "bars" in result:
            bars = result.get("bars", [])
            return {"evidence_id": result.get("evidence_id"),
                    "bars": [{key: bar[key] for key in ("day", "open", "high", "low", "close", "volume")
                              if key in bar} for bar in bars[-8:]]}
        if "items" in result:
            items = []
            for item in result.get("items", [])[:4]:
                picked = {key: item[key] for key in
                          ("evidence_id", "source_type", "published_at", "eligible_at", "pit_level",
                           "metric", "value", "unit", "period", "period_end") if key in item}
                metadata = item.get("source_metadata", {})
                if metadata.get("title"):
                    picked["title"] = metadata["title"][:160]
                if metadata.get("publisher"):
                    picked["publisher"] = metadata["publisher"][:80]
                excerpt = item.get("excerpt")
                if excerpt:
                    picked["excerpt"] = excerpt[:320]
                items.append(picked)
            return {key: result[key] for key in ("status", "source_type", "reason") if key in result} | {
                "items": items}
        return {key: result[key] for key in
                ("status", "error", "reason", "evidence_id", "date", "close", "return_5", "sma_20",
                 "daily_volatility_20", "value", "unit", "period", "eligible_at", "pit_level")
                if key in result}
    if result.get("source_type") == "sec_xbrl":
        fields = ("metric", "value", "unit", "period", "period_end", "evidence_id", "eligible_at", "pit_level")
        return {**result, "items": [{key: item[key] for key in fields} for item in result["items"]]}
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


def _tool_content(result: dict, *, compact: bool = False) -> str:
    separators = (",", ":") if compact or "retrieval_trace" in result else None
    return json.dumps(_llm_tool_result(result, compact=compact), ensure_ascii=False, separators=separators)


def _valid_tool_calls(calls: list[dict], *, allowed: bool, maximum: int) -> bool:
    if not allowed or not isinstance(calls, list) or not calls or len(calls) > maximum:
        return False
    identifiers = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("id"), str):
            return False
        function = call.get("function")
        if (not isinstance(function, dict) or not isinstance(function.get("name"), str)
                or not isinstance(function.get("arguments"), str)):
            return False
        identifiers.append(call["id"])
    return len(set(identifiers)) == len(identifiers)


def build_workflow(view: MarketView, backend: Backend, budget: Budget, react: bool = True,
                   financial: EvidenceView | None = None, *, rag=None, memory=None, a3=None, fundamentals=None,
                   max_tool_calls_per_round: int = 2, compact_tool_results: bool = False,
                   flash_v2_prompt: bool = False, flash_v3: bool = False):
    """Bind tools to a prefiltered single-task view, never to the outcome provider.

    The invariant analyst prompt is identical in B1-M and B1-E. B1-E differs only
    in the additional, cutoff-bound financial_evidence tool and its observed output.
    """
    if not 1 <= max_tool_calls_per_round <= 4:
        raise ValueError("At most four tool calls per round are supported")
    structured = StructuredOutput(backend) if flash_v3 else None
    tools_bound = ToolRegistry(view, financial, rag, fundamentals)
    registry = tools_bound.definitions
    if a3 is not None:
        registry.update(a3.extra_definitions)

    fixed_rag = ((rag is not None and rag.config.dispatch == "fixed")
                 or (a3 is not None and a3.fixed_rag))

    def execute(name: str, raw: str, call_id=None) -> dict:
        if a3 is not None:
            return a3.tool(name, raw, tools_bound.execute, call_id)
        return tools_bound.execute(name, raw)

    def compact_result(result: dict) -> dict:
        return _llm_tool_result(result, compact=compact_tool_results)

    def tool_content(result: dict) -> str:
        return _tool_content(result, compact=compact_tool_results)

    def start(_: State) -> dict:
        prompt = (ANALYST_SYSTEM_PROMPT_V3 if flash_v3 else
                  ANALYST_SYSTEM_PROMPT_V2 if flash_v2_prompt else ANALYST_SYSTEM_PROMPT)
        messages = [{"role": "system", "content": prompt},
            {"role": "user", "content": f"Analyze {view.task.symbol} after the {view.task.decision_date} session. "
             f"Target horizon: {view.task.horizon_sessions} sessions."}]
        evidence, trace = {}, []
        context_ids = set()
        if "xbrl_fundamentals" in registry:
            result = execute("xbrl_fundamentals", "{}")
            evidence.update({item["evidence_id"]: item for item in result.get("items", [])})
            context_ids.update(evidence)
            trace.append({"name": "xbrl_fundamentals", "dispatch": "prefetch",
                          "status": "error" if "error" in result else "ok",
                          "code": result.get("error"), "availability": result.get("status"),
                          "evidence_ids": list(evidence)})
            messages.append({"role": "user", "content": tool_content(result)})
        if fixed_rag:
            result = execute("rag_search", json.dumps({"query": "Recent financial results, risks and company outlook"}))
            evidence.update({item["evidence_id"]: item for item in result.get("items", [])})
            trace.append({"name": "rag_search", "dispatch": "fixed",
                          "status": "error" if "error" in result else "ok",
                          "code": result.get("error"), "retrieval_trace": result.get("retrieval_trace")})
            messages.append({"role": "user", "content": tool_content(result)})
            context_ids.update(item["evidence_id"] for item in compact_result(result).get("items", []))
        if not react:
            result = execute("market_indicators", "{}")
            if "error" not in result:
                evidence[result["evidence_id"]] = result
                context_ids.add(result["evidence_id"])
            trace.append({"name": "market_indicators", "status": "error" if "error" in result else "ok",
                          "dispatch": "prefetch", "code": result.get("error")})
            messages.append({"role": "user", "content": tool_content(result)})
        return {"messages": messages, "evidence": evidence, "rounds": 0, "pending": [],
                "errors": [], "tool_trace": trace, "structured_output_trace": [], "citation_trace": [],
                **({"context_ids": sorted(context_ids)} if a3 is not None or flash_v3 else {})}

    def structured_result(state, stage, messages, evidence, reply=None):
        agent, key = ("analyst", "report") if stage == "market_report" else ("trader", "proposal")
        try:
            value, events, fault = structured.generate(stage, messages, evidence, reply)
        except ModelCallError as exc:
            value, events, fault = None, [], exc
        result = {"structured_output_trace": [*state["structured_output_trace"], *events]}
        if fault:
            result["errors"] = [*state["errors"], {"stage": "model", "agent_stage": agent,
                                                   "code": fault.code, "attempts": fault.attempts}]
        elif value is None:
            result["errors"] = [*state["errors"], {"stage": agent, "code": events[-1]["errors"][0]["type"]}]
        else:
            result[key] = value.model_dump(mode="json")
            result["citation_trace"] = [*state["citation_trace"], _citation_event(stage, value.evidence_ids, evidence)]
        return result

    def analyst_v3(state: State) -> dict:
        messages = state["messages"]
        evidence = {key: state["evidence"][key] for key in state["context_ids"]}
        final = [*messages, {"role": "user", "content": FINAL_JSON_V3}]
        if react and state["rounds"] < budget.max_tool_rounds:
            try:
                reply = backend._call(messages, specs(registry), structured_stage="market_report")
            except ModelCallError as exc:
                return {"pending": [], "errors": [*state["errors"],
                    {"stage": "model", "agent_stage": "analyst", "code": exc.code, "attempts": exc.attempts}]}
            if reply.get("finish_reason") not in {None, "stop", "tool_calls"}:
                return {"pending": [], **structured_result(state, "market_report", final, evidence, reply)}
            calls = reply.get("tool_calls") or []
            if calls:
                if not _valid_tool_calls(calls, allowed=True, maximum=max_tool_calls_per_round):
                    return {"pending": [], "errors": [*state["errors"], {"stage": "analyst", "code": "tool_budget"}]}
                assistant = {key: reply[key] for key in ("role", "content", "tool_calls")}
                return {"messages": [*messages, assistant], "pending": calls, "rounds": state["rounds"] + 1}
        return {"pending": [], **structured_result(state, "market_report", final, evidence)}

    def analyst(state: State) -> dict:
        if flash_v3:
            return analyst_v3(state)
        allowed = react and state["rounds"] < budget.max_tool_rounds
        messages = state["messages"]
        if not allowed:
            messages = [*messages, {"role": "user", "content": "Return the final JSON MarketReport now."}]
        try:
            active = list(registry)
            if fixed_rag:
                active.remove("rag_search")
            tool_specs = specs({name: registry[name] for name in active}) if allowed else []
            if a3 is not None:
                from .agents import context_bytes
                a3.context_sizes.append(context_bytes({"messages": messages, "tools": tool_specs}))
            reply = backend.analyst(messages, tool_specs)
        except ModelCallError as exc:
            return {"pending": [], "errors": [*state["errors"],
                {"stage": "model", "agent_stage": "analyst", "code": exc.code, "attempts": exc.attempts}]}
        except Exception as exc:
            return {"pending": [], "errors": [*state["errors"],
                {"stage": "analyst", "code": type(exc).__name__}]}

        try:
            calls = reply.get("tool_calls") or []
            if calls:
                if not _valid_tool_calls(calls, allowed=allowed, maximum=max_tool_calls_per_round):
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
        context_ids = set(state.get("context_ids", []))
        for call in state["pending"]:
            fn = call["function"]
            if fn["name"] == "rag_search" and fixed_rag:
                result = {"error": "unknown_tool"}
            else:
                result = execute(fn["name"], fn["arguments"], call["id"])
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
                **({"call_id": call["id"]} if a3 is not None else {}),
                **({"retrieval_trace": result["retrieval_trace"]} if "retrieval_trace" in result else {}),
                **({"availability": result["status"],
                    "evidence_ids": [item["evidence_id"] for item in result["items"]]}
                   if "items" in result else {})})
            delivered = compact_result(result)
            context_ids.update(item["evidence_id"] for item in delivered.get("items", []))
            if delivered.get("evidence_id"):
                context_ids.add(delivered["evidence_id"])
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": tool_content(result)})
        if a3 is not None and a3.tool_fault and not a3.config.repair:
            return {"evidence": evidence, "tool_trace": trace, "pending": [], "a3_stop": True,
                    "errors": [*state["errors"], {"stage": "tools", "code": a3.tool_fault}]}
        return {"messages": messages, "evidence": evidence, "tool_trace": trace, "pending": [],
                **({"context_ids": sorted(context_ids)} if a3 is not None or flash_v3 else {})}

    def trader(state: State) -> dict:
        if "report" not in state:
            return {}
        if flash_v3:
            report = state["report"]
            evidence = {key: state["evidence"][key] for key in report["evidence_ids"]}
            compact = {key: (compact_result({"items": [item]})["items"][0]
                             if "excerpt" in item else compact_result(item))
                       for key, item in evidence.items()}
            context = {"report": report, "evidence": compact}
            messages = [{"role": "system", "content": TRADER_SYSTEM_PROMPT_V3},
                        {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"))}]
            return structured_result(state, "proposal", messages, evidence)
        memory_state = {}
        try:
            report = MarketReport.model_validate(state["report"])
            evidence = {i: state["evidence"][i] for i in report.evidence_ids}
            historical = []
            if memory is not None:
                query = f"{view.task.symbol} {report.view[:600]}"
                try:
                    observation = memory.before(view.task, query)
                except Exception as exc:
                    if a3 is None or not a3.config.repair:
                        return {"memory_trace": {"error_type": type(exc).__name__},
                                "errors": [*state["errors"], {"stage": "memory", "code": "recall_error"}]}
                    from time import perf_counter
                    a3.event("memory", perf_counter(), error_code="recall_error", recovery_status="degraded")
                    observation = {"context": [], "error_type": type(exc).__name__,
                                   "recovery_status": "degraded_without_memory"}
                historical = observation["context"]
                memory_state = {"memory_trace": {**observation, "passed_to_backend": bool(
                    historical and hasattr(backend, "trader_with_memory"))}}
            if a3 is not None:
                from .agents import context_bytes
                a3.context_sizes.append(context_bytes([report.model_dump(), evidence, historical,
                                                       state.get("debate", {})]))
            if a3 is not None and hasattr(backend, "trader_a3"):
                raw = backend.trader_a3(report, evidence, historical, state.get("debate", {}),
                                        state.get("a3_repair_hint"))
            elif historical and hasattr(backend, "trader_with_memory"):
                raw = backend.trader_with_memory(report, evidence, historical)
            else:
                raw = backend.trader(report, evidence)
            proposal = Proposal.model_validate(raw)
        except ModelCallError as exc:
            return {**memory_state, "errors": [*state["errors"],
                {"stage": "model", "agent_stage": "trader", "code": exc.code, "attempts": exc.attempts}]}
        except Exception as exc:
            return {**memory_state, "structured_output_trace": [*state.get("structured_output_trace", []),
                        _structured_event("proposal", False, type(exc).__name__)],
                    "errors": [*state["errors"], {"stage": "trader", "code": type(exc).__name__}]}

        output_trace = [*state.get("structured_output_trace", []), _structured_event("proposal", True)]
        citation = _citation_event("proposal", proposal.evidence_ids, evidence)
        citation_trace = [*state.get("citation_trace", []), citation]
        if citation["invalid_ids"] or (proposal.action != Action.HOLD and not proposal.evidence_ids):
            return {**memory_state, "structured_output_trace": output_trace, "citation_trace": citation_trace,
                    "errors": [*state["errors"], {"stage": "trader", "code": "invalid_citation"}]}
        return {**memory_state, "proposal": proposal.model_dump(mode="json"),
                "structured_output_trace": output_trace, "citation_trace": citation_trace,
                **({"a3_verify_retry": False} if a3 is not None else {})}

    def risk(state: State) -> dict:
        if not state.get("proposal") or (a3 is not None and state.get("errors")):
            decision = Decision(action=Action.HOLD, rationale="No valid evidence-backed proposal.",
                                evidence_ids=[], risk_veto=(state["errors"][-1]["code"]
                                    if a3 is not None and state.get("errors") else "workflow_failure"))
        else:
            proposal = Proposal.model_validate(state["proposal"])
            too_volatile = view.indicators()["daily_volatility_20"] > budget.volatility_limit
            if proposal.action != Action.HOLD and too_volatile:
                decision = Decision(action=Action.HOLD, rationale="Volatility exceeds the rule limit.",
                                    evidence_ids=proposal.evidence_ids, risk_veto="volatility_limit")
            else:
                decision = Decision(**proposal.model_dump())
        return {"decision": decision.model_dump(mode="json")}

    class WorkflowState(State, total=False):
        memory_trace: dict

    nodes = {"prepare": start, "analyst": analyst, "tools": tools, "trader": trader, "risk": risk}
    if a3 is None:
        graph = StateGraph(WorkflowState if memory is not None else State)
        for name, fn in nodes.items():
            graph.add_node(name, fn)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "analyst")
        graph.add_conditional_edges("analyst", lambda s: "tools" if s["pending"] else "trader",
            {"tools": "tools", "trader": "trader"})
        graph.add_edge("tools", "analyst")
        graph.add_edge("trader", "risk")
        graph.add_edge("risk", END)
        return graph.compile()

    class A3State(WorkflowState, total=False):
        debate: dict
        debate_prior: dict
        debate_trace: list[dict]
        debate_next: bool
        debate_round: int
        verification: dict
        a3_stop: bool
        a3_verify_retry: bool
        a3_repair_hint: str

    graph = StateGraph(A3State)
    nodes.update(
        analyst=lambda s: a3.recover_node("analyst", analyst, s, backend),
        trader=lambda s: a3.recover_node("trader", trader, s, backend),
        route=lambda s: a3.route_debate(s, view, budget, backend),
        bull=lambda s: a3.research(s, backend, "bull"),
        bear=lambda s: a3.research(s, backend, "bear"),
        adjudicate=lambda s: a3.adjudicate(s, backend),
    )
    if a3.config.verifier:
        nodes["verifier"] = lambda s: a3.verify(s, view, budget)
    for name, fn in nodes.items():
        graph.add_node(name, a3.wrap(name, fn, backend))

    def edge(origin, next_node, choices):
        graph.add_conditional_edges(origin, lambda s: "risk" if s.get("a3_stop") else next_node(s),
                                    {name: name for name in {*choices, "risk"}})

    graph.add_edge(START, "prepare")
    edge("prepare", lambda s: "analyst", ["analyst"])
    edge("analyst", lambda s: "tools" if s["pending"] else "route", ["tools", "route"])
    edge("tools", lambda s: "analyst", ["analyst"])
    edge("route", lambda s: "bull" if s.get("debate_next") else "trader", ["bull", "trader"])
    edge("bull", lambda s: "risk" if s["errors"] else "bear", ["bear"])
    edge("bear", lambda s: "risk" if s["errors"] else "adjudicate", ["adjudicate"])
    edge("adjudicate", lambda s: "bull" if s["debate_next"] else "trader", ["bull", "trader"])
    if a3.config.verifier:
        edge("trader", lambda s: "verifier", ["verifier"])
        edge("verifier", lambda s: "trader" if s.get("a3_verify_retry") else "risk", ["trader"])
    else:
        graph.add_edge("trader", "risk")
    graph.add_edge("risk", END)
    return graph.compile()
