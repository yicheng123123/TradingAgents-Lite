"""Bounded debate and redacted per-task execution tracing."""
from dataclasses import dataclass
import json
from time import perf_counter
from typing import Literal

from pydantic import Field

from .provider import ModelCallError
from .schema import Action, Contract, MarketReport
from .token_budget import TokenBudget
from .tool_registry import FINANCIAL_SPEC, FUNDAMENTALS_SPEC, RAG_SPEC
from .verifier import verify_citations, verify_proposal


@dataclass(frozen=True)
class A3Config:
    debate: Literal["D0", "D1", "D2"] = "D0"
    transport: Literal["local", "mcp"] = "local"
    verifier: bool = False
    repair: bool = False
    supplement: bool = False
    max_steps: int = 24
    partition: str = "development"

    def __post_init__(self):
        if self.debate not in {"D0", "D1", "D2"} or self.transport not in {"local", "mcp"}:
            raise ValueError("Invalid A3 experiment mode")
        if not 3 <= self.max_steps <= 30 or self.supplement and self.debate != "D2":
            raise ValueError("Invalid A3 limits")


class Claim(Contract):
    kind: Literal["fact", "inference", "uncertainty"]
    statement: str = Field(min_length=1, max_length=480)
    evidence_ids: list[str] = Field(min_length=1, max_length=4)


class ResearchView(Contract):
    stance: Literal["bull", "bear"]
    action: Action
    claims: list[Claim] = Field(min_length=1, max_length=3)


def context_bytes(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")).encode())


def role_context(state):
    report = MarketReport.model_validate(state["report"])
    return report, {key: state["evidence"][key] for key in report.evidence_ids}


class A3Runtime:
    def __init__(self, config: A3Config, run_id: str, task, *, transport=None, fixed_rag=False):
        self.config, self.run_id, self.task = config, run_id, task
        self.transport, self.fixed_rag = transport, fixed_rag
        self.extra_definitions = {}
        if transport is not None:
            if transport.config.get("evidence"):
                self.extra_definitions["financial_evidence"] = FINANCIAL_SPEC
            if transport.config.get("rag"):
                self.extra_definitions["rag_search"] = RAG_SPEC
            if transport.config.get("fundamentals"):
                self.extra_definitions["xbrl_fundamentals"] = FUNDAMENTALS_SPEC
        self.events = []
        self.context_sizes = []
        self.steps = 0
        self.role_calls = 0
        self.repaired = set()
        self.deferred_repairs = {}
        self.tool_fault = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.transport is not None:
            self.transport.close()

    def event(self, node, started, *, backend=None, before=0, **fields):
        calls = getattr(backend, "call_log", [])[before:] if backend is not None else []
        usage = {key: sum(c.get(key) or 0 for c in calls)
                 for key in ("prompt_tokens", "completion_tokens", "charged_tokens")}
        row = {"run_id": self.run_id, "task_id": self.task.task_id, "node": node, "agent": node,
               "tool_name": None, "call_id": f"{self.task.task_id}:{len(self.events)+1}",
               "kind": "control", "input_summary": {}, "output_summary": {}, "error_code": None, "retry_count": 0,
               "recovery_status": "not_needed", "latency_ms": (perf_counter()-started)*1000,
               "token_usage": usage, "provider_attempts": len(calls), **fields}
        self.events.append(row)

    def tool(self, name, raw, local, call_id=None):
        started = perf_counter()
        previous = self.tool_fault
        execute = self.transport.call if self.transport is not None else local
        result = execute(name, raw)
        fault = result.get("error")
        if not fault and name == "rag_search" and not result.get("items"):
            fault = "rag_empty"
        code = result.get("error")
        retries = 0
        if (self.config.repair and "tool" not in self.repaired and name != "rag_search"
                and code in {"tool_timeout", "mcp_timeout", "mcp_unavailable"}):
            self.repaired.add("tool")
            self.event("tools", started, kind="tool", tool_name=name, call_id=call_id, error_code=code,
                       input_summary={"bytes": len(raw.encode())}, recovery_status="retrying")
            started = perf_counter()
            result, retries = execute(name, raw), 1
        final_error = result.get("error") or ("rag_empty" if fault == "rag_empty" else None)
        redirected = bool(previous and not final_error and self.config.repair)
        self.event("tools", started, kind="tool", tool_name=name, call_id=call_id,
                   input_summary={"bytes": len(raw.encode())},
                   output_summary={"items": len(result.get("items", [])), "bytes": context_bytes(result)},
                   error_code=final_error, retry_count=retries,
                   repaired_from=previous if redirected else code if retries else None,
                   recovery_status="recovered" if (retries or redirected) and not final_error else
                                   "failed" if fault else "not_needed")
        if final_error:
            self.tool_fault = final_error
        elif redirected:
            self.tool_fault = None
        return result

    def wrap(self, name, fn, backend):
        def action(state):
            started = perf_counter()
            if name != "risk" and self.steps >= self.config.max_steps:
                self.event(name, started, kind="node", error_code="step_limit")
                return {"a3_stop": True, "proposal": None, "pending": [],
                        "errors": [*state.get("errors", []), {"stage": name, "code": "step_limit"}]}
            self.steps += 1
            before = len(getattr(backend, "call_log", []))
            context_start = len(self.context_sizes)
            try:
                out = fn(state)
            except Exception as exc:
                self.event(name, started, kind="node", backend=backend, before=before,
                           error_code=type(exc).__name__)
                raise
            errors = out.get("errors", [])
            new = errors[len(state.get("errors", [])):]
            error = new[-1]["code"] if new else None
            self.event(name, started, kind="node", backend=backend, before=before, error_code=error,
                       input_summary={"context_bytes": sum(self.context_sizes[context_start:])},
                       output_summary={"fields": sorted(out), "round": state.get("debate_round", 0)})
            return out
        return action

    def recover_node(self, name, fn, state, backend):
        result = fn(state)
        error = result.get("errors", [])[-1:] if result.get("errors") != state.get("errors") else []
        expected = "report" if name == "analyst" else "proposal"
        if name in self.deferred_repairs and (expected in result or error):
            failed = self.deferred_repairs.pop(name)
            self.event(name, perf_counter(), retry_count=1, repaired_from=failed,
                       error_code=error[0]["code"] if error else None,
                       recovery_status="failed" if error else "recovered")
        if (not self.config.repair or name in self.repaired or not error
                or error[0]["stage"] != name or error[0]["code"] not in {
                    "ValidationError", "JSONDecodeError", "invalid_citation", "invalid_response_shape"}):
            return result
        self.repaired.add(name)
        started = perf_counter()
        failed = error[0]["code"]
        self.event(name, started, error_code=failed, recovery_status="retrying")
        self.steps += 1
        if self.steps > self.config.max_steps:
            return {"a3_stop": True, "proposal": None, "pending": [],
                    "errors": [*state["errors"], {"stage": name, "code": "step_limit"}]}
        hint = {"role": "user", "content": "Repair the JSON schema and only cite supplied current evidence IDs."}
        fixed = fn({**state, "messages": [*state.get("messages", []), hint], "a3_repair_hint": failed})
        for key in ("structured_output_trace", "citation_trace"):
            prior = state.get(key, [])
            failed_events = result.get(key, [])[len(prior):]
            if failed_events:
                fixed[key] = [*prior, *failed_events, *fixed.get(key, [])[len(prior):]]
        succeeded = expected in fixed and not fixed.get("errors")
        pending = bool(fixed.get("pending"))
        if pending:
            self.deferred_repairs[name] = failed
        self.event(name, started, retry_count=1, error_code=None if succeeded else failed,
                   recovery_status="recovered" if succeeded else "pending" if pending else "failed")
        return fixed

    def enough_budget(self, backend, calls):
        ledger = getattr(backend, "token_budget", None)
        if ledger is None:
            return True
        each = TokenBudget.reservation_size(backend.max_prompt_bytes, backend.budget.max_tokens)
        return ledger.can_reserve(each * calls, experiment_id=backend.experiment_id, phase=backend.phase)["fits"]

    def route_debate(self, state, view, budget, backend):
        if not state.get("report") or state.get("errors"):
            return {"debate_trace": [{"reason": "disabled_or_no_valid_report", "triggered": False}]}
        report, evidence = role_context(state)
        check = verify_citations(report.evidence_ids, evidence, set(state.get("context_ids", [])),
                                 self.task, partition=self.config.partition)
        if not check["passed"]:
            return {"debate_trace": [{"reason": check["error_code"], "triggered": False}],
                    "verification": check, "errors": [{"stage": "verifier", "code": check["error_code"]}],
                    "a3_stop": True}
        if self.config.debate == "D0":
            return {"debate_trace": [{"reason": "disabled", "triggered": False}]}
        indicators = view.indicators()
        reasons = []
        if self.config.debate == "D1":
            reasons.append("fixed")
        else:
            if len(evidence) < 2:
                reasons.append("thin_evidence")
            if abs(indicators["return_5"]) <= budget.momentum_threshold:
                reasons.append("near_momentum_threshold")
            if indicators["daily_volatility_20"] > budget.volatility_limit:
                reasons.append("risk_gate_expected")
        allowed = bool(reasons) and self.enough_budget(backend, 3)
        if reasons and not allowed:
            reasons.append("insufficient_token_budget")
        return {"debate_next": allowed, "debate_round": 1,
                "debate_trace": [{"reason": reasons or ["sufficient_evidence"], "triggered": allowed}]}

    def research(self, state, backend, stance):
        report, evidence = role_context(state)
        prior = state.get("debate_prior") if state["debate_round"] == 2 else None
        self.context_sizes.append(context_bytes({"report": report.model_dump(), "evidence": evidence, "prior": prior}))
        self.role_calls += 1
        try:
            if hasattr(backend, "a3_research"):
                raw = backend.a3_research(stance, report, evidence, prior)
            else:
                raw = {"stance": stance, "action": "buy" if stance == "bull" else "sell", "claims": [{
                    "kind": "uncertainty", "statement": "Check whether the observed momentum persists.",
                    "evidence_ids": report.evidence_ids[:1]}]}
            result = ResearchView.model_validate(raw)
            if result.stance != stance:
                raise ValueError("role_mismatch")
            ids = [key for claim in result.claims for key in claim.evidence_ids]
            check = verify_citations(ids, evidence, set(evidence), self.task, partition=self.config.partition)
            if not check["passed"]:
                return {"errors": [*state["errors"], {"stage": stance, "code": check["error_code"]}],
                        "verification": check}
        except (ModelCallError, ValueError, TypeError) as exc:
            return {"errors": [*state["errors"], {"stage": stance, "code": getattr(exc, "code", "invalid_role_output")}]}
        return {"debate": {**state.get("debate", {}), stance: result.model_dump(mode="json")}}

    def adjudicate(self, state, backend):
        debate = state.get("debate", {})
        bull, bear = debate.get("bull"), debate.get("bear")
        conflict = bool(bull and bear and bull["action"] != bear["action"])
        distinct = bool(conflict and
            {key for c in bull["claims"] for key in c["evidence_ids"]} !=
            {key for c in bear["claims"] for key in c["evidence_ids"]})
        budget_ok = self.enough_budget(backend, 3) if distinct and self.config.supplement else True
        supplement = (self.config.supplement and state["debate_round"] == 1 and distinct and budget_ok)
        reason = ("distinct_evidence_conflict" if supplement else "insufficient_token_budget" if not budget_ok
                  else "max_rounds" if state["debate_round"] == 2 else "stop_after_round")
        return {"debate_next": supplement, "debate_round": 2 if supplement else state["debate_round"],
                **({"debate_prior": debate} if supplement else {}),
                "debate_trace": [*state.get("debate_trace", []),
                                 {"reason": reason, "triggered": supplement, "conflict": conflict}]}

    def verify(self, state, view, budget):
        historical = {key for c in (state.get("memory_trace") or {}).get("context", [])
                      for key in c.get("historical_evidence_ids", [])}
        check = verify_proposal(state.get("proposal"), state.get("evidence", {}),
                                set((state.get("report") or {}).get("evidence_ids", [])),
                                self.task, partition=self.config.partition, historical_ids=historical,
                                errors=state.get("errors", []), volatility=view.indicators()["daily_volatility_20"],
                                volatility_limit=budget.volatility_limit)
        if check["passed"] or state.get("errors"):
            return {"verification": check, "a3_verify_retry": False}
        if self.config.repair and check["repairable"] and "trader" not in self.repaired:
            self.repaired.add("trader")
            return {"verification": check, "proposal": None, "a3_verify_retry": True,
                    "a3_repair_hint": check["error_code"]}
        return {"verification": check, "proposal": None, "a3_verify_retry": False,
                "errors": [*state["errors"], {"stage": "verifier", "code": check["error_code"]}]}
