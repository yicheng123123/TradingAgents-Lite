"""Reproducible offline A3 comparisons and deliberately injected failures."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, median, quantiles
from time import perf_counter
from types import SimpleNamespace

from .agents import A3Config, A3Runtime
from .data import HistoricalData, read_tasks, sha256
from .experiment import ResearchOffline
from .mcp_finance import MCPTransport, bound_registry
from .provider import ModelBackend
from .schema import Budget, MarketReport
from .token_budget import TokenBudget
from .workflow import build_workflow


FAULTS = ("malformed_json", "unknown_tool", "bad_arguments", "mcp_timeout", "mcp_exit",
          "rag_empty", "memory_inconsistent", "invalid_citation", "future_evidence",
          "budget_denied", "role_conflict", "step_limit")


def _call(name, args="{}", key="first"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": key, "type": "function", "function": {"name": name, "arguments": args}}]}


class Scripted(ResearchOffline):
    def __init__(self, budget, fault=None, *, rich=False):
        super().__init__(budget)
        self.fault, self.rich = fault, rich
        self.turns = 0
        self.trader_calls = 0

    def analyst(self, messages, tools):
        self.turns += 1
        if self.turns == 1:
            initial = {
                "malformed_json": {"content": "{"},
                "unknown_tool": _call("made_up"),
                "bad_arguments": _call("market_bars", '{"ticker":"OTHER"}'),
                "rag_empty": _call("rag_search", '{"query":"cash flow"}'),
                "future_evidence": _call("market_bars"),
            }
            if self.fault in initial:
                return initial[self.fault]
            if self.rich:
                result = _call("market_indicators")
                result["tool_calls"].extend(_call("market_bars", '{"limit":2}', "bars")["tool_calls"])
                return result
        tool_data = []
        for message in messages:
            if message["role"] == "tool":
                tool_data.append(json.loads(message["content"]))
        seen = [row["evidence_id"] for row in tool_data if row.get("evidence_id")]
        if not seen:
            return _call("market_indicators", key="retry") if tools else {"content": "{"}
        if self.fault == "future_evidence":
            return {"content": MarketReport(view="Synthetic future bar", evidence_ids=seen).model_dump_json()}
        result = super().analyst(messages, tools)
        if self.rich and result.get("content"):
            value = json.loads(result["content"])
            value["evidence_ids"] = list(dict.fromkeys(value["evidence_ids"] + seen))[:4]
            return {"content": json.dumps(value)}
        return result

    def a3_research(self, stance, report, evidence, prior):
        ids = list(evidence)
        key = ids[-1] if stance == "bear" else ids[0]
        return {"stance": stance, "action": "buy" if stance == "bull" else "sell", "claims": [{
            "kind": "inference", "statement": "Scripted test argument about observed signals.",
            "evidence_ids": [key]}]}

    def trader_a3(self, report, evidence, memory, debate, repair_code=None):
        self.trader_calls += 1
        if self.fault == "invalid_citation" and self.trader_calls == 1:
            return {"action": "buy", "rationale": "Synthetic invalid proposal.", "evidence_ids": ["fabricated"]}
        return super().trader_a3(report, evidence, memory, debate, repair_code)


def _mock_backend(task, budget, path, *, deny=False):
    class MockClient:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if deny:
                raise AssertionError("Budget denial must precede transport")
            payload = (
                MarketReport(view="Simulated held view", evidence_ids=[
                    f"{task.symbol}:{task.decision_date}:indicators"]).model_dump_json()
                if self.calls == 1 else '{"action":"hold","rationale":"Simulated","evidence_ids":[]}'
            )
            return SimpleNamespace(id=f"mock-{self.calls}", model="mock-offline",
                choices=[SimpleNamespace(message=SimpleNamespace(content=payload, tool_calls=[]))],
                usage=SimpleNamespace(prompt_tokens=25, completion_tokens=10))
    kwargs = {"global_limit": 1, "phase_limits": {"A3": 1}} if deny else {}
    ledger = TokenBudget(path, **kwargs)
    backend = ModelBackend(SimpleNamespace(chat=SimpleNamespace(completions=MockClient())), "mock-offline",
                           budget, None, phase="A3", max_retries=0, token_budget=ledger)
    backend.set_budget_context(experiment_id="synthetic", task_id=task.task_id, phase="A3")
    return backend


class FaultRuntime(A3Runtime):
    def __init__(self, *args, fault=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault = fault

    def tool(self, name, raw, local, call_id=None):
        def injectable(tool_name, argument):
            result = local(tool_name, argument)
            if self.fault == "future_evidence" and tool_name == "market_bars":
                return {**result, "bars": [{**result["bars"][0], "day": "2099-01-01"}]}
            return result
        return super().tool(name, raw, injectable, call_id)


def scenario(market, task, output, label, config, *, fault=None, rich=False, mock=False, data_path=None):
    started = perf_counter()
    budget = Budget(max_tokens=512)
    view = market.view(task)
    backend = (_mock_backend(task, budget, output / "mock-ledger.sqlite", deny=fault == "budget_denied")
               if mock or fault == "budget_denied" else Scripted(budget, fault, rich=rich))
    if label.startswith(("T0-", "T1-")):
        backend = ResearchOffline(budget)
    transport = None
    if fault in {"mcp_timeout", "mcp_exit"}:
        # Real SDK startup failure. A fixture changes the next host-bound launch to a valid config.
        binding = {"task": task.model_dump(mode="json"), "market": str(data_path.resolve())}
        if fault == "mcp_exit":
            binding["market"] = str((output / "missing-market").resolve())
        transport = MCPTransport(binding, timeout=.0001 if fault == "mcp_timeout" else 8)
        first = transport.call

        def one_fault(name, raw):
            value = first(name, raw)
            transport.config["market"] = str(data_path.resolve())
            transport.timeout = 8
            return value

        transport.call = one_fault
    elif config.transport == "mcp" and data_path is not None:
        transport = MCPTransport({"task": task.model_dump(mode="json"), "market": str(data_path.resolve())})

    def recall_raises(*args):
        raise RuntimeError("Memory index is inconsistent")

    rag = (SimpleNamespace(config=SimpleNamespace(dispatch="agentic"),
        search=lambda *args: {"status": "unavailable", "items": [], "retrieval_trace": {"stages": {}}})
        if fault == "rag_empty" else None)
    runtime = FaultRuntime(config, label, task, transport=transport, fault=fault)
    with runtime:
        state = build_workflow(view, backend, budget, react=not mock and fault != "budget_denied",
                             rag=rag, memory=SimpleNamespace(before=recall_raises)
                             if fault == "memory_inconsistent" else None, a3=runtime).invoke(
                                 {}, {"recursion_limit": 40})
    verified = state.get("verification") or {}
    errors = state.get("errors", [])
    codes = {err["code"] for err in errors} | {e["error_code"] for e in runtime.events if e["error_code"]}
    recovery = sorted({event["recovery_status"] for event in runtime.events
                       if event["recovery_status"] != "not_needed"})
    completed = not errors and bool(state.get("proposal"))
    return {"case_id": label, "task_id": task.task_id, "fault": fault, "config": asdict(config),
            "completed": completed, "decision": state["decision"]["action"],
            "risk_veto": state["decision"].get("risk_veto"), "errors": sorted(codes), "verification": verified,
            "structured_outputs": [event["valid"] for event in state.get("structured_output_trace", [])],
            "citation_trace": state.get("citation_trace", []), "debate_trace": state.get("debate_trace", []),
            "role_calls": runtime.role_calls, "events": runtime.events, "recovery": recovery,
            "recovered": bool(completed and "recovered" in recovery),
            "degraded": bool(completed and "degraded" in recovery),
            "latency_ms": (perf_counter()-started)*1000, "real_glm_tokens": 0,
            "mock_tokens": sum(e["token_usage"]["prompt_tokens"] + e["token_usage"]["completion_tokens"]
                               for e in runtime.events) if mock else 0}


def equivalence(tasks, market, evidence, rag_index):
    comparisons = []
    for task in tasks:
        binding = {"task": task.model_dump(mode="json"), "market": str(market.resolve()),
                   "evidence": str(evidence.resolve()), "rag": {
                       "index": str(rag_index.resolve()), "models": str(Path("data/local-models").resolve()),
                       "config": {"mode": "R1", "dispatch": "fixed"}, "partition": "development"}}
        calls = {"market_indicators": {}, "market_bars": {"limit": 3},
                 "financial_evidence": {"source_type": "sec_filing"}, "rag_search": {"query": "risk"}}
        with ExitStack() as stack:
            registry = bound_registry(binding, stack)
            local = {}
            for name, args in calls.items():
                start = perf_counter()
                local[name] = (registry.execute(name, args), (perf_counter()-start)*1000)
        with MCPTransport(binding, timeout=20) as child:
            start = perf_counter()
            discover = child.start()
            startup = (perf_counter()-start)*1000
            assert set(discover) == set(calls)
            for name, args in calls.items():
                start = perf_counter()
                observed = child.call(name, json.dumps(args))
                elapsed = (perf_counter()-start)*1000
                expected, local_ms = local[name]
                if "retrieval_trace" in expected and "retrieval_trace" in observed:
                    expected["retrieval_trace"].pop("latency_ms", None)
                    observed["retrieval_trace"].pop("latency_ms", None)
                comparisons.append({"task_id": task.task_id, "tool": name, "equal": observed == expected,
                                    "local_success": "error" not in expected, "mcp_success": "error" not in observed,
                                    "local_ms": local_ms, "mcp_ms": elapsed, "mcp_startup_ms": startup})
    return comparisons


def latency(values):
    return {"n": len(values), "mean": mean(values), "p50": median(values),
            "p95": quantiles(values, n=100, method="inclusive")[94] if len(values) > 1 else values[0]}


def summarize(rows):
    events = [event for row in rows for event in row["events"]]
    tools = [e for e in events if e["tool_name"]]
    structure = [v for row in rows for v in row["structured_outputs"]]
    cites = [e for row in rows for e in row["citation_trace"]]
    checked = [row for row in rows if row["verification"]]
    faults = [row for row in rows if row["fault"]]
    return {"completion": [sum(row["completed"] for row in rows), len(rows)],
            "tool_success": [sum(not e["error_code"] for e in tools), len(tools)],
            "recovery": [sum(row["recovered"] for row in faults), len(faults)],
            "degraded": [sum(row["degraded"] for row in faults), len(faults)],
            "structured_valid": [sum(structure), len(structure)],
            "citation_valid": [sum(len(e["valid_ids"]) for e in cites), sum(len(e["cited_ids"]) for e in cites)],
            "future_detected": [sum("future_evidence" in row["errors"] for row in rows), len(rows)],
            "future_accepted": [sum(row["completed"] and row["fault"] == "future_evidence" for row in rows),
                                sum(row["fault"] == "future_evidence" for row in rows)],
            "verification_passed": [sum(row["verification"]["passed"] for row in checked), len(checked)],
            "debate_trigger": [sum(any(e["triggered"] for e in row["debate_trace"]) for row in rows), len(rows)],
            "role_calls": sum(r["role_calls"] for r in rows), "tool_calls": len(tools),
            "node_calls": sum(e["kind"] == "node" for e in events),
            "context_bytes": sum(e["input_summary"].get("context_bytes", 0) for e in events),
            "latency_ms": latency([row["latency_ms"] for row in rows]),
            "real_glm_tokens": sum(row["real_glm_tokens"] for row in rows),
            "mock_tokens": sum(row["mock_tokens"] for row in rows)}


def benchmark(tasks_path: Path, data_path: Path, evidence: Path, rag_index: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    tasks = read_tasks(tasks_path)
    market = HistoricalData(data_path)
    groups = defaultdict(list)
    for arm, transport in (("T0", "local"), ("T1", "mcp")):
        for task in tasks[:5]:
            groups[arm].append(scenario(market, task, output, f"{arm}-{task.task_id}",
                A3Config(transport=transport, verifier=True), data_path=data_path))
    for mode in ("D0", "D1", "D2"):
        for i, task in enumerate(tasks):
            groups[mode].append(scenario(market, task, output, f"{mode}-{task.task_id}",
                A3Config(debate=mode), rich=i % 4 != 0))
    for mode in ("V0", "V1"):
        for task in tasks[:5]:
            groups[mode].append(scenario(market, task, output, f"{mode}-{task.task_id}",
                A3Config(verifier=mode == "V1"), rich=True))
    for repair in (False, True):
        for fault in FAULTS:
            arm = "F1" if repair else "F0"
            for task in tasks[:5]:
                cfg = A3Config(debate="D2" if fault == "role_conflict" else "D0", verifier=True, repair=repair,
                               supplement=fault == "role_conflict", max_steps=3 if fault == "step_limit" else 24,
                               transport="mcp" if fault.startswith("mcp_") else "local")
                groups[arm].append(scenario(market, task, output, f"{arm}-{fault}-{task.task_id}", cfg,
                    fault=fault, rich=fault == "role_conflict", data_path=data_path))
    groups["mock_llm"].append(scenario(market, tasks[0], output, "mock-llm-valid",
                                       A3Config(verifier=True), mock=True))
    paired = equivalence(tasks[:2], data_path, evidence, rag_index)
    with (output / "cases.jsonl").open("x") as stream:
        for arm, rows in groups.items():
            for row in rows:
                stream.write(json.dumps({"arm": arm, **row}) + "\n")
                stream.flush()
    result = {"tasks_sha256": sha256(tasks_path), "market_sha256": sha256(data_path / "manifest.json"),
              "fault_cases": len(FAULTS)*5, "fault_categories": list(FAULTS), "glm_calls": 0,
              "groups": {arm: summarize(rows) for arm, rows in groups.items()},
              "mcp_equivalence": [sum(row["equal"] and row["mcp_success"] for row in paired), len(paired)],
              "transport_decision_equivalence": [sum(a["completed"] and b["completed"]
                  and (a["decision"], a["risk_veto"]) == (b["decision"], b["risk_veto"])
                  for a, b in zip(groups["T0"], groups["T1"], strict=True)), len(groups["T0"])],
              "transport": {name: latency([r[f"{name}_ms"] for r in paired]) for name in ("local", "mcp")},
              "tools": paired}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
