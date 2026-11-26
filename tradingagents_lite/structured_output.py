"""B1 Flash V3 validation and redacted diagnostics."""
import json
import re
from hashlib import sha256

from pydantic import ValidationError

from .prompts import REPAIR_JSON_V3
from .provider import ModelCallError
from .schema import Action, MarketReport, Proposal


CONTRACTS = {"market_report": MarketReport, "proposal": Proposal}
ERROR_TYPES = {
    "json_invalid": "invalid_json",
    "missing": "missing_field",
    "extra_forbidden": "extra_field",
    "string_too_long": "string_too_long",
    "too_long": "list_too_long",
    "string_too_short": "string_too_short",
    "too_short": "list_too_short",
}
FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", re.IGNORECASE | re.DOTALL)


def schema_hashes() -> dict[str, str]:
    return {stage: sha256(json.dumps(model.model_json_schema(), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
            for stage, model in CONTRACTS.items()}


def validate_reply(stage: str, reply: dict, evidence: dict) -> tuple[object | None, dict]:
    model = CONTRACTS[stage]
    body = reply.get("content") or ""
    finish = reply.get("finish_reason")
    raw_bytes = len(body.encode("utf-8"))
    body = body.strip()
    fence = FENCE.fullmatch(body)
    if fence:
        body = fence[1]
    issues = []
    value = None
    if finish == "length":
        issues.append({"loc": [], "type": "output_truncated"})
    elif reply.get("tool_calls"):
        issues.append({"loc": [], "type": "unexpected_tool_calls"})
    elif finish not in {None, "stop"}:
        issues.append({"loc": [], "type": "unexpected_finish_reason"})
    else:
        try:
            value = model.model_validate_json(body)
        except ValidationError as exc:
            for error in exc.errors(include_input=False, include_context=False, include_url=False):
                # Extra keys are model output and may themselves contain sensitive text.
                location = [part if isinstance(part, int) or part in model.model_fields else "<extra>"
                            for part in error["loc"]]
                issues.append({"loc": location, "type": ERROR_TYPES.get(error["type"], "invalid_field_type"),
                               "pydantic_type": error["type"]})
        else:
            issues.extend({"loc": ["evidence_ids", index], "type": "invalid_citation"}
                          for index, item in enumerate(value.evidence_ids) if item not in evidence)
            if isinstance(value, Proposal) and value.action != Action.HOLD and not value.evidence_ids:
                issues.append({"loc": ["evidence_ids"], "type": "invalid_citation"})
    event = {"stage": stage, "valid": not issues, "errors": issues, "error_count": len(issues),
             "response_bytes": raw_bytes, "finish_reason": finish if finish in {
                 None, "stop", "length", "tool_calls", "sensitive", "network_error", "content_filter"
             } else "other", "schema_hash": schema_hashes()[stage], "removed_markdown_fence": bool(fence)}
    return (None if issues else value), event


class StructuredOutput:
    """Share one repair allowance across the Analyst and Trader for a task instance."""

    def __init__(self, backend):
        self.backend = backend
        self.repair_used = False

    def generate(self, stage, messages, evidence, reply=None):
        if reply is None:
            reply = self.backend._call(messages, [], structured_stage=stage)
        value, first = validate_reply(stage, reply, evidence)
        events = [{**first, "attempt": "initial"}]
        repairable = first["errors"] and all(
            not issue["type"].startswith("unexpected_") for issue in first["errors"])
        if value is not None or self.repair_used or not repairable:
            return value, events, None

        self.repair_used = True
        feedback = [{"loc": issue["loc"], "type": issue["type"]} for issue in first["errors"][:6]]
        repair_messages = [*messages, {"role": "user", "content": REPAIR_JSON_V3
                           + json.dumps(feedback, separators=(",", ":"))}]
        before = len(self.backend.call_log)
        try:
            reply = self.backend._call(repair_messages, [], structured_stage=stage, repair=True)
        except ModelCallError as exc:
            events.append({"stage": stage, "valid": False, "attempt": "repair",
                           "call_error": exc.code, "charged_tokens": sum(
                               call.get("charged_tokens", 0) for call in self.backend.call_log[before:])})
            return None, events, exc
        value, repaired = validate_reply(stage, reply, evidence)
        events.append({**repaired, "attempt": "repair", "charged_tokens": sum(
            call.get("charged_tokens", 0) for call in self.backend.call_log[before:])})
        return value, events, None
