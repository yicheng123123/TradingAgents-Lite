"""Optional native tool calling via an explicitly configured compatible endpoint."""
import json
import os
from urllib.parse import urlsplit

from openai import OpenAI

from .schema import Budget, MarketReport, Proposal


class ModelBackend:
    def __init__(self, client: OpenAI, model: str, budget: Budget, provider_seed: int | None):
        self.client, self.model, self.budget, self.provider_seed = client, model, budget, provider_seed
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "missing_usage_calls": 0}

    def _call(self, messages: list[dict], tools: list[dict]) -> dict:
        options = {"tools": tools, "tool_choice": "auto"} if tools else {}
        if self.provider_seed is not None:
            options["seed"] = self.provider_seed
        self.usage["calls"] += 1
        result = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0,
            max_tokens=self.budget.max_tokens, **options,
        )
        if result.usage:
            self.usage["prompt_tokens"] += result.usage.prompt_tokens
            self.usage["completion_tokens"] += result.usage.completion_tokens
        else:
            self.usage["missing_usage_calls"] += 1
        message = result.choices[0].message
        response = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            response["tool_calls"] = [{"id": c.id, "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in message.tool_calls]
        return response

    def analyst(self, messages, tools):
        return self._call(messages, tools)

    def trader(self, report: MarketReport, evidence: dict) -> Proposal:
        result = self._call([{"role": "system", "content": (
            "Output a research proposal as strict JSON only. Use only the evidence below and exact evidence IDs. "
            "Allowed actions are buy, sell and hold. Do not invent future or outside facts. "
            + json.dumps(Proposal.model_json_schema()))},
            {"role": "user", "content": json.dumps({"report": report.model_dump(), "evidence": evidence})}], [])
        return Proposal.model_validate_json(result["content"])


def client_from_env() -> tuple[OpenAI, str, str]:
    key, model, base = (os.getenv(k, "") for k in ("TALITE_API_KEY", "TALITE_MODEL", "TALITE_BASE_URL"))
    if not all((key, model, base)):
        raise ValueError("LLM mode requires TALITE_API_KEY, TALITE_MODEL and TALITE_BASE_URL")
    url = urlsplit(base)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Set a credential-free HTTPS API base URL")
    # Only host is recorded in artifacts. No secrets are sent to the graph.
    return OpenAI(api_key=key, base_url=base, max_retries=1, timeout=25), model, url.hostname
