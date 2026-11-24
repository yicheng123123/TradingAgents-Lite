"""OpenAI-compatible GLM chat-completions client with bounded retries and audit metadata."""
import hashlib
import json
import os
from time import perf_counter
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from .prompts import TRADER_SYSTEM_PROMPT
from .schema import Budget, MarketReport, Proposal

DEFAULT_MODEL = "glm-5.3"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_MAX_RETRIES = 1
DEFAULT_MAX_PROMPT_BYTES = 32_768


class ModelCallError(RuntimeError):
    """A redacted, machine-readable model failure suitable for experiment artifacts."""
    def __init__(self, code: str, attempts: int):
        super().__init__(code)
        self.code = code
        self.attempts = attempts


def _canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, APITimeoutError, APIConnectionError,
                        RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 408 or exc.status_code >= 500
    return False


def _error_code(exc: Exception) -> str:
    if isinstance(exc, (TimeoutError, APITimeoutError)):
        return "timeout"
    if isinstance(exc, RateLimitError) or (
        isinstance(exc, APIStatusError) and exc.status_code == 429
    ):
        return "rate_limited"
    if isinstance(exc, (ConnectionError, APIConnectionError)):
        return "connection_error"
    if isinstance(exc, APIStatusError):
        return f"http_{exc.status_code}"
    return "api_error"


class ModelBackend:
    def __init__(self, client: OpenAI, model: str, budget: Budget, provider_seed: int | None,
                 *, max_retries: int = DEFAULT_MAX_RETRIES,
                 max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
                 temperature: float = DEFAULT_TEMPERATURE,
                 reasoning_effort: str = DEFAULT_REASONING_EFFORT):
        self.client, self.model, self.budget, self.provider_seed = client, model, budget, provider_seed
        self.max_retries, self.max_prompt_bytes = max_retries, max_prompt_bytes
        self.temperature, self.reasoning_effort = temperature, reasoning_effort
        self.usage = {"calls": 0, "logical_calls": 0, "retries": 0, "failed_calls": 0,
                      "prompt_tokens": 0, "completion_tokens": 0, "missing_usage_calls": 0}
        self.call_log: list[dict] = []

    def _call(self, messages: list[dict], tools: list[dict]) -> dict:
        options = {}
        if tools:
            options.update(tools=tools, tool_choice="auto")
        if self.provider_seed is not None:
            options["seed"] = self.provider_seed
        if self.model.lower().startswith("glm-5.3"):
            options.update(thinking={"type": "enabled"}, reasoning_effort=self.reasoning_effort)
        request = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "max_tokens": self.budget.max_tokens, **options}
        request_sha, prompt_sha, tools_sha = (_canonical_hash(request), _canonical_hash(messages),
                                               _canonical_hash(tools))
        byte_count = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False,
            sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
        self.usage["logical_calls"] += 1
        logical_call = self.usage["logical_calls"]
        if byte_count > self.max_prompt_bytes:
            self.usage["failed_calls"] += 1
            self.call_log.append({"logical_call": logical_call, "attempt": 0, "status": "blocked",
                "error_code": "prompt_byte_limit", "request_bytes": byte_count,
                "max_prompt_bytes": self.max_prompt_bytes, "request_sha256": request_sha,
                "prompt_sha256": prompt_sha, "tools_sha256": tools_sha})
            raise ModelCallError("prompt_byte_limit", 0)

        for attempt in range(self.max_retries + 1):
            started = perf_counter()
            self.usage["calls"] += 1
            try:
                result = self.client.chat.completions.create(**request)
            except Exception as exc:
                elapsed = perf_counter() - started
                code = _error_code(exc)
                retry = _retryable(exc) and attempt < self.max_retries
                self.call_log.append({"logical_call": logical_call, "attempt": attempt + 1,
                    "status": "retrying" if retry else "error", "error_code": code,
                    "http_status": exc.status_code if isinstance(exc, APIStatusError) else None,
                    "latency_seconds": elapsed, "request_bytes": byte_count,
                    "request_sha256": request_sha, "prompt_sha256": prompt_sha, "tools_sha256": tools_sha})
                if retry:
                    self.usage["retries"] += 1
                    continue
                self.usage["failed_calls"] += 1
                raise ModelCallError(code, attempt + 1) from None

            elapsed = perf_counter() - started
            usage = result.usage
            if usage is None:
                self.usage["missing_usage_calls"] += 1
                prompt_tokens = completion_tokens = None
            else:
                prompt_tokens, completion_tokens = int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0)
                self.usage["prompt_tokens"] += prompt_tokens
                self.usage["completion_tokens"] += completion_tokens
            try:
                message = result.choices[0].message
                content = message.content or ""
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                response = {"role": "assistant", "content": content}
                calls = [{"id": call.id, "type": "function",
                          "function": {"name": call.function.name, "arguments": call.function.arguments}}
                         for call in (message.tool_calls or [])]
            except Exception:
                self.usage["failed_calls"] += 1
                self.call_log.append({"logical_call": logical_call, "attempt": attempt + 1,
                    "status": "error", "error_code": "invalid_response_shape",
                    "response_id": getattr(result, "id", None), "actual_model": getattr(result, "model", None),
                    "latency_seconds": elapsed, "request_bytes": byte_count,
                    "request_sha256": request_sha, "prompt_sha256": prompt_sha, "tools_sha256": tools_sha})
                raise ModelCallError("invalid_response_shape", attempt + 1) from None
            if calls:
                response["tool_calls"] = calls
            self.call_log.append({"logical_call": logical_call, "attempt": attempt + 1, "status": "ok",
                "response_id": getattr(result, "id", None), "actual_model": getattr(result, "model", None),
                "tool_names": [item["function"]["name"] for item in calls],
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "latency_seconds": elapsed, "request_bytes": byte_count,
                "request_sha256": request_sha, "prompt_sha256": prompt_sha, "tools_sha256": tools_sha})
            return response
        raise ModelCallError("retry_budget_exhausted", self.max_retries + 1)

    def analyst(self, messages, tools):
        return self._call(messages, tools)

    def trader(self, report: MarketReport, evidence: dict) -> Proposal:
        result = self._call(
            [{"role": "system", "content": TRADER_SYSTEM_PROMPT},
             {"role": "user", "content": json.dumps({"report": report.model_dump(), "evidence": evidence},
                 ensure_ascii=False, sort_keys=True)}],
            [],
        )
        return Proposal.model_validate_json(result["content"])


def client_from_env() -> tuple[OpenAI, str, str]:
    key = os.getenv("TALITE_API_KEY", "")
    model = os.getenv("TALITE_MODEL", DEFAULT_MODEL)
    base = os.getenv("TALITE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if not key:
        raise ValueError("LLM mode requires TALITE_API_KEY; TALITE_MODEL and TALITE_BASE_URL have GLM defaults")
    url = urlsplit(base)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Set a credential-free HTTPS API base URL")
    return OpenAI(api_key=key, base_url=base, max_retries=0, timeout=25), model, url.hostname
