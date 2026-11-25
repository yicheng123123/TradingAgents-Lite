"""Official SDK STDIO adapter for a finance registry bound by the host."""
from contextlib import ExitStack
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import anyio
from anyio.from_thread import start_blocking_portal
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from .data import HistoricalData
from .evidence import EvidenceStore
from .schema import Task
from .tool_registry import ToolRegistry


def _timed_out(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return any(_timed_out(nested) for nested in exc.exceptions)
    return isinstance(exc, TimeoutError) or getattr(getattr(exc, "error", None), "code", None) == 408


def bound_registry(config: dict, stack: ExitStack) -> ToolRegistry:
    task = Task.model_validate(config["task"])
    view = HistoricalData(Path(config["market"])).view(task)
    financial = EvidenceStore.load(Path(config["evidence"])).view(task) if config.get("evidence") else None
    rag = None
    if config.get("rag"):
        from .rag import RagConfig, RagView, Retriever
        from .vectors import LocalEmbedding, LocalReranker, VectorStore

        options = config["rag"]
        cfg = RagConfig(**options["config"])
        models = Path(options["models"])
        encoder = LocalEmbedding(models, allow_download=False)
        store = stack.enter_context(VectorStore(Path(options["index"]), encoder, "chunks"))
        reranker = LocalReranker(models, allow_download=False) if cfg.mode == "R4" else None
        rag = RagView(Retriever(store, reranker), task, options["partition"], cfg, provenance=True)
    fundamentals = None
    if config.get("fundamentals"):
        from .fundamentals import FundamentalsView

        options = config["fundamentals"]
        fundamentals = FundamentalsView.load(Path(options["root"]), task, options["partition"])
    return ToolRegistry(view, financial, rag, fundamentals)


def server_for(registry: ToolRegistry):
    server = Server("talite-task-finance")

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=name, description=description, inputSchema=model.model_json_schema())
                for name, (model, description) in registry.definitions.items()]

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        payload = registry.execute(name, arguments or {})
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(payload))],
                                    structuredContent=payload, isError="error" in payload)

    return server


async def serve(config: dict):
    with ExitStack() as stack:
        server = server_for(bound_registry(config, stack))
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


class MCPTransport:
    """One lazy child and SDK session per task. Only bound config crosses at startup."""

    def __init__(self, config: dict, *, timeout: float = 8):
        self.config, self.timeout = config, timeout
        self.stack = ExitStack()
        self.portal = self.session = None
        self.discovered = {}

    async def _serve_session(self, *, task_status=anyio.TASK_STATUS_IGNORED):
        settings = StdioServerParameters(command=sys.executable,
            args=["-m", "tradingagents_lite.mcp_finance"],
            env={"TALITE_BOUND_TASK": json.dumps(self.config), "PYTHONUNBUFFERED": "1"})
        async with stdio_client(settings) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self.timeout)) as session:
                with anyio.fail_after(self.timeout):
                    await session.initialize()
                    listing = await session.list_tools()
                task_status.started((session, {t.name: t.inputSchema for t in listing.tools}))
                await anyio.sleep_forever()

    def start(self):
        if self.session is None:
            self.portal = self.stack.enter_context(start_blocking_portal())
            try:
                future, (self.session, self.discovered) = self.portal.start_task(self._serve_session)
                self.stack.callback(future.cancel)
            except BaseException:
                self.close()
                raise
        return self.discovered

    def call(self, name: str, raw: str) -> dict:
        try:
            if len(raw.encode()) > 4096:
                return {"error": "invalid_arguments"}
            args = json.loads(raw)
            if not isinstance(args, dict):
                return {"error": "invalid_arguments"}
        except (ValueError, AttributeError):
            return {"error": "invalid_arguments"}
        try:
            if name not in self.start():
                return {"error": "unknown_tool"}
            result = self.portal.call(self.session.call_tool, name, args)
            if isinstance(result.structuredContent, dict):
                return result.structuredContent
            return {"error": "mcp_protocol_error"}
        except Exception as exc:
            self.close()
            return {"error": "mcp_timeout" if _timed_out(exc) else "mcp_unavailable"}

    def close(self):
        self.stack.close()
        self.portal = self.session = None
        self.discovered = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    anyio.run(serve, json.loads(os.environ.pop("TALITE_BOUND_TASK")))
