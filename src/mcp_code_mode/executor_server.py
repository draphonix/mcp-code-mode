"""FastMCP server that exposes the Agentic Code Execution capabilities."""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Mapping, AsyncIterator

import dspy
from fastmcp import Context, FastMCP

from .agent import CodeExecutionAgent
from .executor import (
    ExecutionResult,
    SandboxedPythonExecutor,
    LocalPythonExecutor,
)
from .mcp_integration import setup_mcp_tools
from .mcp_manager import MCPServerManager
from .sandbox_config import DEFAULT_SANDBOX_OPTIONS
from .policies import enforce_guardrails
from .tool_bridge import MCPToolBridge
from dotenv import load_dotenv

load_dotenv()
LOGGER = logging.getLogger(__name__)

# --- Global State ---
# We store the manager and tool context here so they persist across requests
SERVER_STATE: Dict[str, Any] = {
    "manager": None,
    "mcp_tools": [],
    "tool_context": "",
    "tool_bridge": None,
}


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Manage the lifecycle of the MCP server connections."""
    LOGGER.info("Initializing Agent Server...")

    # 1. Configure DSpy
    # We default to gpt-4o-mini if available, or let dspy auto-configure if env vars are set
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Prefer OpenAI for speed/reliability if available, then Gemini
    if openai_key:
        lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_key)
        dspy.configure(lm=lm)
        print("✅ DSpy configured with OpenAI (gpt-4o-mini)", file=sys.stderr)
    elif gemini_key:
        # Use dspy.LM with gemini/ prefix which uses litellm under the hood
        try:
            # Note: dspy.Google is deprecated/removed in newer versions, use dspy.LM
            lm = dspy.LM("gemini/gemini-2.5-pro", api_key=gemini_key)
            dspy.configure(lm=lm)
            print("✅ DSpy configured with Gemini (gemini/gemini-2.5-pro)", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to configure Gemini: {e}", file=sys.stderr)
            return
    else:
        print("❌ No API key found. Please set GEMINI_API_KEY or OPENAI_API_KEY.", file=sys.stderr)
        return

    # 2. Connect to upstream MCP servers
    manager = MCPServerManager()
    bridge: MCPToolBridge | None = None
    try:
        # This discovers tools from mcp_servers.json
        setup = await setup_mcp_tools(manager)
        
        SERVER_STATE["manager"] = setup["manager"]
        SERVER_STATE["mcp_tools"] = setup["tools"]
        SERVER_STATE["tool_context"] = setup["llm_context"]
        bridge = MCPToolBridge(setup["tools"])
        await bridge.start()
        SERVER_STATE["tool_bridge"] = bridge
        
        LOGGER.info(f"Discovered {len(SERVER_STATE['mcp_tools'])} tools")
        
        yield
        
    finally:
        LOGGER.info("Shutting down Agent Server...")
        await manager.shutdown()
        if bridge is not None:
            await bridge.stop()
        SERVER_STATE.clear()


# Initialize FastMCP with lifespan
mcp = FastMCP("Code Executor Agent", lifespan=server_lifespan)

# --- Execution Backend Selection ---
# Default to the local executor because Deno+Pyodide cannot perform HTTP
# requests to the MCP tool bridge in some environments (WebAssembly stack
# switching not supported). Users can force the sandbox by setting
# MCP_EXECUTOR=pyodide.
_EXECUTOR_BACKEND = os.environ.get("MCP_EXECUTOR", "local").lower()
if _EXECUTOR_BACKEND in {"pyodide", "sandbox", "deno"}:
    EXECUTOR = SandboxedPythonExecutor(options=DEFAULT_SANDBOX_OPTIONS)
    LOGGER.info("Using Pyodide sandbox executor")
else:
    EXECUTOR = LocalPythonExecutor(max_output_chars=DEFAULT_SANDBOX_OPTIONS.max_output_chars)
    LOGGER.info("Using local in-process executor (network-friendly)")

DEFAULT_TIMEOUT = 30

@mcp.tool()
async def execute_code(
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
    ctx: Context | None = None,
    variables: Mapping[str, Any] | None = None,
) -> ExecutionResult:
    """Execute Python code inside DSpy's sandboxed interpreter.
    
    This is a low-level tool that runs exactly what you give it.
    For agentic problem solving, use 'run_agent' instead.
    """

    try:
        numeric_timeout = _coerce_timeout(timeout)
    except ValueError as exc:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=str(exc),
            duration_ms=0,
            diagnostics={"error_type": "INVALID_ARGUMENT"},
        )

    allowed, violation = enforce_guardrails(code)
    if not allowed:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=violation or "Policy violation",
            duration_ms=0,
            diagnostics={"error_type": "POLICY_VIOLATION"},
        )

    if ctx is not None:
        await ctx.info(
            f"Executing snippet ({len(code)} chars, timeout={numeric_timeout}s)"
        )

    result = await EXECUTOR.run(
        code,
        timeout=numeric_timeout,
        variables=variables,
    )

    if ctx is not None and not result.get("success", False):
        diagnostics = result.get("diagnostics") or {}
        await ctx.error(
            f"Execution failed: {diagnostics.get('error_type', 'Unknown error')}"
        )

    return result


# --- Phase 4: The Agent ---

@mcp.tool()
async def run_agent(
    task: str,
    timeout: int = 120,
    ctx: Context | None = None,
) -> str:
    """
    Run an autonomous coding agent to solve a task.
    
    The agent can:
    1. Access upstream MCP tools (filesystem, memory, etc.)
    2. Write and execute Python code to solve the problem
    3. iterate if errors occur
    
    Args:
        task: The natural language task description (e.g., "Read /tmp/data.txt and summarize it")
    """
    if not SERVER_STATE.get("mcp_tools"):
        return "Error: No tools available. Is the server initialized?"
    if SERVER_STATE.get("tool_bridge") is None:
        return "Error: Tool bridge unavailable"

    if ctx:
        await ctx.info(f"Agent received task: {task}")
        await ctx.report_progress(0, 100)

    try:
        # Initialize the agent with the discovered tools and formatted context
        agent = CodeExecutionAgent(
            mcp_tools=SERVER_STATE["mcp_tools"],
            tool_context=SERVER_STATE["tool_context"],
            sandbox_runner=execute_code.fn,
            tool_bridge=SERVER_STATE["tool_bridge"],
        )

        # Run the agent
        if ctx:
            await ctx.info("Agent is reasoning and generating code...")
        
        result = await agent.run(task, timeout=timeout, ctx=ctx)
        
        execution_result = result["execution_result"]
        generated_code = result["generated_code"]
        
        success = execution_result.get("success", False)
        stdout = execution_result.get("stdout", "")
        stderr = execution_result.get("stderr", "")

        # Format the output for the user
        status_icon = "✅" if success else "❌"
        response = [
            f"{status_icon} **Task Complete**" if success else f"{status_icon} **Task Failed**",
            "",
            "### Execution Output",
            "```",
            stdout if stdout else "(No output)",
            "```",
        ]
        
        if stderr:
            response.extend([
                "",
                "### Errors",
                "```",
                stderr,
                "```"
            ])
            
        response.extend([
            "",
            "### Generated Code",
            "```python",
            generated_code,
            "```"
        ])

        if ctx:
            await ctx.report_progress(100, 100)

        return "\n".join(response)

    except Exception as e:
        LOGGER.exception("Agent failed")
        return f"Agent encountered a critical error: {str(e)}"


def _coerce_timeout(raw: Any) -> float:
    """Convert timeout values from MCP clients into a float."""
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be numeric") from exc

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero seconds")
    return timeout


def main() -> None:
    """CLI entry point for running the executor server."""
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Starting Code Executor Agent Server")
    # FastMCP 2.0+ handles lifespan automatically when run() is called
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
