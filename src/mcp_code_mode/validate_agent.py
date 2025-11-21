"""Tech validation script for CodeExecutionAgent.

This script validates that the agent can:
1. Accept a task and tool context.
2. Generate code using DSpy.
3. Use the provided tools in the generated code.
"""
import asyncio
import logging
import sys
import os
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

try:
    import dspy
    from mcp_code_mode.agent import CodeExecutionAgent
    from mcp_code_mode.executor import LocalPythonExecutor
    from mcp_code_mode.tool_formatter import ToolSchemaFormatter
    from mcp_code_mode.tool_bridge import MCPToolBridge
except ImportError as e:
    LOGGER.error("Failed to import dependencies: %s", e)
    sys.exit(1)


# Mock Tool for testing
class MockTool:
    def __init__(self, name: str, description: str, schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = schema

    def __call__(self, **kwargs):
        return f"Mock result for {self.name} with args: {kwargs}"


async def validate_agent():
    """Run validation steps for the CodeExecutionAgent."""
    
    # 1. Setup Mock Tools
    LOGGER.info("Step 1: Setting up mock tools...")
    read_file_tool = MockTool(
        name="read_file",
        description="Read the contents of a file",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to file"}
            },
            "required": ["path"],
        },
    )
    
    write_file_tool = MockTool(
        name="write_file",
        description="Write content to a file",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    )
    
    tools = [read_file_tool, write_file_tool]

    # 2. Format Tools for LLM
    LOGGER.info("Step 2: Formatting tools for LLM...")
    formatter = ToolSchemaFormatter(tools)
    tool_context = formatter.format_for_llm()
    LOGGER.info("Tool Context Preview:\n%s", tool_context[:200] + "...")

    # 3. Initialize DSpy + Agent dependencies
    LOGGER.info("Step 3: Initializing CodeExecutionAgent...")

    try:
        if not dspy.settings.lm:
            LOGGER.warning("No DSpy LM configured. Attempting to configure default...")
            
            openai_key = os.environ.get("OPENAI_API_KEY")
            openai_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            
            if openai_key:
                # Ensure we use the openai provider prefix for LiteLLM
                model = openai_model if openai_model.startswith("openai/") else f"openai/{openai_model}"
                lm = dspy.LM(model, api_key=openai_key, api_base=openai_base)
                dspy.configure(lm=lm)
                LOGGER.info(f"DSpy configured with OpenAI ({openai_model}) at {openai_base}")
            else:
                # Fallback to default
                lm = dspy.LM("openai/gpt-4o-mini")
                dspy.configure(lm=lm)
    except Exception as e:
        LOGGER.error("Failed to configure default LM: %s", e)
        LOGGER.error("Please ensure OPENAI_API_KEY is set or configure DSpy manually.")
        return

    executor = LocalPythonExecutor()
    bridge = MCPToolBridge(tools)
    await bridge.start()

    async def sandbox_runner(code, timeout=30, ctx=None, variables=None):
        return await executor.run(code, timeout=timeout, variables=variables)

    try:
        agent = CodeExecutionAgent(
            mcp_tools=tools,
            tool_context=tool_context,
            sandbox_runner=sandbox_runner,
            tool_bridge=bridge,
        )
    except Exception as e:
        await bridge.stop()
        LOGGER.error("Failed to initialize agent: %s", e)
        return

    # 4. Run Agent with a Task
    task = "Read the file at /tmp/test.txt and print its content."
    LOGGER.info("Step 4: Running agent with task: '%s'", task)

    try:
        result = await agent.run(task)

        LOGGER.info("Validation Result:")
        LOGGER.info("  Success: %s", result["execution_result"]["success"])
        LOGGER.info("  Generated Code:\n%s", result["generated_code"])

        # 5. Verify Output
        if "read_file" in result["generated_code"]:
            LOGGER.info("✅ PASS: Generated code uses 'read_file'")
        else:
            LOGGER.error("❌ FAIL: Generated code does NOT use 'read_file'")

        if result["execution_result"]["success"]:
            LOGGER.info("✅ PASS: Execution simulation successful")
        else:
            LOGGER.error("❌ FAIL: Execution simulation failed")

    except Exception as e:
        LOGGER.error("Agent run failed: %s", e)
        import traceback
        traceback.print_exc()
    finally:
        await bridge.stop()


if __name__ == "__main__":
    asyncio.run(validate_agent())
