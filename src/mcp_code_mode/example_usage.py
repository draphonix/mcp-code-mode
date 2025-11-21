"""
End-to-end example usage of the Phase 3 Code Generation Agent.

This script:
1. Configures DSpy (requires OPENAI_API_KEY env var).
2. Discovers MCP tools using `mcp_servers.json`.
3. Initializes the CodeExecutionAgent.
4. Runs a sample task.
"""
import asyncio
import logging
import os
import sys

# Add the src directory to sys.path so we can import mcp_code_mode
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    import dspy
    from dotenv import load_dotenv
    from mcp_code_mode.agent import CodeExecutionAgent
    from mcp_code_mode.executor import LocalPythonExecutor
    from mcp_code_mode.mcp_integration import setup_mcp_tools
    from mcp_code_mode.tool_bridge import MCPToolBridge
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please ensure you have installed the requirements (including python-dotenv).")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


async def main():
    # 1. Configure DSpy
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Prefer OpenAI for speed/reliability if available, then Gemini
    if openai_key:
        # Ensure we use the openai provider prefix for LiteLLM
        model = openai_model if openai_model.startswith("openai/") else f"openai/{openai_model}"
        lm = dspy.LM(model, api_key=openai_key, api_base=openai_base)
        dspy.configure(lm=lm)
        print(f"✅ DSpy configured with OpenAI ({openai_model}){' at ' + openai_base if openai_base else ''}")
    elif gemini_key:
        # Use dspy.LM with gemini/ prefix which uses litellm under the hood
        try:
            # Note: dspy.Google is deprecated/removed in newer versions, use dspy.LM
            lm = dspy.LM("gemini/gemini-2.5-pro", api_key=gemini_key)
            dspy.configure(lm=lm)
            print("✅ DSpy configured with Gemini (gemini/gemini-2.5-pro)")
        except Exception as e:
            print(f"❌ Failed to configure Gemini: {e}")
            return
    else:
        print("❌ No API key found. Please set GEMINI_API_KEY or OPENAI_API_KEY.")
        return

    # 2. Discover MCP Tools
    print("\n🔍 Discovering MCP tools...")
    
    # Find mcp_servers.json
    # It should be in the project root (parent of src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    config_path = os.path.join(project_root, "mcp_servers.json")
    
    if not os.path.exists(config_path):
        # Fallback to current directory
        config_path = "mcp_servers.json"
    
    print(f"Using config: {config_path}")

    from mcp_code_mode.mcp_manager import MCPServerManager

    manager = MCPServerManager(config_path=config_path)
    bridge = None

    try:
        mcp_setup = await setup_mcp_tools(manager=manager)
    except Exception as e:
        print(f"❌ Failed to discover tools: {e}")
        print(f"Make sure `{config_path}` exists and servers are installed.")
        await manager.shutdown()
        return

    tools = mcp_setup["tools"]
    tool_context = mcp_setup["llm_context"]
    print(f"✅ Discovered {len(tools)} tools")

    # 3. Initialize the sandbox + bridge used by the agent
    executor = LocalPythonExecutor()
    bridge = MCPToolBridge(tools)
    await bridge.start()

    async def sandbox_runner(code, timeout=30, ctx=None, variables=None):
        return await executor.run(code, timeout=timeout, variables=variables)

    # 4. Initialize Agent
    print("\n🤖 Initializing CodeExecutionAgent...")
    agent = CodeExecutionAgent(
        mcp_tools=tools,
        tool_context=tool_context,
        sandbox_runner=sandbox_runner,
        tool_bridge=bridge,
    )

    # 5. Run a Task
    # We'll ask for something that uses the filesystem tool if available,
    # or just a simple calculation if not.
    task = "Create a file named 'hello_mcp.txt' in /tmp with the content 'Hello from Phase 3 Agent!'"
    
    print(f"\n🚀 Running task: {task}")
    try:
        result = await agent.run(task, timeout=120)

        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"Task: {result['task']}")
        print("-" * 20)
        print(f"Generated Code:\n{result['generated_code']}")
        print("-" * 20)
        print(f"Execution Result:\n{result['execution_result']}")
        print("="*60)

        # Verify if file was created (if execution succeeded)
        # Note: This check runs in the HOST environment.
        if os.path.exists("/tmp/hello_mcp.txt"):
            with open("/tmp/hello_mcp.txt", "r", encoding="utf-8") as f:
                content = f.read()
            print(f"\n✅ Verification: File found with content: {content}")
        else:
            print("\n⚠️ Verification: File not found in /tmp (expected if sandbox is isolated or execution failed)")

    except Exception as e:
        print(f"\n❌ Agent failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bridge is not None:
            await bridge.stop()
        await manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
