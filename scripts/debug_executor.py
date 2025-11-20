
import asyncio
import os
import sys
import logging
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
LOGGER = logging.getLogger("debug_client")

async def main():
    # Load environment variables (API keys)
    load_dotenv()
    
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        LOGGER.warning("No API keys found in environment! processing might fail.")

    # Path to the python executable
    python_exe = sys.executable
    
    # Server parameters
    # We run the module as a script
    server_params = StdioServerParameters(
        command=python_exe,
        args=["-m", "mcp_code_mode.executor_server"],
        env=dict(os.environ),  # Pass current environment to the server
    )

    LOGGER.info("Starting MCP server...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            LOGGER.info("Connected to server.")
            
            # Initialize
            await session.initialize()
            LOGGER.info("Initialized session.")

            # List tools
            tools_result = await session.list_tools()
            tools = tools_result.tools
            LOGGER.info(f"Discovered {len(tools)} tools:")
            for t in tools:
                LOGGER.info(f"  - {t.name}: {t.description}")

            # Determine which tool to call
            # We want to test 'run_agent' if available, otherwise 'execute_code'
            tool_name = "run_agent"
            if not any(t.name == tool_name for t in tools):
                tool_name = "execute_code"
                if not any(t.name == tool_name for t in tools):
                    LOGGER.error("Neither run_agent nor execute_code found!")
                    return

            LOGGER.info(f"Calling tool: {tool_name}")
            
            # Define the task
            if tool_name == "run_agent":
                args = {"task": "Find me the names from the memory with the memory tool"}
            else:
                # execute_code
                args = {"code": "print('Hello from direct execution!')"}

            try:
                result = await session.call_tool(tool_name, arguments=args)
                LOGGER.info("Tool call successful.")
                
                # Print the result content
                for content in result.content:
                    if content.type == "text":
                        print("\n--- Result Output ---\n")
                        print(content.text)
                        print("\n---------------------\n")
                    else:
                        LOGGER.info(f"Received non-text content: {content}")
                        
            except Exception as e:
                LOGGER.exception(f"Tool call failed: {e}")

if __name__ == "__main__":
    # Add src to sys.path so we can find the module if needed, 
    # though running as subprocess with -m handles it if CWD is correct.
    # We will assume this script is run from the project root.
    asyncio.run(main())
