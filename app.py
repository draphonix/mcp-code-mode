import gradio as gr
import asyncio
import os
import sys
import json
import tempfile
from typing import List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# Load system env vars (optional system defaults)
load_dotenv()

# Configuration
PYTHON_EXE = sys.executable
# Using the installed package module path
SERVER_SCRIPT = "mcp_code_mode.executor_server"

# Load default MCP servers config
def load_default_mcp_config() -> str:
    """Load mcp_servers_hf.json and return as formatted JSON string."""
    config_path = os.path.join(os.path.dirname(__file__), "mcp_servers_hf.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return json.dumps(config, indent=2)
    except FileNotFoundError:
        return json.dumps({"servers": {}}, indent=2)

async def run_agent_task(task_message: str, history: List, openai_key: str, gemini_key: str, mcp_config: str):
    """
    Connects to the MCP server using the user's provided keys and MCP configuration.
    """
    if not openai_key and not gemini_key:
        return "⚠️ Error: Please provide an OpenAI or Gemini API Key in the settings."

    # Validate MCP config JSON
    try:
        mcp_config_dict = json.loads(mcp_config)
    except json.JSONDecodeError as e:
        return f"⚠️ Error: Invalid MCP configuration JSON: {str(e)}"

    # Write MCP config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(mcp_config_dict, tmp)
        mcp_config_path = tmp.name

    # Prepare environment for the subprocess
    # We inherit system env but override/add the user's keys and MCP config path
    env = dict(os.environ)
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key
    if gemini_key:
        env["GEMINI_API_KEY"] = gemini_key
    env["MCP_SERVERS_CONFIG"] = mcp_config_path
    
    # Server parameters
    server_params = StdioServerParameters(
        command=PYTHON_EXE,
        args=["-m", SERVER_SCRIPT],
        env=env,
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Find the run_agent tool
                tools_result = await session.list_tools()
                tools = tools_result.tools
                
                # Check for run_agent
                agent_tool = next((t for t in tools if t.name == "run_agent"), None)
                if not agent_tool:
                    return "Error: 'run_agent' tool not found on server."

                # Call the tool
                # The agent logic might take some time
                result = await session.call_tool("run_agent", arguments={"task": task_message})
                
                # Format output
                output_text = ""
                for content in result.content:
                    if content.type == "text":
                        output_text += content.text + "\n"
                
                return output_text

    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        # Clean up temporary config file
        try:
            if os.path.exists(mcp_config_path):
                os.unlink(mcp_config_path)
        except Exception:
            pass

# Sync wrapper for Gradio
def chat_wrapper(message, history, openai_key, gemini_key, mcp_config):
    return asyncio.run(run_agent_task(message, history, openai_key, gemini_key, mcp_config))

# Create Interface
with gr.Blocks(title="MCP Agent Code Executor") as demo:
    gr.Markdown("# 🤖 MCP Agent Code Executor")
    gr.Markdown("Enter your API Key(s) below, then describe a coding task.")
    
    with gr.Accordion("⚙️ API Configuration", open=True):
        openai_input = gr.Textbox(
            label="OpenAI API Key", 
            placeholder="sk-...", 
            type="password",
            value=os.environ.get("OPENAI_API_KEY", "") # Pre-fill if env var exists
        )
        gemini_input = gr.Textbox(
            label="Gemini API Key", 
            placeholder="AIza...", 
            type="password",
            value=os.environ.get("GEMINI_API_KEY", "")
        )
    
    with gr.Accordion("🔧 MCP Servers Configuration", open=False):
        mcp_config_input = gr.Textbox(
            label="MCP Servers JSON",
            value=load_default_mcp_config(),
            lines=10,
            info="Configure your MCP servers here. JSON must be valid."
        )
    
    chat_interface = gr.ChatInterface(
        fn=chat_wrapper,
        additional_inputs=[openai_input, gemini_input, mcp_config_input],
        examples=[["Create a fibonacci function in python", "", "", ""], ["Analyze this repository", "", "", ""]],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
