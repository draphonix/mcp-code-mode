# Plan: Host MCP Server on Hugging Face Spaces with Gradio

This plan outlines the steps to convert your existing MCP Code Executor server into a Gradio application hosted on Hugging Face Spaces.

## 1. Architecture Strategy

We will use **Pattern 3 (Separate Process via Stdio)**. This is the most robust approach because:
- It reuses your existing `executor_server.py` without major refactoring.
- It maintains the complex state management (tools, DSpy config) inside the server process.
- It allows the Gradio app to act as a client to the MCP server, similar to how `debug_executor.py` works.

### Components
1. **MCP Server (Subprocess)**: Runs `src/mcp_code_mode/executor_server.py`. Handles the agent logic, tool execution, and LLM interaction.
2. **Gradio App (Main Process)**: Runs `app.py`. Provides the Web UI, captures user input, and sends requests to the MCP Server via the MCP Python Client.
3. **Hugging Face Space (Container)**: Hosts both processes.

## 2. Constraints & Risks: LocalPythonInterpreter

**Warning**: Your server uses `LocalPythonExecutor` by default.
- In a **Hugging Face Space**, this tool will have access to the container's file system and environment variables (including your API keys).
- **Recommendation**:
    - **Private Space**: If only you use it, `LocalPythonExecutor` is acceptable.
    - **Public Space**: You MUST switch to a sandboxed executor. However, your code notes that `pyodide` has limitations with HTTP requests.
    - **Mitigation**: The plan assumes a **Private Space** initially. If making public, you must implement stricter sandboxing or disable the `execute_code` tool for external users.

## 3. Implementation Plan

### Step 1: Prepare Dependencies (`requirements.txt`)
We need to ensure all dependencies are listed for the HF Space.

**File**: `docs/requirements.txt` (Create or Append)
```text
gradio>=4.0.0
mcp>=0.1.0
python-dotenv
fastmcp
dspy-ai
openai
# Add other dependencies from your current environment
```

### Step 2: Create the Gradio Application (`app.py`)
This file will replace `debug_executor.py` as the entry point. It will:
1. Launch the MCP server as a subprocess **per session/request**.
2. Accept **User API Keys** via the UI.
3. Inject keys into the subprocess environment.
4. Expose a Chat Interface.

**File**: `docs/app.py`
```python
import gradio as gr
import asyncio
import os
import sys
from typing import List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# Load system env vars (optional system defaults)
load_dotenv()

# Configuration
PYTHON_EXE = sys.executable
SERVER_SCRIPT = "src.mcp_code_mode.executor_server" # Module path

async def run_agent_task(task_message: str, history: List, openai_key: str, gemini_key: str):
    """
    Connects to the MCP server using the user's provided keys.
    """
    if not openai_key and not gemini_key:
        return "⚠️ Error: Please provide an OpenAI or Gemini API Key in the settings."

    # Prepare environment for the subprocess
    # We inherit system env but override/add the user's keys
    env = dict(os.environ)
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key
    if gemini_key:
        env["GEMINI_API_KEY"] = gemini_key
    
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

# Sync wrapper for Gradio
def chat_wrapper(message, history, openai_key, gemini_key):
    return asyncio.run(run_agent_task(message, history, openai_key, gemini_key))

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
    
    chat_interface = gr.ChatInterface(
        fn=chat_wrapper,
        additional_inputs=[openai_input, gemini_input],
        examples=["Create a fibonacci function in python", "Analyze this repository"],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

### Step 3: Dockerfile for HF Spaces
Configure the container to run `app.py` and respect the port `7860`.

**File**: `docs/Dockerfile`
```dockerfile
FROM python:3.11-slim

# Set up user to avoid running as root (HF requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# Copy project files
COPY --chown=user . .

# Install dependencies
# Note: We install from the current directory to include the package itself if needed
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
```

### Step 4: Deployment Steps
1. **Create Space**: Go to Hugging Face -> New Space -> SDK: Docker.
2. **Set Secrets**: In Space Settings, add `OPENAI_API_KEY` (and `GEMINI_API_KEY` if used).
3. **Push Code**: Upload `app.py`, `Dockerfile`, `requirements.txt`, and the `src/` folder to the Space.

## 4. Next Steps
1. Create the files listed above in `docs/`.
2. Test locally by running `python docs/app.py`.
3. Once verified, deploy to Hugging Face.
