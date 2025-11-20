# MCP Code Mode

Prototype implementation for the Code Execution MCP Server with DSpy. The "Code Execution with MCP" architecture combines the strengths of Large Language Models at code generation with the Model Context Protocol for tool integration. This system enables an AI agent to write Python code that runs in an isolated sandbox while seamlessly calling external MCP tools.

## Project Status
✅ **Phase 0-2 Completed**: Core executor, tool discovery, and formatting are implemented.
🚧 **Phase 3 Active**: Building the DSpy Code Generation Agent.
See [**Roadmap**](docs/ROADMAP.md) for details.

## Quick Start

### 1. Installation
Requires Python 3.11+ and Node.js 20+.

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .[dev]

# Install Node.js dependencies for reference servers
npm install -g npm@latest
```

### 2. Configuration
Copy the example environment file and configure secrets:
```bash
cp .env.example .env
```

Configure your MCP servers in `mcp_servers.json`:
```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/your-working-folder"],
      "description": "Local file system operations"
    }
  }
}
```

### 3. Running the Server
Launch the Code Execution MCP server:
```bash
python -m mcp_code_mode.executor_server
```

## Development Commands

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests |
| `ruff check .` | Lint the codebase |
| `black .` | Format the codebase |
| `mypy src` | Type check the source |
| `python scripts/test_dspy_sandbox.py` | Sanity check the sandbox |

## Sandbox Guardrails

The system enforces policies before code execution:
- **Limits**: 8k characters / 400 lines max.
- **Imports**: Allowlist only (`json`, `math`, `re`, `datetime`, etc.).
- **Tokens**: Disallows potentially dangerous tokens (`subprocess`, `exec`, `eval`).

Violations return a `POLICY_VIOLATION` error.

## Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   MCP Client (Claude, etc.)                  │
└────────────────────────┬────────────────────────────────────┘
                         │ MCP Protocol (stdio/HTTP/SSE)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastMCP Server                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  @mcp.tool                                           │  │
│  │  async def execute_code(code: str):                 │  │
│  │      # 1. Execute in DSpy sandbox                   │  │
│  │      result = await sandbox.run(code)               │  │
│  │      return result                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │    Sandbox Engine:           │
          │  • DSpy PythonInterpreter   │
          │    (Deno + Pyodide)         │
          └──────────────────────────────┘
```

### Why Code Mode?

Traditional MCP implementations face critical challenges:
1. **Context Window Bloat**: Every tool definition consumes tokens, limiting scalability.
2. **Token Cost**: Multiple back-and-forth tool calls are expensive.
3. **Latency**: Sequential tool invocations create cumulative delays.
4. **Composability**: Complex workflows require many discrete steps.

Code Mode addresses these by leveraging what LLMs excel at: writing code. Rather than making multiple tool calls, the agent writes a Python script that orchestrates all necessary operations internally.

### Core Components

1. **The Executor Server (FastMCP)** (`src/mcp_code_mode/executor_server.py`)
   The server exposes an `execute_code` tool backed by DSpy's sandboxed Python interpreter. Uses `fastmcp` to handle the MCP protocol and `dspy` for execution.

2. **Configuration-Driven Discovery** (`mcp_servers.json`)
   The system uses `mcp_servers.json` to explicitly configure which MCP servers to connect to. Loaded by `src/mcp_code_mode/mcp_manager.py`.

3. **Tool Schema Formatting** (`src/mcp_code_mode/tool_formatter.py`)
   Formats discovered MCP tools into readable documentation that gets passed to the code generation LLM, so it knows what tools exist.

4. **Context Injection**
   The formatted tool schemas are passed as an input field to the LLM. The LLM knows tool names, parameters, and usage examples *before* it writes the code.

### Information Flow

```
1. mcp_servers.json (Defines servers)
   ↓
2. MCPServerManager.initialize()
   ├─ Connect to configured servers
   ├─ Call list_tools() on each
   └─ Convert to DSpy tools
   ↓
3. ToolSchemaFormatter.format_for_llm()
   └─ Creates readable documentation
   ↓
4. CodeExecutionAgent
   └─ Stores both callable tools and schemas
   ↓
5. Agent Generation
   └─ Passes tool_context to LLM
   ↓
6. Code Execution
   └─ Code runs in sandbox, calling actual tools via MCP
```

### Design Evolution

| Component | Original Research | Current Implementation | Reason |
|-----------|-------------------|------------------------|--------|
| **Sandbox** | Docker containers | DSpy PythonInterpreter | Simpler, lighter, WebAssembly-based |
| **Bridge** | Flask/HTTP RPC | Direct MCP Connection | No custom server needed |
| **Discovery**| Unclear | `mcp_servers.json` | Explicit configuration |
| **Tools** | Auto-generated `mcp_tools.py` | `dspy.Tool.from_mcp_tool` | Native DSpy integration |

### Troubleshooting

**Timeout Issues**:
If the interpreter times out, it may enter a bad state. Currently, the best fix is to restart the server or reconnect the client to get a fresh interpreter instance.

**Missing Tools**:
Ensure `mcp_servers.json` paths are correct and that you have run `npm install` if using Node-based servers.

## References
- [DSpy Documentation](https://dspy.ai)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [FastMCP](https://github.com/jlowin/fastmcp)
# mcp-code-mode
