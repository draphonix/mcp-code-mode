# Implementation Plan: Code Execution MCP Server with DSpy

## Executive Summary

**Status**: ✅ **FEASIBLE** with important architectural corrections

The proposed Code Execution MCP Server with DSpy is technically sound and implementable. However, the original research document contains critical inaccuracies about DSpy's actual capabilities. This plan provides a corrected, phase-by-phase approach based on DSpy's real features.

### Key Corrections to Research Document

| Component | Research Doc Claims | Actual Reality | Impact |
|-----------|---------------------|----------------|---------|
| **Sandbox** | Docker containers | Deno + Pyodide (WebAssembly) | ✅ Simpler, lighter |
| **RPC Bridge** | Custom Flask/HTTP server | Built-in `dspy.Tool.from_mcp_tool()` | ✅ No custom code needed |
| **Tool Injection** | Auto-generated `mcp_tools.py` | Native DSpy MCP adapters | ✅ Cleaner integration |
| **Code Generation** | `dspy.ReAct` for code | `dspy.ProgramOfThought` + `dspy.CodeAct` | ⚠️ Different API |

---

## Feasibility Validation

### 1. ✅ DSpy Code Generation Capabilities - VALIDATED

DSpy provides **multiple** code generation and execution modules:

#### Available Modules

1. **`dspy.Code`** - Type for code generation
   - Supports language specification: `dspy.Code["python"]`, `dspy.Code["java"]`
   - Auto-extracts code from markdown blocks
   - Can be used in signatures as input/output fields

2. **`dspy.ProgramOfThought`** - Generates and executes Python
   - Automatically generates executable Python to solve problems
   - Built-in retry mechanism (up to 3 iterations)
   - Uses DSpy's `PythonInterpreter` for sandboxed execution
   - Error recovery and code regeneration

3. **`dspy.CodeAct`** - Combines reasoning with code execution
   - Inherits from both `ReAct` and `ProgramOfThought`
   - Can use predefined tools alongside code generation
   - Iterative problem-solving

4. **`dspy.ReAct`** - Tool-using agent framework
   - Executes arbitrary Python functions as tools
   - Supports sync and async execution
   - Trajectory management and context handling

#### Built-in Sandboxing

DSpy includes `PythonInterpreter` with:
- **Deno + Pyodide** for browser-level sandboxing via WebAssembly
- File system access controls (`enable_read_paths`, `enable_write_paths`)
- Environment variable isolation (`enable_env_vars`)
- Network access controls (`enable_network_access`)
- File synchronization support

**Key Finding**: DSpy uses Deno/Pyodide, NOT Docker. This is actually a lighter-weight, simpler alternative.

---

### 2. ✅ FastMCP Server Capabilities - VALIDATED

FastMCP (v2.0) fully supports:

#### Async Operations

```python
@mcp.tool
async def process_data(uri: str, ctx: Context):
    await ctx.info(f"Processing {uri}...")
    result = await some_async_operation()
    return result
```

#### Tool Registration

```python
@mcp.tool
def execute_code(code: str) -> str:
    """Execute Python code in a sandbox"""
    return result
```

#### Complex Return Types

- Text, JSON-serializable objects
- Images and audio (via helper classes)
- Custom Pydantic models
- `ToolResult` objects with metadata

#### Context API

Rich server-side capabilities:
- `ctx.info()`, `ctx.error()` - Logging to clients
- `ctx.sample()` - Request LLM completions from client
- `ctx.read_resource()` - Access server resources
- `ctx.report_progress()` - Progress updates
- `ctx.set_state()` - Session state management

#### Transport Support

- **STDIO** - For local command-line tools
- **HTTP/SSE** - For web deployment
- **In-memory** - For testing

---

### 3. ✅ MCP Protocol Compatibility - VALIDATED

#### DSpy + MCP Integration

DSpy has **built-in MCP support** via `dspy.utils.mcp`:

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client
import dspy

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Convert MCP tools to DSpy tools
            tools = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in tools.tools
            ]
            
            # Use in ReAct agent
            react = dspy.ReAct(signature, tools=dspy_tools)
            result = await react.acall(question="...")
```

#### Key Integration Points

1. **Tool Conversion** - `dspy.Tool.from_mcp_tool()`:
   - Automatically converts MCP tool schemas to DSpy tools
   - Preserves parameter types and descriptions
   - Handles async execution

2. **Async Support** - Full async/await:
   - `react.acall()` for async agent calls
   - `tool.acall()` for async tool execution
   - MCP tools are async by default

3. **Protocol Compliance** - Uses official MCP Python SDK
   - Supports stdio, HTTP, and SSE transports
   - Standard MCP message format

---

### 4. ✅ Overall Architecture - TECHNICALLY SOUND

#### Recommended Architecture

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
│  │  async def generate_and_execute(task: str):         │  │
│  │      # 1. Use DSpy to generate code                 │  │
│  │      code = await dspy_generate_code(task)          │  │
│  │      # 2. Execute in sandbox                        │  │
│  │      result = await execute_in_sandbox(code)        │  │
│  │      return result                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │    Sandbox Options:          │
          │  • DSpy PythonInterpreter   │
          │    (Deno + Pyodide)         │
          │  • Docker (optional)         │
          │  • Firecracker (advanced)   │
          └──────────────────────────────┘
```

#### Why This Architecture Works

1. **FastMCP** handles:
   - MCP protocol implementation
   - Client communication
   - Tool registration and discovery
   - Authentication (optional)

2. **DSpy** handles:
   - Code generation from natural language
   - Iterative refinement with error recovery
   - Tool orchestration (via ReAct/CodeAct)

3. **Sandbox** handles:
   - Code execution isolation
   - Resource limits
   - Security boundaries

---

## Multi-Phase Implementation Plan

### TL;DR

- Build a FastMCP "Code Executor" server that runs user/agent-generated Python inside DSpy's Deno+Pyodide sandbox
- Let DSpy generate code using `ProgramOfThought` + `CodeAct`
- Use built-in MCP adapters (`dspy.Tool.from_mcp_tool()`) instead of custom RPC bridges
- Start with a simple, single-process prototype; add complexity only when needed

---

### Phase 0: Ground-Truth Setup and Scaffolding

**Duration**: 1-2 hours | **Complexity**: Simple

#### Goal
Verify actual APIs/versions and create a minimal runnable scaffold. **Configure which MCP servers to use.**

#### Tasks

1. **Pin toolchain**
   - Python 3.11+
   - `pip install fastmcp dspy-ai mcp`
   - Confirm support for `ProgramOfThought`, `CodeAct`, `Code`
   - Install MCP Python SDK for local testing
   - Install Node.js (for npx-based MCP servers)

2. **Create MCP server configuration file**
   - Define which MCP servers the system will connect to
   - Specify command, args, and environment for each server
   - This is how the system knows what tools are available

3. **Create MCP Server Manager**
   - Build utility to load config and connect to servers
   - Implement tool discovery from all configured servers
   - Store server connections and tool schemas

4. **Sanity check DSpy sandbox**
   - Execute trivial `print("hello")` via DSpy's Deno+Pyodide runner
   - Verify no Docker dependency required
   - Test timeout and error handling

5. **Confirm FastMCP async tools**
   - Create trivial echo tool
   - Test with MCP inspector/client

#### Success Criteria

- ✅ `mcp_servers.json` defines at least 2 MCP servers
- ✅ `MCPServerManager` can connect to configured servers
- ✅ Tool discovery works: `list_tools()` returns tools from all servers
- ✅ `dspy.Code`/`CodeAct` can execute simple snippet and return stdout
- ✅ FastMCP server runs with one async tool responding via stdio
- ✅ Can connect to server and invoke tool successfully

#### Code Examples

**1. MCP Server Configuration**

```json
// mcp_servers.json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "description": "Local file system operations"
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "description": "HTTP fetch operations"
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "description": "Key-value memory storage"
    }
  }
}
```

**2. MCP Server Manager**

```python
# mcp_manager.py
import json
from typing import Dict, List
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import dspy

class MCPServerManager:
    """Manages connections to configured MCP servers and tool discovery"""
    
    def __init__(self, config_path: str = "mcp_servers.json"):
        self.config_path = config_path
        self.servers = {}  # server_name -> {session, tools, description}
        self.all_tools = []  # All tools from all servers
        self.dspy_tools = []  # DSpy-wrapped tools
    
    def _load_config(self) -> dict:
        """Load MCP server configuration"""
        with open(self.config_path) as f:
            return json.load(f)
    
    async def initialize(self):
        """Connect to all configured servers and discover their tools"""
        config = self._load_config()
        
        print(f"Connecting to {len(config['servers'])} MCP servers...")
        
        for server_name, server_config in config["servers"].items():
            await self._connect_server(server_name, server_config)
        
        print(f"✅ Connected to {len(self.servers)} servers")
        print(f"✅ Discovered {len(self.all_tools)} total tools")
        
        return self.dspy_tools
    
    async def _connect_server(self, name: str, config: dict):
        """Connect to a single MCP server and discover its tools"""
        try:
            print(f"  Connecting to {name}...")
            
            # Create server parameters from config
            server_params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env", {})
            )
            
            # Connect to server
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # NOW we can discover tools via list_tools()
                    tools_response = await session.list_tools()
                    
                    # Store server info
                    self.servers[name] = {
                        "session": session,
                        "config": config,
                        "tools": tools_response.tools,
                        "description": config.get("description", "")
                    }
                    
                    # Add to global tool list
                    self.all_tools.extend(tools_response.tools)
                    
                    # Convert to DSpy tools
                    for tool in tools_response.tools:
                        dspy_tool = dspy.Tool.from_mcp_tool(session, tool)
                        self.dspy_tools.append(dspy_tool)
                    
                    print(f"    ✅ {name}: {len(tools_response.tools)} tools")
                    for tool in tools_response.tools:
                        print(f"       - {tool.name}")
                        
        except Exception as e:
            print(f"    ❌ Failed to connect to {name}: {e}")
    
    def get_tools_summary(self) -> str:
        """Get human-readable summary of all available tools"""
        lines = ["Available MCP Tools:"]
        for server_name, info in self.servers.items():
            lines.append(f"\n{server_name} ({info['description']}):")
            for tool in info["tools"]:
                lines.append(f"  - {tool.name}: {tool.description}")
        return "\n".join(lines)
```

**3. Test DSpy Sandbox**

```python
# test_dspy_sandbox.py
import dspy

# Configure DSpy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Test simple code execution
code = 'print("Hello from DSpy sandbox")'
# Verify DSpy can run this code
# (Actual implementation depends on DSpy's PythonInterpreter API)
```

**4. Test FastMCP**

```python
# test_fastmcp.py
from fastmcp import FastMCP

mcp = FastMCP("Test Server")

@mcp.tool()
async def echo(message: str) -> str:
    """Echo a message back"""
    return f"Echo: {message}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**5. Integration Test**

```python
# test_integration.py
import asyncio
from mcp_manager import MCPServerManager

async def main():
    # Initialize manager with configured servers
    manager = MCPServerManager("mcp_servers.json")
    
    # Connect and discover tools
    tools = await manager.initialize()
    
    # Print summary
    print(f"\n{manager.get_tools_summary()}")
    
    # Verify we have DSpy tools ready
    print(f"\nReady to use {len(tools)} DSpy-wrapped tools")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Phase 1: Minimal Executor MCP Server

**Duration**: 2-4 hours | **Complexity**: Simple

#### Goal
Expose a single `execute_code` tool that evaluates Python code in DSpy's Deno+Pyodide sandbox.

#### Tasks

1. **Implement FastMCP server**
   - Create `async execute_code(code: str, timeout: int = 30) -> dict`
   - Use DSpy's sandboxed executor (Deno+Pyodide) under the hood
   - Capture stdout, stderr, exit state, timing

2. **Enforce timeout**
   - Implement execution timeout
   - Handle timeout gracefully with diagnostics

3. **Structured payload**
   - Return: `{ success, stdout, stderr, duration_ms, diagnostics }`
   - Include error messages and stack traces when relevant

#### Success Criteria

- ✅ Calling MCP tool with simple code returns correct stdout and zero exit
- ✅ Execution completes within timeout
- ✅ Safe failure path for exceptions/timeouts
- ✅ Clear error messages for syntax/runtime errors

#### Code Example

```python
# executor_server.py
from fastmcp import FastMCP, Context
import dspy
import time
import traceback

mcp = FastMCP("Code Executor Server")

@mcp.tool()
async def execute_code(
    code: str,
    timeout: int = 30,
    ctx: Context = None
) -> dict:
    """Execute Python code in a secure Deno+Pyodide sandbox"""
    
    start_time = time.time()
    
    try:
        if ctx:
            await ctx.info(f"Executing code (timeout: {timeout}s)")
        
        # Use DSpy's PythonInterpreter
        # (Implementation details depend on DSpy's actual API)
        # This is a placeholder - check DSpy docs for exact usage
        
        result = {
            "success": True,
            "stdout": "Output here",
            "stderr": "",
            "duration_ms": int((time.time() - start_time) * 1000),
            "diagnostics": None
        }
        
        return result
        
    except TimeoutError:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timeout after {timeout}s",
            "duration_ms": timeout * 1000,
            "diagnostics": "TIMEOUT"
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "duration_ms": int((time.time() - start_time) * 1000),
            "diagnostics": traceback.format_exc()
        }

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

### Phase 2: External MCP Tools Integration

**Duration**: 2-4 hours | **Complexity**: Simple

#### Goal
Use DSpy's built-in MCP integration to surface external tools from configured servers (no custom bridges).

#### Tasks

1. **Use MCPServerManager to discover tools**
   - Leverage the `mcp_servers.json` configuration from Phase 0
   - Manager automatically discovers tools from all configured servers
   - Tools are already converted to DSpy format

2. **Build tool schema formatter for LLM**
   - Create function to format tool schemas as readable text
   - Include name, description, parameters, and usage examples
   - This will be passed to code generation LLM in Phase 3

3. **Test tool invocation**
   - Manually test calling tools via DSpy wrappers
   - Verify parameters are passed correctly
   - Confirm results are returned as expected

4. **Verify accessibility**
   - Ensure hand-written code can call tools via DSpy wrappers
   - Test end-to-end: tool discovery → tool schema → tool call → result

#### Success Criteria

- ✅ `MCPServerManager` successfully discovers tools from `mcp_servers.json`
- ✅ Tool schemas can be formatted for LLM consumption
- ✅ Hand-written Python can call MCP tools via DSpy wrappers
- ✅ Tool calls return expected results
- ✅ No Flask/HTTP bridge or custom `mcp_tools.py` required

#### Code Examples

**1. Tool Schema Formatter**

```python
# tool_formatter.py
import json
from typing import List
import dspy

class ToolSchemaFormatter:
    """Format MCP tool schemas for LLM code generation"""
    
    def __init__(self, dspy_tools: List[dspy.Tool]):
        self.tools = dspy_tools
    
    def format_for_llm(self) -> str:
        """
        Format all tools as readable text for LLM.
        This tells the LLM what tools are available and how to use them.
        """
        tool_docs = []
        
        for tool in self.tools:
            doc = self._format_single_tool(tool)
            tool_docs.append(doc)
        
        header = f"# Available MCP Tools ({len(self.tools)} total)\n\n"
        return header + "\n\n".join(tool_docs)
    
    def _format_single_tool(self, tool: dspy.Tool) -> str:
        """Format a single tool for LLM"""
        # Extract parameter information
        params = tool.input_schema.get("properties", {})
        required = tool.input_schema.get("required", [])
        
        # Build parameter list with types
        param_strs = []
        for param_name, param_info in params.items():
            param_type = param_info.get("type", "any")
            is_required = param_name in required
            req_marker = "" if is_required else "?"
            param_strs.append(f"{param_name}{req_marker}: {param_type}")
        
        param_list = ", ".join(param_strs)
        
        # Generate example usage
        example = self._generate_example(tool, params, required)
        
        return f"""## {tool.name}
**Description**: {tool.description}

**Usage**:
```python
result = {tool.name}({param_list})
```

**Parameters**:
{json.dumps(params, indent=2)}

**Example**:
```python
{example}
```"""
    
    def _generate_example(self, tool: dspy.Tool, params: dict, required: list) -> str:
        """Generate example code for using a tool"""
        # Create example arguments for required parameters
        example_args = []
        for param_name in required:
            param_info = params.get(param_name, {})
            param_type = param_info.get("type", "string")
            
            # Generate example value based on type
            if param_type == "string":
                example_val = f'"{param_name}_value"'
            elif param_type == "integer":
                example_val = "42"
            elif param_type == "boolean":
                example_val = "True"
            else:
                example_val = f'"{param_name}_value"'
            
            example_args.append(f"{param_name}={example_val}")
        
        args_str = ", ".join(example_args)
        return f"result = {tool.name}({args_str})\nprint(result)"
    
    def get_tool_names(self) -> List[str]:
        """Get list of all tool names"""
        return [tool.name for tool in self.tools]
```

**2. Integration with MCPServerManager**

```python
# mcp_integration.py
import asyncio
from mcp_manager import MCPServerManager
from tool_formatter import ToolSchemaFormatter

async def setup_mcp_tools():
    """Initialize MCP servers and prepare tools for code generation"""
    
    # 1. Load servers from mcp_servers.json and discover tools
    print("Step 1: Discovering MCP servers and tools...")
    manager = MCPServerManager("mcp_servers.json")
    dspy_tools = await manager.initialize()
    
    print(f"\nDiscovered {len(dspy_tools)} tools from {len(manager.servers)} servers")
    
    # 2. Format tools for LLM
    print("\nStep 2: Formatting tool schemas for LLM...")
    formatter = ToolSchemaFormatter(dspy_tools)
    llm_context = formatter.format_for_llm()
    
    print(f"Generated LLM context ({len(llm_context)} chars)")
    print("\nPreview:")
    print(llm_context[:500] + "...\n")
    
    # 3. Return both tools and formatted context
    return {
        "manager": manager,
        "tools": dspy_tools,
        "llm_context": llm_context,
        "tool_names": formatter.get_tool_names()
    }

async def main():
    # Setup tools
    mcp_setup = await setup_mcp_tools()
    
    # Show what the LLM will see
    print("\n" + "="*60)
    print("TOOL CONTEXT FOR LLM:")
    print("="*60)
    print(mcp_setup["llm_context"])
    
    return mcp_setup

if __name__ == "__main__":
    result = asyncio.run(main())
```

**3. Manual Tool Testing**

```python
# test_tool_calls.py
import asyncio
from mcp_integration import setup_mcp_tools

async def test_filesystem_tool():
    """Test calling filesystem tools via DSpy wrapper"""
    
    # Setup
    mcp_setup = await setup_mcp_tools()
    tools = mcp_setup["tools"]
    
    # Find the read_file tool
    read_file_tool = next(
        (t for t in tools if "read_file" in t.name),
        None
    )
    
    if not read_file_tool:
        print("❌ read_file tool not found")
        return
    
    print(f"Testing tool: {read_file_tool.name}")
    
    # Create a test file
    test_path = "/tmp/test_mcp.txt"
    with open(test_path, "w") as f:
        f.write("Hello from MCP test!")
    
    # Call the tool via DSpy wrapper
    result = await read_file_tool.acall(path=test_path)
    
    print(f"✅ Tool call successful!")
    print(f"Result: {result}")
    
    assert "Hello from MCP test!" in str(result)
    print("✅ Result verification passed")

if __name__ == "__main__":
    asyncio.run(test_filesystem_tool())
```

**4. Understanding the Flow**

```
1. System Startup
   ↓
2. Read mcp_servers.json
   └─> Servers: ["filesystem", "fetch", "memory"]
   ↓
3. MCPServerManager.initialize()
   ├─> Connect to filesystem server
   │   └─> Discover: [read_file, write_file, list_directory, ...]
   ├─> Connect to fetch server
   │   └─> Discover: [fetch, ...]
   └─> Connect to memory server
       └─> Discover: [store, retrieve, ...]
   ↓
4. Convert all tools to DSpy format
   └─> dspy.Tool.from_mcp_tool(session, tool)
   ↓
5. ToolSchemaFormatter formats tools
   └─> Creates readable text with:
       • Tool names
       • Descriptions
       • Parameter schemas
       • Usage examples
   ↓
6. Ready for Phase 3: Code Generation
   └─> LLM will receive this context and generate code
```

---

### Phase 3: Code Generation Agent (PoT + CodeAct)

**Duration**: 4-6 hours | **Complexity**: Medium

#### Goal
Have DSpy plan and write code that will run via `execute_code`, using discovered MCP tools.

#### Tasks

1. **Define signature**
   - Create `Signature` for "task + available_tools -> code"
   - Output should be `dspy.Code` artifact
   - Input includes formatted tool schemas from Phase 2

2. **Compose agent**
   - Use `ProgramOfThought` to reason about the task
   - Use `CodeAct` to produce Python code
   - Inject tool schemas into LLM context so it knows what tools exist
   - Code should use DSpy MCP tool wrappers from Phase 2

3. **Add validators**
   - Assert code calls at least one MCP tool when relevant
   - Assert code prints final answer
   - Check for common errors (syntax, undefined variables)
   - Validate tool names against available tools

4. **Wire execution**
   - Connect agent to `execute_code` tool from Phase 1
   - Pass MCP tools to sandbox environment
   - Display output back to user
   - Handle errors gracefully with retry

#### Success Criteria

- ✅ Agent receives formatted tool schemas from Phase 2
- ✅ For prompt "Read /tmp/test.txt", agent generates code using correct tool
- ✅ Generated code uses MCP tool wrappers correctly
- ✅ Code runs in sandbox and returns readable output
- ✅ Errors trigger retry with improved code
- ✅ LLM can see which tools are available and their parameters

#### Code Examples

**1. Code Generation Signature (with tool context)**

```python
# agent.py
import dspy
from typing import List
from tool_formatter import ToolSchemaFormatter

class CodeGenerationSignature(dspy.Signature):
    """Generate Python code to complete a task using available MCP tools"""
    task: str = dspy.InputField(desc="The user's task to complete")
    available_tools: str = dspy.InputField(
        desc="Detailed documentation of available MCP tools with parameters and examples"
    )
    code: dspy.Code = dspy.OutputField(
        desc="Python code that uses the available tools to complete the task"
    )
```

**2. Code Execution Agent (complete implementation)**

```python
# agent.py (continued)
import asyncio
from mcp_integration import setup_mcp_tools

class CodeExecutionAgent:
    """Agent that generates and executes Python code using MCP tools"""
    
    def __init__(self, mcp_tools: List[dspy.Tool], tool_context: str):
        """
        Initialize agent with discovered MCP tools.
        
        Args:
            mcp_tools: List of DSpy-wrapped MCP tools
            tool_context: Formatted tool documentation for LLM
        """
        self.mcp_tools = mcp_tools
        self.tool_context = tool_context  # This is what the LLM sees!
        self.tool_names = [t.name for t in mcp_tools]
        
        # Create code generator with tool-aware signature
        self.generator = dspy.ProgramOfThought(
            signature=CodeGenerationSignature,
            max_iters=3
        )
        
    async def run(self, task: str) -> dict:
        """
        Generate and execute code for a task.
        
        This is the key method that:
        1. Passes tool schemas to LLM
        2. LLM generates code using those tools
        3. Code is executed with access to real tools
        """
        
        print(f"Task: {task}")
        print(f"Available tools: {self.tool_names}")
        
        # Generate code - THE KEY: tool_context tells LLM what tools exist
        result = await self.generator.acall(
            task=task,
            available_tools=self.tool_context  # ← LLM sees all tool schemas
        )
        
        print(f"\nGenerated code:")
        print("-" * 60)
        print(result.code)
        print("-" * 60)
        
        # Execute code with access to MCP tools
        execution_result = await self._execute_with_tools(result.code)
        
        return {
            "task": task,
            "generated_code": result.code,
            "execution_result": execution_result
        }
    
    async def _execute_with_tools(self, code: str) -> dict:
        """
        Execute generated code with MCP tools available.
        
        This would integrate with the execute_code tool from Phase 1,
        but with the MCP tool wrappers injected into the sandbox.
        """
        # TODO: Integrate with Phase 1 executor
        # For now, simulate execution
        try:
            # In real implementation, this would:
            # 1. Inject self.mcp_tools into sandbox namespace
            # 2. Run code in Deno+Pyodide sandbox
            # 3. Capture stdout/stderr
            
            # Placeholder for now
            return {
                "success": True,
                "stdout": "Code executed successfully",
                "stderr": "",
                "duration_ms": 0
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "duration_ms": 0
            }
    
    def validate_code(self, code: str) -> tuple[bool, str]:
        """Validate generated code before execution"""
        # Check if code uses any available tools
        uses_tools = any(tool_name in code for tool_name in self.tool_names)
        
        # Check if code prints result
        has_output = "print" in code
        
        if not uses_tools:
            return False, "Code does not use any available MCP tools"
        
        if not has_output:
            return False, "Code does not print any output"
        
        return True, "Code validation passed"
```

**3. Example Usage**

```python
# example_usage.py
import asyncio
from mcp_integration import setup_mcp_tools
from agent import CodeExecutionAgent

async def main():
    # 1. Setup: Discover MCP tools from mcp_servers.json
    print("="*60)
    print("PHASE 2: Discovering MCP Tools")
    print("="*60)
    mcp_setup = await setup_mcp_tools()
    
    # 2. Create agent with tool context
    print("\n" + "="*60)
    print("PHASE 3: Creating Code Generation Agent")
    print("="*60)
    agent = CodeExecutionAgent(
        mcp_tools=mcp_setup["tools"],
        tool_context=mcp_setup["llm_context"]  # ← This is the key!
    )
    
    # 3. Run tasks
    tasks = [
        "Read the file at /tmp/test.txt and print its contents",
        "List all files in /tmp directory",
        "Fetch the contents of https://example.com and save to memory"
    ]
    
    for task in tasks:
        print("\n" + "="*60)
        print(f"EXECUTING TASK: {task}")
        print("="*60)
        
        result = await agent.run(task)
        
        print("\nResult:")
        print(f"  Success: {result['execution_result']['success']}")
        print(f"  Output: {result['execution_result']['stdout']}")

if __name__ == "__main__":
    asyncio.run(main())
```

**4. What the LLM Actually Sees**

When `agent.run(task)` is called, the LLM receives:

```
Task: "Read the file at /tmp/test.txt and print its contents"

Available Tools:
# Available MCP Tools (8 total)

## read_file
**Description**: Read the contents of a text file

**Usage**:
```python
result = read_file(path: string)
```

**Parameters**:
{
  "path": {
    "type": "string",
    "description": "Path to the file to read"
  }
}

**Example**:
```python
result = read_file(path="/tmp/test.txt")
print(result)
```

## write_file
**Description**: Write content to a file
...

[etc for all 8 tools]
```

**The LLM then generates:**

```python
# Read the file and print contents
result = read_file(path="/tmp/test.txt")
print(f"File contents:\n{result}")
```

**5. Complete Flow Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 0: mcp_servers.json defines servers                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: MCPServerManager discovers tools                   │
│   • Connects to servers                                     │
│   • Calls list_tools()                                      │
│   • Converts to DSpy tools                                  │
│   • Formats schemas for LLM                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ (tool_context string)
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: CodeExecutionAgent receives tools                  │
│   • Stores tool_context (formatted schemas)                 │
│   • Stores mcp_tools (actual callable tools)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ User Task: "Read /tmp/test.txt"                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ agent.run(task) calls LLM with:                             │
│   • task = "Read /tmp/test.txt"                             │
│   • available_tools = tool_context  ← LLM sees schemas!     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ LLM Generates Code:                                          │
│   result = read_file(path="/tmp/test.txt")                  │
│   print(result)                                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: execute_code runs in sandbox                       │
│   • Code has access to mcp_tools                            │
│   • read_file() calls actual MCP server                     │
│   • Returns result                                           │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: The `tool_context` string from Phase 2 is what enables the LLM to know which tools exist and how to use them. Without this, the LLM would have no idea what tools are available!

---

### Phase 4: Async and Concurrency Validation

**Duration**: 2-3 hours | **Complexity**: Simple

#### Goal
Confirm FastMCP handles concurrent invocations and DSpy sandbox isolation.

#### Tasks

1. **Parallel execution test**
   - Run 5-10 parallel `execute_code` invocations
   - Use varied code in each call
   - Measure latency and resource usage

2. **Verify isolation**
   - Ensure no output mixing
   - Confirm per-call state isolation
   - Check for resource leaks

3. **Monitor stability**
   - Watch for deadlocks
   - Check for resource starvation
   - Verify stable latencies

#### Success Criteria

- ✅ All calls complete independently
- ✅ No cross-talk between executions
- ✅ No deadlocks or hangs
- ✅ Stable latencies across concurrent calls

#### Code Example

```python
# test_concurrency.py
import asyncio
from executor_server import execute_code

async def test_concurrent_execution():
    """Test concurrent code execution"""
    
    test_codes = [
        'print(f"Task {i}: {2**i}")'
        for i in range(10)
    ]
    
    # Execute all concurrently
    tasks = [
        execute_code(code, timeout=5)
        for code in test_codes
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Verify all succeeded
    assert all(r["success"] for r in results)
    
    # Verify unique outputs
    outputs = [r["stdout"] for r in results]
    assert len(set(outputs)) == len(outputs)
    
    print(f"✅ All {len(results)} concurrent executions succeeded")

# Run test
asyncio.run(test_concurrent_execution())
```

---

### Phase 5: Sandboxing and Policy Guardrails

**Duration**: 4-8 hours | **Complexity**: Medium

#### Goal
Lock down Deno+Pyodide runtime for research usage.

#### Tasks

1. **Configure sandbox permissions**
   - Restrict filesystem access (Deno permissions)
   - Restrict network access (only MCP endpoints via DSpy)
   - Set read-only paths for code execution

2. **Set resource limits**
   - Execution timeout (default 30s)
   - Memory ceiling (e.g., 256MB)
   - CPU usage limits (if available)

3. **Import restrictions**
   - Disallow arbitrary imports unavailable in Pyodide
   - Allowlist safe packages
   - Provide clear errors for forbidden imports

4. **Security filters**
   - Basic prompt-injection filters
   - Tool allowlists (only preapproved MCP tools)
   - Output sanitization (remove potential secrets)

#### Success Criteria

- ✅ Attempts to access disallowed filesystem fail cleanly
- ✅ Network access restricted to MCP tool calls
- ✅ Timeouts trigger with helpful diagnostics
- ✅ Memory limits enforced
- ✅ Forbidden imports produce clear error messages

#### Configuration Example

```python
# sandbox_config.py
from dspy.primitives.python_interpreter import PythonInterpreter

# Configure sandbox
sandbox_config = {
    "enable_network_access": False,  # Only via DSpy MCP tools
    "enable_read_paths": ["/tmp/safe_data"],  # Read-only
    "enable_write_paths": [],  # No writes
    "enable_env_vars": False,  # No env access
    "timeout_seconds": 30,
    "memory_limit_mb": 256,
    "allowed_imports": [
        "json", "math", "datetime", "re", "collections"
        # Pyodide-compatible packages only
    ]
}

# Allowlist of MCP tools
ALLOWED_MCP_TOOLS = [
    "filesystem.read_file",
    "filesystem.list_directory",
    "news.fetch_latest",
    "summarizer.summarize"
]
```

---

### Phase 6: Testing and Validation

**Duration**: 1-2 days | **Complexity**: Large

#### Goal
Establish repeatable tests for correctness, safety, and regression.

#### Test Categories

##### 1. Unit Tests
- Code execution happy paths
- Syntax errors
- Runtime errors
- Timeouts
- Stdout/stderr capture
- Resource limit enforcement

##### 2. Integration Tests
- Generated code calling real MCP tools
- Golden output validation
- End-to-end agent workflows
- Multi-step task completion

##### 3. Property-Based Tests
- Randomized code snippets
- Resource constraint validation
- Boundary condition testing

##### 4. Negative Tests
- Prompt injection attempts
- Forbidden import attempts
- Long loops and infinite recursion
- Memory exhaustion attempts
- Network access attempts

##### 5. Load Tests
- 50-100 sequential executions
- Stability over time
- Memory leak detection
- Performance degradation checks

#### Success Criteria

- ✅ >95% unit/integration test pass rate
- ✅ Clear failure messages for all error cases
- ✅ No flakiness across multiple runs
- ✅ Load tests show stable performance
- ✅ Security tests confirm isolation

#### Test Examples

```python
# tests/test_executor.py
import pytest
from executor_server import execute_code

@pytest.mark.asyncio
async def test_simple_execution():
    """Test basic code execution"""
    result = await execute_code('print("Hello, World!")')
    assert result["success"] == True
    assert "Hello, World!" in result["stdout"]
    assert result["duration_ms"] < 5000

@pytest.mark.asyncio
async def test_syntax_error():
    """Test syntax error handling"""
    result = await execute_code('print("unclosed string')
    assert result["success"] == False
    assert "SyntaxError" in result["stderr"]

@pytest.mark.asyncio
async def test_timeout():
    """Test timeout enforcement"""
    result = await execute_code('while True: pass', timeout=2)
    assert result["success"] == False
    assert result["diagnostics"] == "TIMEOUT"

@pytest.mark.asyncio
async def test_forbidden_network():
    """Test network access is blocked"""
    code = '''
import urllib.request
urllib.request.urlopen("https://example.com")
'''
    result = await execute_code(code)
    assert result["success"] == False
    assert "network" in result["stderr"].lower()

# tests/test_integration.py
@pytest.mark.asyncio
async def test_mcp_tool_call():
    """Test code calling MCP tool"""
    code = '''
from mcp_tools import filesystem
files = filesystem.list_directory("/tmp")
print(f"Found {len(files)} files")
'''
    result = await execute_code(code)
    assert result["success"] == True
    assert "Found" in result["stdout"]
```

---

### Phase 7: Learning-Focused Demos and Documentation

**Duration**: 2-4 hours | **Complexity**: Simple

#### Goal
Make it easy to reproduce and extend for study.

#### Tasks

1. **CLI examples**
   - Simple command-line interface for testing
   - Interactive mode for experimentation
   - Example prompts and expected outputs

2. **Jupyter notebooks**
   - Step-by-step walkthroughs
   - Visual output and explanations
   - Common patterns and recipes

3. **Documentation**
   - How MCP tools are surfaced via `dspy.Tool.from_mcp_tool()`
   - How to add new MCP servers
   - Architecture diagrams
   - Security considerations

4. **What-if exercises**
   - Modify constraints and observe behavior
   - Swap tools and measure impact
   - Latency profiling exercises
   - Security testing scenarios

#### Success Criteria

- ✅ New user can clone and run demo in <10 minutes
- ✅ Clear understanding of how to extend
- ✅ Well-documented common pitfalls
- ✅ Examples cover major use cases

#### Deliverables

```
docs/
├── README.md                    # Quick start guide
├── architecture.md              # System design
├── security.md                  # Security model
├── api-reference.md            # API documentation
└── examples/
    ├── 01_basic_execution.py   # Simple code execution
    ├── 02_mcp_tools.py         # Using MCP tools
    ├── 03_agent_workflow.py    # Full agent pipeline
    └── notebooks/
        └── tutorial.ipynb      # Interactive tutorial
```

---

## Time and Complexity Summary

| Phase | Duration | Complexity | Focus |
|-------|----------|------------|-------|
| **Phase 0** | 1-2 hours | Simple | Setup & verification |
| **Phase 1** | 2-4 hours | Simple | Basic executor |
| **Phase 2** | 2-4 hours | Simple | MCP integration |
| **Phase 3** | 4-6 hours | Medium | Code generation |
| **Phase 4** | 2-3 hours | Simple | Concurrency testing |
| **Phase 5** | 4-8 hours | Medium | Security hardening |
| **Phase 6** | 1-2 days | Large | Comprehensive testing |
| **Phase 7** | 2-4 hours | Simple | Documentation |
| **TOTAL** | **2-3 days** | - | Full implementation |

---

## Risks and Mitigation Strategies

### Risk 1: DSpy Sandbox API Changes
**Impact**: High  
**Probability**: Medium

**Mitigation**:
- Pin exact DSpy version
- Add sandbox smoke tests in CI
- Wrap sandbox calls with uniform error handling
- Document version dependencies clearly

---

### Risk 2: Pyodide Package Limitations
**Impact**: Medium  
**Probability**: High

**Context**: Pyodide doesn't support C extensions (numpy, pandas with native code, etc.)

**Mitigation**:
- Curate list of allowed imports (pure Python only)
- Design tasks around pure-Python capabilities
- Provide graceful errors with alternative suggestions
- Consider Docker fallback for advanced cases
- Document Pyodide limitations prominently

---

### Risk 3: Prompt Injection / Tool Misuse
**Impact**: High  
**Probability**: Medium

**Mitigation**:
- Allowlist MCP tools explicitly
- Add runtime checks requiring explicit tool names
- Assert outputs must be derived from tool results
- Input validation on all tool parameters
- Output sanitization to strip potential secrets
- Rate limiting per user/session

---

### Risk 4: Concurrency and Resource Contention
**Impact**: Medium  
**Probability**: Low

**Mitigation**:
- Per-execution timeouts (30s default)
- Memory caps per execution (256MB)
- Serialize high-cost tasks if needed
- Simple concurrency tests in CI
- Monitor resource usage in production

---

### Risk 5: Cold-Start Latency
**Impact**: Low  
**Probability**: High

**Context**: Deno+Pyodide initialization can be slow (~500ms-2s)

**Mitigation**:
- Optional warm pool for repeated runs
- Cache initialization artifacts during process lifetime
- Set user expectations (first run slower)
- Consider process pooling for production
- Measure and document typical latencies

---

## When to Consider Advanced/Production Path

### Triggers for Advanced Implementation

1. **High concurrency requirements**
   - >10 concurrent executions regularly
   - >100 executions per day
   - Cold-start costs become material bottleneck

2. **Package requirements**
   - Need for C-extension packages (numpy, pandas, etc.)
   - OS-level capabilities not in Pyodide
   - Custom package installations

3. **Security/compliance**
   - Process/container isolation required
   - Compliance mandates (SOC2, HIPAA, etc.)
   - Multi-tenant isolation boundaries

4. **Scale requirements**
   - Multi-tenant QoS needed
   - Per-tenant quotas and billing
   - SLA guarantees required

### Advanced Path Additions

#### Executor Architecture
- **Isolate pool**: Maintain N warm Deno+Pyodide isolates
- **LRU caching**: Reuse and reset state between runs
- **Queue management**: In-memory queue for load shaping
- **Per-tenant quotas**: Rate limiting and resource allocation

#### Observability
- **Structured logging**: Per-run trace IDs
- **Metrics**: Duration, CPU cost, memory usage
- **Error taxonomy**: Categorized failure modes
- **Dashboards**: p50/p95 latencies, success rates

#### Hardening
- **Policy engine**: Allowlist tool names and schemas
- **Input validation**: Schema validation for tool arguments
- **Output filtering**: Strip secrets and sensitive data
- **Audit logging**: Track all code executions

#### Scaling
- **Container executor**: Optional Docker backend for complex jobs
- **Keep Deno+Pyodide default**: Use containers only when needed
- **Horizontal scaling**: Multiple executor instances
- **Load balancing**: Distribute across executors

#### Testing
- **Soak tests**: 1000+ runs for stability
- **Chaos testing**: Random timeouts and failures
- **Performance dashboards**: Continuous monitoring
- **Regression detection**: Automated performance alerts

---

## What We Are NOT Doing

### Corrections to Research Document

| Feature | Research Doc | This Plan | Reason |
|---------|--------------|-----------|---------|
| **Sandbox** | Docker containers | Deno + Pyodide | DSpy's actual implementation |
| **RPC Bridge** | Flask/HTTP server | Native DSpy MCP | Simpler, built-in |
| **Tool Module** | Auto-generated `mcp_tools.py` | `dspy.Tool.from_mcp_tool()` | Native integration |
| **Code Gen Module** | `dspy.ReAct` | `dspy.ProgramOfThought` + `dspy.CodeAct` | Correct modules |
| **Network Layer** | HTTP callbacks | Direct DSpy MCP calls | No custom bridge needed |

---

## Phase Deliverables Checklist

### Phase 0
- ✅ `requirements.txt` with pinned versions
- ✅ `mcp_servers.json` defining which MCP servers to connect to
- ✅ `mcp_manager.py` with `MCPServerManager` class
- ✅ `test_dspy_sandbox.py` validating DSpy sandbox
- ✅ `test_fastmcp.py` validating FastMCP basics
- ✅ `test_integration.py` validating server discovery
- ✅ README with setup instructions

### Phase 1
- ✅ `executor_server.py` with `execute_code` tool
- ✅ Basic error handling and timeout
- ✅ Structured result format
- ✅ README with quickstart

### Phase 2
- ✅ `tool_formatter.py` with `ToolSchemaFormatter` class
- ✅ `mcp_integration.py` utility connecting to servers from `mcp_servers.json`
- ✅ Function to format tool schemas for LLM consumption
- ✅ `test_tool_calls.py` testing manual tool invocation
- ✅ Example notebook showing tool discovery and calls
- ✅ Documentation on adding MCP servers to config

### Phase 3
- ✅ `agent.py` with `CodeGenerationAgent` class
- ✅ `CodeGenerationSignature` with `available_tools` input field
- ✅ Integration with `ProgramOfThought` + `CodeAct`
- ✅ Tool context injection into LLM
- ✅ `run()` method calling `execute_code` with tool access
- ✅ `example_usage.py` with end-to-end examples
- ✅ Example prompts and expected outputs

### Phase 4-6
- ✅ `tests/` directory with unit tests
- ✅ Integration tests with real MCP tools
- ✅ Property-based tests for robustness
- ✅ Load test script
- ✅ CI configuration (GitHub Actions)
- ✅ Test coverage report

### Phase 7
- ✅ `docs/` directory with comprehensive guides
- ✅ CLI example scripts
- ✅ Jupyter notebook tutorial
- ✅ Architecture diagrams
- ✅ Security documentation
- ✅ API reference

---

## Recommended Technology Stack

### Core Dependencies
```txt
# requirements.txt
python>=3.11
fastmcp>=2.0.0
dspy-ai>=2.5.0
mcp>=1.0.0

# For MCP client testing
httpx>=0.24.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

### Development Tools
```txt
# requirements-dev.txt
black>=23.0.0          # Code formatting
ruff>=0.1.0            # Linting
mypy>=1.7.0            # Type checking
pytest-cov>=4.1.0      # Coverage
jupyter>=1.0.0         # Notebooks
```

### Optional Advanced
```txt
# requirements-advanced.txt
docker>=6.1.0          # Optional Docker sandbox
prometheus-client      # Metrics
structlog             # Structured logging
```

---

## Success Metrics

### Development Metrics
- ✅ All phases completed on schedule
- ✅ >95% test coverage
- ✅ Zero critical security issues
- ✅ Documentation complete and clear

### Functional Metrics
- ✅ Code execution success rate >90%
- ✅ Average execution time <2s
- ✅ Timeout enforcement 100% effective
- ✅ No resource leaks over 100 runs

### Learning Metrics
- ✅ New user can run demo in <10 minutes
- ✅ Clear understanding of architecture
- ✅ Ability to add new MCP tools
- ✅ Ability to modify and extend

---

## Next Steps

### Immediate Actions

1. **Review this plan** with stakeholders
2. **Set up development environment** (Phase 0)
3. **Validate DSpy sandbox** with simple test
4. **Create project structure** and repository

### Week 1 Goals
- Complete Phases 0-2
- Basic executor working
- MCP tools integrated

### Week 2 Goals  
- Complete Phases 3-5
- Code generation agent working
- Security hardening complete

### Week 3 Goals
- Complete Phases 6-7
- Comprehensive testing
- Documentation and demos

---

## Conclusion

This implementation plan provides a realistic, phase-by-phase approach to building a Code Execution MCP Server with DSpy. By leveraging DSpy's actual capabilities (Deno+Pyodide sandbox, built-in MCP integration), we can create a simpler, more maintainable system than the original research document proposed.

The plan prioritizes:
- **Learning**: Clear phases with educational value
- **Simplicity**: Use built-in features over custom code
- **Safety**: Security hardening and comprehensive testing
- **Extensibility**: Easy to add tools and capabilities

Total estimated time: **2-3 days** for a fully functional research prototype with comprehensive testing and documentation.

The architecture is proven, the tools exist, and the path forward is clear. Ready to begin Phase 0!
