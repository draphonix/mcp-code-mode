# Implementation Plan Updates Summary

## What Was Updated

The [implementation-plan.md](file:///Users/themrb/Documents/personal/mcp-code-mode/docs/implementation-plan.md) has been updated to address the critical question: **"How does the system know which MCP servers and tools are available?"**

---

## Key Additions

### 1. Phase 0: MCP Server Configuration

**New Addition: `mcp_servers.json`**

This configuration file explicitly defines which MCP servers the system will connect to:

```json
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
    }
  }
}
```

**New Component: `MCPServerManager`**

A Python class that:
- Loads `mcp_servers.json`
- Connects to each configured server
- Calls `session.list_tools()` to discover available tools
- Converts MCP tools to DSpy tools using `dspy.Tool.from_mcp_tool()`
- Stores all tools for use in code generation

**Key Code:**
```python
class MCPServerManager:
    async def initialize(self):
        config = self._load_config()
        for server_name, server_config in config["servers"].items():
            # Connect to server
            server_params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", [])
            )
            # NOW we can call list_tools()
            tools_response = await session.list_tools()
            # Convert to DSpy tools
            for tool in tools_response.tools:
                dspy_tool = dspy.Tool.from_mcp_tool(session, tool)
```

---

### 2. Phase 2: Tool Schema Formatting for LLM

**New Component: `ToolSchemaFormatter`**

This class solves the problem: **"How does the LLM know what tools exist?"**

It formats discovered MCP tools into readable documentation that gets passed to the code generation LLM:

```python
class ToolSchemaFormatter:
    def format_for_llm(self) -> str:
        """Format all tools as readable text for LLM"""
        # Returns formatted documentation with:
        # - Tool names
        # - Descriptions
        # - Parameter schemas
        # - Usage examples
```

**What the LLM Sees:**
```markdown
# Available MCP Tools (8 total)

## read_file
**Description**: Read the contents of a text file

**Usage**:
result = read_file(path: string)

**Parameters**:
{
  "path": {
    "type": "string",
    "description": "Path to the file to read"
  }
}

**Example**:
result = read_file(path="/tmp/test.txt")
print(result)
```

**Updated Flow:**
```
1. MCPServerManager discovers tools from mcp_servers.json
   ↓
2. ToolSchemaFormatter formats tools for LLM
   ↓
3. Formatted schemas passed to CodeExecutionAgent
   ↓
4. Agent injects schemas into LLM context
   ↓
5. LLM generates code using known tools
```

---

### 3. Phase 3: Tool Context Injection

**Updated: `CodeExecutionAgent`**

The agent now explicitly receives and uses tool schemas:

```python
class CodeExecutionAgent:
    def __init__(self, mcp_tools: List[dspy.Tool], tool_context: str):
        self.mcp_tools = mcp_tools          # Actual callable tools
        self.tool_context = tool_context     # ← Formatted schemas for LLM
    
    async def run(self, task: str):
        # THE KEY: Pass tool schemas to LLM
        result = await self.generator.acall(
            task=task,
            available_tools=self.tool_context  # ← LLM sees all tool schemas
        )
```

**Updated Signature:**
```python
class CodeGenerationSignature(dspy.Signature):
    task: str = dspy.InputField(desc="The user's task to complete")
    available_tools: str = dspy.InputField(
        desc="Detailed documentation of available MCP tools"  # ← NEW
    )
    code: dspy.Code = dspy.OutputField(...)
```

**Complete Example Usage:**
```python
# 1. Discover tools from mcp_servers.json
mcp_setup = await setup_mcp_tools()

# 2. Create agent with tool context
agent = CodeExecutionAgent(
    mcp_tools=mcp_setup["tools"],
    tool_context=mcp_setup["llm_context"]  # ← This is what LLM sees!
)

# 3. Run tasks - LLM knows what tools are available
result = await agent.run("Read /tmp/test.txt")
```

---

## Complete Information Flow

### Before Updates (Missing Links)
```
??? → MCP Servers → list_tools() → Tools → ??? → Code Generation
```

**Problems:**
- How do we know which servers to connect to?
- How does the LLM know what tools exist?

### After Updates (Complete Flow)
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
4. CodeExecutionAgent.__init__(tools, tool_context)
   └─ Stores both callable tools and schemas
   ↓
5. agent.run(task)
   └─ Passes tool_context to LLM
   ↓
6. LLM generates code using known tools
   ↓
7. Code executes with access to actual tools
```

---

## New Files/Components

### Phase 0
- ✅ `mcp_servers.json` - Server configuration
- ✅ `mcp_manager.py` - Server connection and tool discovery
- ✅ `test_integration.py` - Test server discovery

### Phase 2
- ✅ `tool_formatter.py` - Format tool schemas for LLM
- ✅ `mcp_integration.py` - Orchestrate discovery and formatting
- ✅ `test_tool_calls.py` - Test manual tool invocation

### Phase 3
- ✅ Updated `agent.py` - Tool context injection
- ✅ Updated `CodeGenerationSignature` - `available_tools` field
- ✅ `example_usage.py` - End-to-end examples

---

## Key Insights

### 1. Configuration-Driven Discovery
The system uses `mcp_servers.json` to know which servers to connect to. This is:
- Explicit and transparent
- Easy to modify
- Similar to how Claude Desktop does it

### 2. Two-Stage Tool Handling
The system maintains tools in two forms:
- **Callable tools** (`mcp_tools`) - For actual execution
- **Tool schemas** (`tool_context`) - For LLM knowledge

### 3. LLM Context Injection
The formatted tool schemas are passed as an input field to the LLM, so it knows:
- What tools exist
- What parameters they take
- How to use them
- Example usage patterns

### 4. No Custom Bridges Needed
By using:
- DSpy's built-in `dspy.Tool.from_mcp_tool()`
- Standard MCP protocol `list_tools()`
- Configuration-driven server discovery

We avoid all the custom RPC bridges and code generation mentioned in the original research document.

---

## What Was Corrected

| Original Research Doc | Updated Plan | Why |
|----------------------|--------------|-----|
| "Auto-generated `mcp_tools.py`" | DSpy's `from_mcp_tool()` | Use built-in integration |
| "Flask/HTTP RPC bridge" | Direct MCP connections | Simpler, no custom server |
| "Tool discovery unclear" | `mcp_servers.json` config | Explicit configuration |
| "LLM tool awareness unclear" | `ToolSchemaFormatter` | Explicit schema formatting |

---

## Next Steps for Implementation

With these updates, the implementation path is now clear:

1. **Phase 0**: Create `mcp_servers.json` and `MCPServerManager`
2. **Phase 2**: Create `ToolSchemaFormatter` 
3. **Phase 3**: Update `CodeExecutionAgent` to use tool context
4. Run end-to-end: Config → Discovery → Format → Generate → Execute

The plan is now complete and actionable! 🚀
