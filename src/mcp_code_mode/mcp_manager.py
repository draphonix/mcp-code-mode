"""MCP server manager for discovering and wrapping MCP tools.

Phase 0 focuses on ground-truthing the configuration for Model Context Protocol
servers. This module loads `mcp_servers.json`, connects to every configured
server via stdio, lists their tools, and exposes helper methods for summarizing
and wrapping those tools for DSpy.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

try:  # DSpy is optional during unit tests but required at runtime
    import dspy
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    dspy = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)


@dataclass
class ServerConnection:
    """Represents the client/session pair for a single MCP server."""

    name: str
    description: str
    config: Dict[str, Any]
    params: StdioServerParameters
    stdio_cm: Any = field(init=False, default=None)
    session_cm: Optional[ClientSession] = field(init=False, default=None)
    session: Optional[ClientSession] = field(init=False, default=None)
    tools: List[Any] = field(default_factory=list)
    dspy_tools: List[Any] = field(default_factory=list)

    async def open(self) -> None:
        """Open the stdio transport and initialize the MCP session."""

        LOGGER.debug("Opening server %s", self.name)
        self.stdio_cm = stdio_client(self.params)
        read, write = await self.stdio_cm.__aenter__()

        self.session_cm = ClientSession(read, write)
        self.session = await self.session_cm.__aenter__()
        await self.session.initialize()

    async def close(self) -> None:
        """Close the session and stdio transport."""

        LOGGER.debug("Closing server %s", self.name)
        if self.session_cm is not None:
            await self.session_cm.__aexit__(None, None, None)
            self.session_cm = None
            self.session = None

        if self.stdio_cm is not None:
            await self.stdio_cm.__aexit__(None, None, None)
            self.stdio_cm = None

    async def discover_tools(self) -> None:
        """List and cache the tools exposed by this server."""

        if self.session is None:
            raise RuntimeError(f"Server {self.name} not connected")

        response = await self.session.list_tools()
        self.tools = list(response.tools)

        if dspy is None:
            raise RuntimeError(
                "dspy-ai is not installed; install the runtime dependencies"
            ) from _IMPORT_ERROR

        self.dspy_tools = [
            dspy.Tool.from_mcp_tool(self.session, tool)
            for tool in self.tools
        ]

    def summary(self) -> str:
        lines = [f"{self.name} ({self.description or 'No description'})"]
        for tool in self.tools:
            lines.append(f"  - {tool.name}: {tool.description}")
        return "\n".join(lines)


class MCPServerManager:
    """Loads MCP server configs and keeps their sessions alive."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        # If no config path provided, check env var, then use default
        if config_path is None:
            config_path = os.environ.get("MCP_SERVERS_CONFIG", "mcp_servers.json")
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.servers: Dict[str, ServerConnection] = {}
        self.all_dspy_tools: List[Any] = []
        self._initialized = False

    async def __aenter__(self) -> "MCPServerManager":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"MCP config file not found: {self.config_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    async def initialize(self) -> List[Any]:
        """Connect to all configured servers and discover their tools."""

        if self._initialized:
            LOGGER.debug("MCPServerManager already initialized")
            return self.all_dspy_tools

        self.config = self._load_config()
        servers = self.config.get("servers", {})
        if not servers:
            raise ValueError("No servers configured in mcp_servers.json")

        LOGGER.info("Connecting to %d MCP servers", len(servers))

        for name, cfg in servers.items():
            await self._connect_and_discover(name, cfg)

        self._initialized = True
        LOGGER.info(
            "Connected to %d servers; discovered %d tools",
            len(self.servers),
            len(self.all_dspy_tools),
        )
        return self.all_dspy_tools

    async def _connect_and_discover(self, name: str, cfg: Dict[str, Any]) -> None:
        if name in self.servers:
            LOGGER.debug("Server %s already connected", name)
            return

        # Inherit system environment and overlay config env
        env = dict(os.environ)
        env.update(cfg.get("env", {}))

        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=env,
        )

        connection = ServerConnection(
            name=name,
            description=cfg.get("description", ""),
            config=cfg,
            params=params,
        )

        try:
            await connection.open()
            await connection.discover_tools()
        except Exception:
            LOGGER.exception("Failed to connect to server %s", name)
            await connection.close()
            raise

        self.servers[name] = connection
        self.all_dspy_tools.extend(connection.dspy_tools)

    async def shutdown(self) -> None:
        """Close all open MCP sessions."""

        # Connections were opened sequentially, so close them in LIFO order to
        # unwind the nested async context managers correctly.
        for connection in reversed(list(self.servers.values())):
            try:
                # Close sequentially so stdio_client.__aexit__ runs
                # in the same task where __aenter__ was awaited.
                await connection.close()
            except Exception:
                LOGGER.exception("Failed to close server %s", connection.name)
        self.servers.clear()
        self.all_dspy_tools = []
        self._initialized = False

    def get_tools_summary(self) -> str:
        """Return a readable summary of all known tools."""

        if not self.servers:
            return "No servers connected"

        sections = ["Available MCP Tools:"]
        for connection in self.servers.values():
            sections.append("")
            sections.append(connection.summary())
        return "\n".join(sections)

    @property
    def tools(self) -> List[Any]:
        if not self._initialized:
            raise RuntimeError("MCPServerManager.initialize() has not been called")
        return self.all_dspy_tools


async def main() -> None:
    """Convenience entry point for manual verification."""

    manager = MCPServerManager()
    try:
        await manager.initialize()
        print(manager.get_tools_summary())
    finally:
        await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
