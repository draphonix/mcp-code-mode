"""Integration smoke test for Phase 0 MCP wiring."""
import asyncio
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcp_code_mode.mcp_manager import MCPServerManager

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    manager = MCPServerManager()
    async with manager:
        summary = manager.get_tools_summary()
        print("\n" + "=" * 60)
        print("Phase 0 MCP Tool Discovery Summary")
        print("=" * 60)
        print(summary)
        print("=" * 60)
        print(f"Total DSpy tools discovered: {len(manager.tools)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FileNotFoundError as exc:
        print(f"Configuration missing: {exc}")
