"""FastMCP echo server used for Phase 0 sanity checks."""
from fastmcp import Context, FastMCP

mcp = FastMCP("Phase0 Echo Server")


@mcp.tool()
async def echo(message: str, ctx: Context | None = None) -> str:
    """Echo a message back to the caller."""

    if ctx is not None:
        await ctx.info(f"Echoing message of length {len(message)}")
    return f"Echo: {message}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
