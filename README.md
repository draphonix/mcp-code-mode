# MCP Code Mode

Prototype implementation for the Code Execution MCP Server with DSpy. This repo follows the implementation plan in `docs/implementation-plan.md`.

## Toolchain Requirements

- Python 3.11 (3.11.0 or newer, <3.13 recommended)
- Node.js 20+ with `npx` available (needed for the reference MCP servers)
- `pip` for installing the Python dependencies listed in `pyproject.toml` / `requirements*.txt`

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

To keep the Node-based MCP servers current, run:

```bash
npm install -g npm@latest
```

The `mcp_servers.json` file enumerates the default MCP servers (filesystem, memory, fetch). Update this file to point at any additional servers you want available during experimentation.
