# mcp-code-mode Agent Guide

## Commands

- **Setup**: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]`
- **Server**: `python -m mcp_code_mode.executor_server`
- **Test All**: `pytest` or with coverage `pytest --cov=mcp_code_mode`
- **Test Single**: `pytest tests/test_executor.py` or `pytest tests/test_executor.py::test_name`
- **Lint**: `ruff check .` (auto-fix: `ruff check . --fix`), `black .`, `mypy src`
- **Verify**: `python scripts/test_dspy_sandbox.py` (integration sanity check)

## Code Style

- **Python**: 3.11-3.12 only (strict >=3.11,<3.13)
- **Formatting**: Black (line length 88), Ruff linting, full type hints for mypy --strict
- **Imports**: Absolute from `mcp_code_mode` (e.g., `from mcp_code_mode.executor import ...`)
- **Async**: Prefer `async/await` over threads; use `asyncio.run()` for entry points
- **Types**: Use `TypedDict`, `dataclass`, type annotations everywhere; avoid `Any` unless unavoidable
- **Errors**: Raise specific exceptions (e.g., `RuntimeError`, `ValueError`), include context in messages
- **Naming**: snake_case functions/vars, PascalCase classes, UPPER_SNAKE constants, descriptive names
- **Tests**: Use `pytest`, `pytest-asyncio` for async, stub external deps (see `StubInterpreter` pattern)
- **Docstrings**: Module-level for non-trivial files, class/function docstrings for public APIs

## Context

- **Purpose**: MCP server exposing DSpy sandbox (Deno + Pyodide) for secure code execution
- **Key files**: `executor.py` (sandbox), `agent.py` (DSpy CodeAct), `executor_server.py` (MCP server)
- **Env**: Copy `.env.example` to `.env` for API keys (Gemini, etc.)
