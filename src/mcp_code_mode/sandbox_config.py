"""Centralized sandbox configuration and guardrail constants."""
from __future__ import annotations

from .executor import SandboxOptions

# Network access is limited to localhost so the tool bridge can relay requests
# without exposing the sandbox to the wider internet. External network access
# should always happen via MCP tools.
_LOCALHOST = ("127.0.0.1", "localhost")

# Only a minimal set of environment variables may leak into the sandbox.
_ENV_ALLOWLIST = (
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_MODEL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)

DEFAULT_SANDBOX_OPTIONS = SandboxOptions(
    enable_network_access=_LOCALHOST,
    enable_env_vars=_ENV_ALLOWLIST,
    enable_read_paths=(),
    enable_write_paths=(),
    max_output_chars=64_000,
)

__all__ = ["DEFAULT_SANDBOX_OPTIONS"]
