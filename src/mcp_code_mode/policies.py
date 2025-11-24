"""Lightweight policy checks for sandboxed code snippets."""
from __future__ import annotations

import re
from typing import Callable, Tuple

MAX_CHARS = 18000
MAX_LINES = 400

ALLOWED_IMPORTS = {
    "math",
    "json",
    "re",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "datetime",
    "typing",
    # Needed by the MCP tool bridge prelude inside the sandbox
    "urllib",
}

DISALLOWED_TOKENS = (
)

IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)
# Match bare "open(" that is not part of another identifier (e.g., blocks open(
# but allows urlopen()
OPEN_CALL_RE = re.compile(r"(?<![a-z0-9_])open\(", re.IGNORECASE)


def enforce_guardrails(code: str) -> Tuple[bool, str | None]:
    """Return (allowed, message) after scanning the snippet."""

    for checker in (_check_size, _check_imports, _check_tokens):
        allowed, reason = checker(code)
        if not allowed:
            return allowed, reason
    return True, None


def _check_size(code: str) -> Tuple[bool, str | None]:
    if len(code) > MAX_CHARS:
        return False, f"Snippet too large ({len(code)} chars > {MAX_CHARS})"
    line_count = code.count("\n") + 1
    if line_count > MAX_LINES:
        return False, f"Snippet has too many lines ({line_count} > {MAX_LINES})"
    return True, None


def _check_imports(code: str) -> Tuple[bool, str | None]:
    for match in IMPORT_RE.finditer(code):
        module = match.group(1).split(".")[0]
        if module not in ALLOWED_IMPORTS:
            return False, f"Import '{module}' is not allowed in the sandbox"
    return True, None


def _check_tokens(code: str) -> Tuple[bool, str | None]:
    lowered = code.lower()
    # Block bare open( but allow namespaced usages like urllib.request.urlopen
    if OPEN_CALL_RE.search(lowered):
        return False, "Disallowed pattern detected: open("
    for token in DISALLOWED_TOKENS:
        if token in lowered:
            return False, f"Disallowed pattern detected: {token}"
    return True, None


__all__ = [
    "enforce_guardrails",
    "DISALLOWED_TOKENS",
    "ALLOWED_IMPORTS",
    "MAX_CHARS",
    "MAX_LINES",
]
