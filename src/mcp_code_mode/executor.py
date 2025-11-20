"""Sandboxed code execution utilities backed by DSpy's PythonInterpreter.

Phase 1 of the implementation plan requires exposing a single MCP tool that can
execute Python inside DSpy's Deno + Pyodide sandbox. This module provides the
building blocks for that tool: a configurable executor that enforces timeouts,
normalizes interpreter output, and returns a structured result payload.
"""
from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass
import contextlib
import io
from threading import Lock
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TypedDict

try:  # Import guard so unit tests can stub the interpreter if DSpy is missing.
    from dspy.primitives.python_interpreter import (
        InterpreterError,
        PythonInterpreter,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    PythonInterpreter = Any  # type: ignore[assignment]
    InterpreterError = RuntimeError  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:  # pragma: no cover - exercised in integration tests
    _IMPORT_ERROR = None


class ExecutionResult(TypedDict, total=False):
    """Standard response payload for code execution."""

    success: bool
    stdout: str
    stderr: str
    duration_ms: int
    diagnostics: Optional[Dict[str, Any]]


@dataclass
class SandboxOptions:
    """Configuration passed to DSpy's PythonInterpreter."""

    enable_network_access: bool | Sequence[str] = False
    enable_env_vars: bool | Sequence[str] = False
    enable_read_paths: Sequence[str] = ()
    enable_write_paths: Sequence[str] = ()
    max_output_chars: int = 32_768

    def to_interpreter_kwargs(self) -> Dict[str, Any]:
        """Translate the dataclass to PythonInterpreter keyword arguments."""
        import os

        def _coerce_paths(value: Sequence[str]) -> list[str]:
            return list(value) if value else []

        enable_network_access = self.enable_network_access
        if enable_network_access is True:
            # dspy does not support True for allow-all, so we default to localhost
            # which is required for the tool bridge.
            enable_network_access = ["localhost", "127.0.0.1"]
        
        enable_env_vars = self.enable_env_vars
        if enable_env_vars is True:
            # dspy does not support True for allow-all, so we list all current env vars
            enable_env_vars = list(os.environ.keys())

        return {
            "enable_network_access": enable_network_access,
            "enable_env_vars": enable_env_vars,
            "enable_read_paths": _coerce_paths(self.enable_read_paths),
            "enable_write_paths": _coerce_paths(self.enable_write_paths),
        }


class SandboxedPythonExecutor:
    """Executes Python snippets inside the DSpy sandbox with timeouts."""

    def __init__(
        self,
        *,
        options: SandboxOptions | None = None,
        interpreter_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.options = options or SandboxOptions()
        self._interpreter_factory = interpreter_factory or self._default_factory
        self._interpreter: Any | None = None
        self._lock = Lock()

    def _default_factory(self) -> Any:
        """Instantiate the real DSpy PythonInterpreter."""

        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "dspy-ai is not installed. Activate your virtual environment and run "
                "'pip install -r requirements.txt' to install the runtime dependencies."
            ) from _IMPORT_ERROR

        return PythonInterpreter(**self.options.to_interpreter_kwargs())

    def _ensure_interpreter(self) -> Any:
        if self._interpreter is None:
            self._interpreter = self._interpreter_factory()
        return self._interpreter

    async def run(
        self,
        code: str,
        *,
        timeout: float = 30,
        variables: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute `code` with a timeout and normalized result payload."""

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero seconds")

        started = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(self._execute_sync, code, variables),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return self._timeout_result(started, timeout)
        except InterpreterError as exc:
            return self._exception_result(started, exc)
        except Exception as exc:  # pragma: no cover - safety net
            return self._exception_result(started, exc)

        duration_ms = self._elapsed_ms(started)
        stdout, stderr = self._normalize_output(raw)
        return ExecutionResult(
            success=True,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            diagnostics=None,
        )

    def _execute_sync(
        self,
        code: str,
        variables: Mapping[str, Any] | None,
    ) -> Any:
        """Invoke the underlying interpreter in a thread-safe manner."""

        with self._lock:
            interpreter = self._ensure_interpreter()
            return interpreter.execute(code, variables=variables)

    def _normalize_output(self, raw: Any) -> tuple[str, str]:
        """Extract stdout/stderr pairs from interpreter responses."""

        stdout = ""
        stderr = ""

        if isinstance(raw, dict):
            stdout = raw.get("stdout") or raw.get("output") or ""
            stderr = raw.get("stderr") or ""
        else:
            stdout_attr = getattr(raw, "stdout", None)
            stderr_attr = getattr(raw, "stderr", None)
            if stdout_attr is not None or stderr_attr is not None:
                stdout = stdout_attr or ""
                stderr = stderr_attr or ""
            else:  # Fall back to treating the entire result as stdout
                stdout = raw if isinstance(raw, str) else repr(raw)

        return (self._truncate(stdout), self._truncate(stderr))

    def _truncate(self, text: Any) -> str:
        """Limit stdout/stderr size to prevent runaway buffers."""

        if text is None:
            return ""
        safe_text = str(text)
        limit = self.options.max_output_chars
        if limit and len(safe_text) > limit:
            suffix = f"... [truncated {len(safe_text) - limit} chars]"
            return safe_text[:limit] + suffix
        return safe_text

    def _timeout_result(self, started: float, timeout: float) -> ExecutionResult:
        duration_ms = self._elapsed_ms(started)
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Execution timed out after {timeout:.2f}s",
            duration_ms=duration_ms,
            diagnostics={
                "error_type": "TIMEOUT",
                "timeout_seconds": timeout,
            },
        )

    def _exception_result(
        self,
        started: float,
        exc: BaseException,
    ) -> ExecutionResult:
        duration_ms = self._elapsed_ms(started)
        tb = traceback.format_exc()
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=str(exc),
            duration_ms=duration_ms,
            diagnostics={
                "error_type": exc.__class__.__name__,
                "traceback": tb,
            },
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)


class LocalPythonExecutor:
    """
    In-process executor used as a fallback when the Pyodide sandbox cannot
    perform network I/O (e.g., calling the MCP tool bridge from Pyodide
    currently fails with stack-switching errors inside Deno).
    """

    def __init__(self, max_output_chars: int = 64_000) -> None:
        self.max_output_chars = max_output_chars

    async def run(
        self,
        code: str,
        *,
        timeout: float = 30,
        variables: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero seconds")

        started = time.perf_counter()
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.to_thread(self._execute_sync, code, variables),
                timeout=timeout,
            )
            success = True
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout:.2f}s",
                duration_ms=self._elapsed_ms(started),
                diagnostics={"error_type": "TIMEOUT", "timeout_seconds": timeout},
            )
        except Exception as exc:  # pragma: no cover - convenience fallback
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(exc),
                duration_ms=self._elapsed_ms(started),
                diagnostics={
                    "error_type": exc.__class__.__name__,
                    "traceback": traceback.format_exc(),
                },
            )

        return ExecutionResult(
            success=success,
            stdout=self._truncate(stdout),
            stderr=self._truncate(stderr),
            duration_ms=self._elapsed_ms(started),
            diagnostics=None,
        )

    def _execute_sync(
        self,
        code: str,
        variables: Mapping[str, Any] | None,
    ) -> tuple[str, str]:
        env: Dict[str, Any] = {}
        if variables:
            env.update(variables)

        out_buf = io.StringIO()
        err_buf = io.StringIO()

        try:
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(
                err_buf
            ):
                exec(code, env, env)
        except Exception:
            err_buf.write(traceback.format_exc())
            raise

        return out_buf.getvalue(), err_buf.getvalue()

    def _truncate(self, text: Any) -> str:
        if text is None:
            return ""
        safe_text = str(text)
        if self.max_output_chars and len(safe_text) > self.max_output_chars:
            suffix = f"... [truncated {len(safe_text) - self.max_output_chars} chars]"
            return safe_text[: self.max_output_chars] + suffix
        return safe_text

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)


__all__ = [
    "ExecutionResult",
    "SandboxOptions",
    "SandboxedPythonExecutor",
    "LocalPythonExecutor",
]
