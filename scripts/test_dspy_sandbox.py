"""Simple smoke test for the DSpy PythonInterpreter sandbox."""
from typing import Any, Dict

try:
    from dspy.primitives.python_interpreter import (
        InterpreterError,
        PythonInterpreter,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "dspy-ai is not installed. Activate your virtualenv and run"
        " 'pip install -r requirements.txt' before executing this script."
    ) from exc


def _normalize_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return {
            "stdout": str(result.get("stdout") or result.get("output") or ""),
            "stderr": str(result.get("stderr") or ""),
        }

    stdout = getattr(result, "stdout", None)
    stderr = getattr(result, "stderr", None)
    if stdout is not None or stderr is not None:
        return {"stdout": stdout or "", "stderr": stderr or ""}
    return {"stdout": str(result), "stderr": ""}


def main() -> None:
    interpreter = PythonInterpreter(
        enable_network_access=[],  # No outbound hosts allowed during Phase 0
        enable_env_vars=[],  # Disallow env propagation for the smoke test
        enable_read_paths=["/tmp"],
        enable_write_paths=["/tmp"],
    )

    snippet = 'print("Hello from DSpy sandbox!")'

    try:
        raw = interpreter.execute(snippet)
    except InterpreterError as exc:
        raise SystemExit(
            "Failed to start the DSpy sandbox. This usually means Deno could not "
            "download the bundled Pyodide runtime. Ensure the machine has outbound "
            "network access to https://registry.npmjs.org and rerun the script."
        ) from exc
    result = _normalize_result(raw)

    print("stdout:\n" + result.get("stdout", ""))
    if result.get("stderr"):
        print("stderr:\n" + result["stderr"])


if __name__ == "__main__":
    main()
