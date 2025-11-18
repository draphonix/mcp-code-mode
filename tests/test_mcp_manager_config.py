import json

import pytest

from mcp_code_mode.mcp_manager import MCPServerManager


def test_load_config_success(tmp_path):
    config = {
        "servers": {
            "example": {
                "command": "echo",
                "args": ["hello"],
                "description": "Example server"
            }
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    manager = MCPServerManager(config_path)
    loaded = manager._load_config()

    assert loaded == config


def test_load_config_missing(tmp_path):
    missing_path = tmp_path / "missing.json"
    manager = MCPServerManager(missing_path)

    with pytest.raises(FileNotFoundError):
        manager._load_config()
