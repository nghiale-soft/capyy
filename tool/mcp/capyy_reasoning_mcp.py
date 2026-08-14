#!/usr/bin/env python3
"""A tiny host-side MCP server for a collapsible Capyy reasoning step.

It deliberately has no filesystem, shell, network, or environment access.
The client renders this tool call in its normal tool timeline; the gateway can
then keep FreeBuff reasoning visible without pretending it was a file read.
"""

from __future__ import annotations

import json
import sys
from typing import Any


TOOL = {
    "name": "show_reasoning",
    "description": "Display sanitized FreeBuff reasoning in a collapsible Capyy tool step.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Reasoning to display."},
            "truncated": {"type": "boolean", "description": "Whether the gateway truncated it."},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
}


def _response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """Handle the small MCP surface required by Claude/Codex stdio clients."""
    request_id = message.get("id")
    method = message.get("method")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return _response(
            request_id,
            {
                "protocolVersion": message.get("params", {}).get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "capyy-reasoning", "version": "0.1.0"},
            },
        )
    if method == "tools/list":
        return _response(request_id, {"tools": [TOOL]})
    if method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") != "show_reasoning":
            return _error(request_id, -32602, "Unknown tool")
        arguments = params.get("arguments") or {}
        text = arguments.get("text")
        if not isinstance(text, str):
            return _error(request_id, -32602, "text must be a string")
        # The reasoning itself remains in the tool input (where the IDE can
        # collapse it); avoid duplicating it into the tool result/context.
        return _response(
            request_id,
            {"content": [{"type": "text", "text": f"Capyy displayed FreeBuff reasoning ({len(text)} chars)."}]},
        )
    if request_id is None:
        return None
    return _error(request_id, -32601, f"Method not found: {method}")


def main() -> None:
    for raw in sys.stdin:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue
        response = handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
