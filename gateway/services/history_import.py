from __future__ import annotations

"""Parse chat history from AI tools into the gateway's standard records.

Used by history_scan (auto-scan of ~/.claude, ~/.codex).
Supports:
- Claude Code: JSONL files (``~/.claude/projects/.../*.jsonl``)
- Codex CLI: session JSONL files (``~/.codex/sessions/**/*.jsonl``)

Result is a list of records ``{ts, role, content, thinking?, tool_calls?}`` — the
internal schema (see gateway.services.chat_history).
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .chat_history import _content_to_text

logger = logging.getLogger("gateway.services.history_import")


def _records_from_messages(
    messages: list[Any],
    ts: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            text, thinking, tool_calls = _assistant_blocks(content)
            # OpenAI puts tool_calls at message level (not inside content)
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    tool_calls.append(_block_to_tool_call(call))
            record: dict[str, Any] = {
                "role": "assistant",
                "content": text,
                "ts": ts,
            }
            if thinking:
                record["thinking"] = thinking
            if tool_calls:
                record["tool_calls"] = tool_calls
            records.append(record)
        elif role == "user":
            records.append(
                {"role": "user", "content": _content_to_text(content), "ts": ts}
            )
        # Skip system/tool (tool_result lives inside Anthropic user content)
    return records


def _assistant_blocks(content: Any) -> tuple[str, str, list[dict[str, Any]]]:
    """Split assistant content into (text, thinking, tool_calls)."""
    if isinstance(content, str):
        return content, "", []
    if not isinstance(content, list):
        return str(content), "", []

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            if isinstance(block, str):
                text_parts.append(block)
            continue
        block_type = block.get("type")

        if block_type in ("text", "output_text"):
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        elif block_type == "thinking":
            thinking = block.get("thinking")
            if isinstance(thinking, str):
                thinking_parts.append(thinking)
        elif block_type == "reasoning":
            summary = block.get("summary")
            if isinstance(summary, list):
                thinking_parts.append(
                    "\n".join(
                        str(item.get("summary", ""))
                        for item in summary
                        if isinstance(item, dict)
                    )
                )
        elif block_type in ("tool_use", "function_call"):
            tool_calls.append(_block_to_tool_call(block))
        elif block_type == "function":
            tool_calls.append(_block_to_tool_call(block))

    return (
        "\n".join(text_parts) if text_parts else "",
        "\n".join(part for part in thinking_parts if part),
        tool_calls,
    )


def _block_to_tool_call(block: dict[str, Any]) -> dict[str, Any]:
    name = block.get("name") or (block.get("function") or {}).get("name")
    tool_input = block.get("input")
    if tool_input is None:
        tool_input = (block.get("function") or {}).get("arguments") or {}
    if not isinstance(tool_input, dict):
        try:
            tool_input = json.loads(tool_input) if isinstance(tool_input, str) else {}
        except json.JSONDecodeError:
            tool_input = {}
    try:
        arguments = json.dumps(tool_input, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        arguments = "{}"
    return {
        "id": block.get("id") or "",
        "type": "function",
        "function": {"name": str(name or "tool"), "arguments": arguments},
    }


# ----------------------------------------------------------------------
# Claude Code: ~/.claude/projects/<slug>/*.jsonl
# ----------------------------------------------------------------------

def _claude_entries(data: Any) -> list[Any]:
    """Parse JSONL text or a list of entries into a dict list."""
    if isinstance(data, str):
        lines = []
        for raw in data.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return lines
    if isinstance(data, list):
        return data
    raise ValueError("claude_code import requires JSONL text or list of entries")


def _claude_records(entries: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ts = _iso_to_ms(entry.get("timestamp"))
        entry_type = entry.get("type")
        if entry_type in ("summary", "system"):
            continue

        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            # content blocks: text + tool_result
            text = _content_to_text(content)
            if text:
                records.append({"role": "user", "content": text, "ts": ts})
        elif role == "assistant":
            text, thinking, tool_calls = _assistant_blocks(content)
            record: dict[str, Any] = {"role": "assistant", "content": text, "ts": ts}
            if thinking:
                record["thinking"] = thinking
            if tool_calls:
                record["tool_calls"] = tool_calls
            records.append(record)
        elif entry_type == "toolUseResult" and content:
            records.append(
                {"role": "user", "content": f"[tool_result] {content}", "ts": ts}
            )

    return records


def claude_code_session(data: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Parse one Claude Code session -> (records, cwd)."""
    entries = _claude_entries(data)
    cwd: str | None = None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("cwd"):
            cwd = str(entry["cwd"])
            break
    return _claude_records(entries), cwd


# ----------------------------------------------------------------------
# Codex CLI: ~/.codex/sessions/**/*.jsonl (mới) hoặc *.json (cũ)
# ----------------------------------------------------------------------

def _codex_lines(data: Any) -> list[Any]:
    if isinstance(data, str):
        lines = []
        for raw in data.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return lines
    if isinstance(data, list):
        return data
    raise ValueError("codex import requires JSONL text or list of entries")


def _codex_records_from_lines(lines: list[Any]) -> tuple[list[dict[str, Any]], str | None]:
    """Parse a Codex JSONL session -> (records, cwd).

    ``session_meta`` rows carry ``payload.cwd``; ``response_item`` rows with
    ``payload.type == "message"`` are user/assistant messages.
    """
    cwd: str | None = None
    records: list[dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_type = line.get("type")
        if line_type == "session_meta":
            payload = line.get("payload")
            if isinstance(payload, dict) and payload.get("cwd"):
                cwd = str(payload["cwd"])
            continue
        if line_type != "response_item":
            continue
        payload = line.get("payload")
        if not isinstance(payload, dict):
            continue
        ts = _iso_to_ms(line.get("timestamp"))
        if payload.get("type") == "function_call":
            records.append({
                "role": "assistant", "content": "", "ts": ts,
                "tool_calls": [{"id": payload.get("call_id") or payload.get("id"), "type": "function", "function": {"name": payload.get("name") or "tool", "arguments": payload.get("arguments") or "{}"}}],
            })
            continue
        if payload.get("type") == "function_call_output":
            output = payload.get("output", "")
            if isinstance(output, list):
                output = "\n".join(str(item.get("text", "")) for item in output if isinstance(item, dict))
            records.append({"role": "user", "content": "[tool_result] " + str(output), "ts": ts})
            continue
        if payload.get("type") != "message":
            continue
        role = payload.get("role")
        if role not in ("user", "assistant"):
            continue
        converted = _records_from_messages(
            [
                {
                    "role": role,
                    "content": payload.get("content"),
                    # tool_calls may live at message level (payload)
                    "tool_calls": payload.get("tool_calls"),
                }
            ],
            ts=ts,
        )
        records.extend(converted)
    return records, cwd


def codex_session(data: Any) -> tuple[list[dict[str, Any]], str | None]:
    return _codex_records_from_lines(_codex_lines(data))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _iso_to_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value) if value > 1e12 else int(value * 1000)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        return None
