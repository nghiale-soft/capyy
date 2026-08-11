"""Tool approval service (Dashboard-driven approve / deny / auto).

When the agent loop wants to run a tool, it asks this service first. Each tool
has a mode:

- ``allow`` — run immediately (default for read-only tools)
- ``deny``  — never run; the model gets a denial message
- ``ask``   — create a *pending* approval (visible on the Dashboard), then
  pause the loop until the user approves / denies (or the timeout expires).

Modes are persisted in ``config/tool-permissions.json`` (gitignored) and are
editable from the Dashboard → Settings. Pending approvals live in memory — the
dashboard and the agent loop share the process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("gateway.services.tool_approval")

TOOL_MODES = ("allow", "ask", "deny")

# Tool -> human label shown on the dashboard.
TOOL_LABELS: dict[str, str] = {
    "read_file": "Read file",
    "read_file_lines": "Read file lines",
    "write_file": "Write file",
    "edit_file": "Edit file",
    "list_dir": "List directory",
    "glob": "Glob",
    "grep": "Grep",
    "bash": "Run command",
    "git_status": "Git status",
    "git_diff": "Git diff",
    "http_get": "Fetch URL",
    "base64_encode": "Base64 encode",
    "base64_decode": "Base64 decode",
    "url_encode": "URL encode",
    "url_decode": "URL decode",
    "uuid": "UUID",
    "timestamp": "Timestamp",
    "json_parse": "Parse JSON",
    "browser_open": "Open web page",
    "browser_navigate": "Navigate browser",
    "browser_snapshot": "Browser snapshot",
    "browser_click": "Browser click",
    "browser_type": "Browser type",
    "browser_eval": "Browser eval",
    "browser_screenshot": "Browser screenshot",
    "browser_close": "Close browser",
    "figma_get_file": "Figma file",
    "figma_get_node": "Figma node",
    "figma_export_image": "Figma export",
}

# Defaults: read-only tools auto-run; writes, network and browser/figma actions
# require approval.
DEFAULT_PERMISSIONS: dict[str, str] = {
    "read_file": "allow",
    "read_file_lines": "allow",
    "list_dir": "allow",
    "glob": "allow",
    "grep": "allow",
    "git_status": "allow",
    "git_diff": "allow",
    "base64_encode": "allow",
    "base64_decode": "allow",
    "url_encode": "allow",
    "url_decode": "allow",
    "uuid": "allow",
    "timestamp": "allow",
    "json_parse": "allow",
    "browser_snapshot": "allow",
    "browser_screenshot": "allow",
    "browser_close": "allow",
    "write_file": "ask",
    "edit_file": "ask",
    "bash": "ask",
    "http_get": "ask",
    "browser_open": "ask",
    "browser_navigate": "ask",
    "browser_click": "ask",
    "browser_type": "ask",
    "browser_eval": "ask",
    "figma_get_file": "ask",
    "figma_get_node": "ask",
    "figma_export_image": "ask",
}


def _summarize_args(tool: str, arguments: dict[str, Any]) -> str:
    """Short human summary of the tool arguments.

    Each tool shows its most useful info: write_file previews the content,
    figma shows file_key/node_id, browser shows the URL/selector, bash shows
    the command, file tools show the path.
    """
    if not isinstance(arguments, dict):
        return ""

    def _first(*keys: str) -> str:
        for key in keys:
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    if tool == "write_file":
        path = str(arguments.get("path") or "").strip()
        content = arguments.get("content")
        preview = ""
        if isinstance(content, str) and content.strip():
            first = content.strip().splitlines()[0]
            preview = (
                f" → {first[:80]}" + ("…" if len(first) > 80 else "")
            )
        return f"{path or '(path?)'}{preview}"[:160]
    if tool.startswith("figma_"):
        file_key = _first("file_key")
        node_id = _first("node_id")
        parts = []
        if file_key:
            parts.append(f"file {file_key[:60]}")
        if node_id:
            parts.append(f"node {node_id[:40]}")
        if parts:
            return " · ".join(parts)[:120]
    if tool in ("browser_open", "browser_navigate"):
        url = _first("url")
        if url:
            return url[:120]
    if tool in ("browser_click", "browser_type"):
        selector = _first("selector")
        if selector:
            return f"{selector[:60]}" + (
                f" ← {_first('text')[:40]}" if _first("text") else ""
            )
    if tool == "browser_eval":
        js = _first("js")
        if js:
            return js[:120]
    if tool in ("http_get", "figma_export_image"):
        url = _first("url", "file_key")
        if url:
            return url[:120]
    if tool == "bash":
        command = _first("command")
        if command:
            return command[:120]
    if tool == "edit_file":
        path = _first("path")
        old = _first("old_string")
        if path:
            return f"{path[:80]}" + (f" → {old[:60]}" if old else "")[:60]
    for key in ("path", "pattern", "command"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:120]
    return json.dumps(arguments, ensure_ascii=False)[:120]


class ToolApprovalService:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.permissions_file = Path(settings.tool_permissions_file)
        self._permissions: dict[str, str] = dict(DEFAULT_PERMISSIONS)
        self._pending: dict[str, dict[str, Any]] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._load_permissions()

    # ------------------------------------------------------------------
    # Permissions (persisted, editable from the Dashboard)
    # ------------------------------------------------------------------

    def _load_permissions(self) -> None:
        try:
            data = json.loads(self.permissions_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        for tool, mode in data.items():
            if isinstance(mode, str) and mode in TOOL_MODES:
                self._permissions[str(tool)] = mode

    def _save_permissions(self) -> None:
        try:
            self.permissions_file.parent.mkdir(parents=True, exist_ok=True)
            self.permissions_file.write_text(
                json.dumps(self._permissions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            logger.warning("failed to save tool permissions: %s", error)

    def mode_for(self, tool: str) -> str:
        return self._permissions.get(tool, "allow")

    def permissions(self) -> dict[str, str]:
        return dict(self._permissions)

    def set_permission(self, tool: str, mode: str) -> None:
        if mode not in TOOL_MODES:
            raise ValueError(f"invalid tool mode: {mode}")
        self._permissions[str(tool)] = mode
        self._save_permissions()
        logger.info("tool permission %s -> %s", tool, mode)

    # ------------------------------------------------------------------
    # Pending approvals (in-memory; dashboard + agent loop share process)
    # ------------------------------------------------------------------

    def list_pending(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._pending.values()]

    async def request(
        self,
        tool: str,
        arguments: dict[str, Any],
        workdir: str,
    ) -> str | None:
        """Ask before running a tool.

        Returns ``None`` when the tool may run, otherwise a message to feed back
        to the model (denied / timed out / rejected).
        """
        mode = self.mode_for(tool)
        if mode == "allow":
            return None
        if mode == "deny":
            return f"[tool denied] {tool} is disabled on this gateway"

        approval_id = uuid.uuid4().hex[:12]
        item: dict[str, Any] = {
            "id": approval_id,
            "tool": tool,
            "label": TOOL_LABELS.get(tool, tool),
            "summary": _summarize_args(tool, arguments),
            "workdir": workdir,
            "ts": int(time.time() * 1000),
            "status": "pending",
        }
        event = asyncio.Event()
        async with self._lock:
            self._pending[approval_id] = item
            self._events[approval_id] = event
        logger.info("tool approval requested id=%s tool=%s args=%s", approval_id, tool, item["summary"])
        try:
            try:
                await asyncio.wait_for(
                    event.wait(),
                    timeout=max(1.0, self.settings.tool_approval_timeout),
                )
            except asyncio.TimeoutError:
                # The user may have approved just as the timeout fired — re-check
                # once before giving up (approve wins over the timeout message).
                async with self._lock:
                    current = self._pending.get(approval_id) or item
                if current.get("status") == "approved":
                    return None
                return (
                    f"[tool approval timed out] {tool}: {item['summary']}"
                    " (no response on the dashboard)"
                )
            current = self._pending.get(approval_id) or item
            if current.get("status") == "approved":
                return None
            return f"[tool denied by user] {tool}: {item['summary']}"
        finally:
            async with self._lock:
                self._events.pop(approval_id, None)
                self._pending.pop(approval_id, None)

    async def decide(self, approval_id: str, decision: str) -> bool:
        """Approve or deny a pending approval; wakes the waiting loop."""
        if decision not in ("approved", "denied"):
            raise ValueError("decision must be 'approved' or 'denied'")
        async with self._lock:
            item = self._pending.get(approval_id)
            if not item or item.get("status") != "pending":
                return False
            item["status"] = decision
            event = self._events.get(approval_id)
        logger.info("tool approval %s id=%s", decision, approval_id)
        if event is not None:
            event.set()
        return True
