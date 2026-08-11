"""Privacy-safe, user-approved tool-mapping contributions."""
from __future__ import annotations

import json
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any


class ToolMappingContributions:
    def __init__(self, path: str, repository: str) -> None:
        self.path = Path(path)
        self.repository = repository
        self._items: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
        except (OSError, ValueError):
            return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._items, ensure_ascii=False, indent=2), encoding="utf-8")

    def list(self) -> list[dict[str, Any]]:
        return list(self._items)

    def add(self, client: str, upstream_tool: str, client_tool: str, argument_mapping: dict[str, str]) -> dict[str, Any]:
        # Deliberately accept names/keys only: never persist tool values, paths,
        # commands, prompts, source, results or credentials.
        item = {"id": uuid.uuid4().hex, "status": "pending", "created_at": int(time.time()),
                "client": client, "upstream_tool": upstream_tool, "client_tool": client_tool,
                "argument_mapping": argument_mapping}
        self._items.append(item)
        self._save()
        return item

    def issue_url(self, contribution_id: str) -> str | None:
        item = next((x for x in self._items if x.get("id") == contribution_id), None)
        if item is None or item.get("status") != "pending" or not self.repository:
            return None
        body = "```json\n" + json.dumps({k: item[k] for k in ("client", "upstream_tool", "client_tool", "argument_mapping")}, ensure_ascii=False, indent=2) + "\n```\n\nNo prompts, paths, commands, tool values, source code, results, or credentials are included."
        item["status"] = "approved"
        self._save()
        return "https://github.com/" + self.repository + "/issues/new?" + urllib.parse.urlencode({"title": f"tool-mapping: {item['upstream_tool']} → {item['client_tool']}", "body": body, "labels": "tool-mapping"})

    def deny(self, contribution_id: str) -> bool:
        item = next((x for x in self._items if x.get("id") == contribution_id), None)
        if item is None or item.get("status") != "pending": return False
        item["status"] = "denied"; self._save(); return True
