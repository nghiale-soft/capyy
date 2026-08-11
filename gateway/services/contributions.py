"""Privacy-safe, user-approved community issue drafts."""
from __future__ import annotations

import json
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any


class Contributions:
    """Persist only a deliberately small, non-sensitive diagnostic summary."""

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

    def add(self, kind: str, title: str, summary: str, metadata: dict[str, str]) -> dict[str, Any]:
        item = {
            "id": uuid.uuid4().hex,
            "status": "pending",
            "created_at": int(time.time()),
            "kind": kind,
            "title": title,
            "summary": summary,
            "metadata": metadata,
        }
        self._items.append(item)
        self._save()
        return item

    def issue_url(self, contribution_id: str) -> str | None:
        item = next((x for x in self._items if x.get("id") == contribution_id), None)
        if item is None or item.get("status") != "pending" or not self.repository:
            return None
        body = (
            f"## {item['kind']}\n\n{item['summary']}\n\n"
            "```json\n"
            + json.dumps(item["metadata"], ensure_ascii=False, indent=2)
            + "\n```\n\n"
            "This report intentionally excludes prompts, source, paths, commands, tool output, tokens, and credentials."
        )
        item["status"] = "approved"
        self._save()
        return "https://github.com/" + self.repository + "/issues/new?" + urllib.parse.urlencode(
            {"title": f"{item['kind']}: {item['title']}", "body": body, "labels": "contribution"}
        )

    def deny(self, contribution_id: str) -> bool:
        item = next((x for x in self._items if x.get("id") == contribution_id), None)
        if item is None or item.get("status") != "pending":
            return False
        item["status"] = "denied"
        self._save()
        return True
