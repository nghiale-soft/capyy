"""Figma REST API tools for the gateway agent loop.

Tokens form a **pool** (multiple Figma accounts), configured on the Dashboard →
Settings → Figma tokens. ``token_for()`` picks from the pool round-robin so
quota is spread across accounts; fallbacks: legacy ``config/figma-token.json``
(``{"token": ...}``) and ``FIGMA_TOKEN`` env.

File: ``config/figma-tokens.json`` → ``{"tokens": ["figd_a", "figd_b"]}``
(gitignored). If nothing is configured the tools return a clear setup message.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("gateway.services.figma")

FIGMA_API = "https://api.figma.com/v1"
TOKENS_FILE = "config/figma-tokens.json"
LEGACY_TOKEN_FILE = "config/figma-token.json"
_MAX = 50000

# Round-robin cursor shared across requests so the pool spreads evenly.
_rr_index = 0


def _mask_token(token: str, keep: int = 4) -> str:
    """Mask a token for the UI — keep only a few leading/trailing chars."""
    if len(token) <= keep:
        return "*" * len(token)
    if len(token) <= keep + 4:
        return "*" * min(len(token) - keep, 4) + token[-keep:]
    return token[:3] + "*" * 8 + token[-keep:]


class FigmaTokenStore:
    """Persisted Figma token pool (multiple accounts).

    File: ``config/figma-tokens.json`` → ``{"tokens": ["figd_a", "figd_b"]}``
    (gitignored). The dashboard edits this list directly; the tools pick from
    it round-robin like the FreeBuff account pool.
    """

    def __init__(self, path: str | Path = TOKENS_FILE) -> None:
        self.path = Path(path)

    # ---------------------------------------------------------------- load/save

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            logger.warning("failed to save figma tokens: %s", error)

    def _pool_tokens(self) -> list[str]:
        """Editable pool tokens (persisted in the file), backwards-compat shape."""
        data = self._load()
        raw = data.get("tokens")
        tokens = [t for t in raw if isinstance(t, str) and t.strip()] if isinstance(raw, list) else []
        # Backwards-compat: old single-token shape.
        if not tokens and isinstance(data.get("default"), str) and data.get("default").strip():
            tokens = [data["default"].strip()]
        return tokens

    def _fallback_tokens(self) -> list[str]:
        """Read-only fallbacks (legacy file + env) — not editable from the UI."""
        try:
            legacy = json.loads(Path(LEGACY_TOKEN_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            legacy = {}
        legacy_token = legacy.get("token") if isinstance(legacy, dict) else None
        fallbacks = []
        if isinstance(legacy_token, str) and legacy_token.strip():
            fallbacks.append(legacy_token.strip())
        env_token = os.getenv("FIGMA_TOKEN")
        if isinstance(env_token, str) and env_token.strip():
            fallbacks.append(env_token.strip())
        return fallbacks

    # ------------------------------------------------------------------- read

    def token_for(self, project_key: str = "") -> str:
        """Pick the next token from pool + fallbacks (round-robin)."""
        global _rr_index
        tokens = self._pool_tokens() + self._fallback_tokens()
        if not tokens:
            return ""
        _rr_index += 1
        return tokens[_rr_index % len(tokens)]

    def status(self, *, reveal: bool = False) -> dict[str, Any]:
        """Masked status for the dashboard (reveal=1 also returns real values).

        ``tokens`` = editable pool (indices match the delete endpoint).
        ``fallback`` = env / legacy file tokens (read-only, listed separately).
        """
        tokens = self._pool_tokens()
        entries = []
        for i, token in enumerate(tokens):
            entry: dict[str, Any] = {"index": i, "masked": _mask_token(token)}
            if reveal:
                entry["value"] = token
            entries.append(entry)
        fallback_entries = []
        for token in self._fallback_tokens():
            entry: dict[str, Any] = {"masked": _mask_token(token)}
            if reveal:
                entry["value"] = token
            fallback_entries.append(entry)
        return {
            "tokens": entries,
            "fallback": fallback_entries,
            "configured": bool(entries) or bool(fallback_entries),
            "source": "file" if (self._load() or self.path.exists()) else "env",
        }

    # ------------------------------------------------------------------ write

    def add_token(self, token: str) -> list[str]:
        """Append one token to the pool; returns the new pool."""
        token = str(token or "").strip()
        if not token:
            raise ValueError("token is required")
        data = self._load()
        raw = data.get("tokens")
        tokens = [t for t in raw if isinstance(t, str) and t.strip()] if isinstance(raw, list) else []
        if token in tokens:
            raise ValueError("token already exists")
        tokens.append(token)
        self._save({"tokens": tokens})
        return tokens

    def remove_token(self, index: int) -> list[str]:
        """Remove one token by index; returns the new pool."""
        data = self._load()
        raw = data.get("tokens")
        tokens = [t for t in raw if isinstance(t, str) and t.strip()] if isinstance(raw, list) else []
        try:
            removed = tokens.pop(index)
        except IndexError:
            raise IndexError("token index out of range")
        logger.info("removed figma token %s", _mask_token(removed))
        self._save({"tokens": tokens})
        return tokens

    def replace_tokens(self, tokens: list[str]) -> list[str]:
        """Replace the whole pool; returns the new pool."""
        cleaned = [t.strip() for t in tokens if isinstance(t, str) and t.strip()]
        if not cleaned:
            raise ValueError("at least one token is required")
        self._save({"tokens": cleaned})
        return cleaned

    def clear_all(self) -> None:
        self._save({"tokens": []})


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


async def _figma_get(
    path: str,
    params: dict[str, Any] | None = None,
    *,
    project_key: str = "",
    token: str | None = None,
) -> tuple[int, Any]:
    if token is None:
        token = FigmaTokenStore().token_for(project_key)
    if not token:
        return (
            401,
            "FIGMA_TOKEN is not configured. Add one on the Dashboard → "
            "Settings → Figma tokens (or set the FIGMA_TOKEN env var / create "
            f"{TOKENS_FILE} with {{\"default\": \"...\"}}).",
        )
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{FIGMA_API}{path}",
                params=params,
                headers={"X-Figma-Token": token},
            )
    except Exception as exc:  # noqa: BLE001
        return 0, f"Figma request failed: {exc}"
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        data = {"raw": resp.text[:2000]}
    return resp.status_code, data


def _compact(node: Any, depth: int = 0, max_depth: int = 3) -> Any:
    """Small summary of a Figma node (id/name/type + child count)."""
    if not isinstance(node, dict) or depth > max_depth:
        return None
    out: dict[str, Any] = {
        "id": node.get("id"),
        "name": node.get("name"),
        "type": node.get("type"),
    }
    children = node.get("children")
    if isinstance(children, list):
        out["children"] = len(children)
        if children and depth < max_depth:
            out["first"] = [_compact(c, depth + 1, max_depth) for c in children[:3]]
    return out


def _capped(value: str) -> str:
    return value if len(value) <= _MAX else value[:_MAX] + "\n...[truncated]"


async def figma_get_file(file_key: str, *, project_key: str = "") -> str:
    status, data = await _figma_get(
        f"/files/{str(file_key).strip()}",
        project_key=project_key,
    )
    if status != 200:
        return f"Figma error {status}: {data}"
    return _capped(json.dumps(_compact(data), ensure_ascii=False, indent=2))


async def figma_get_node(
    file_key: str,
    node_id: str,
    *,
    project_key: str = "",
) -> str:
    status, data = await _figma_get(
        f"/files/{str(file_key).strip()}/nodes",
        {"ids": str(node_id).strip()},
        project_key=project_key,
    )
    if status != 200:
        return f"Figma error {status}: {data}"
    nodes = data.get("nodes") or {}
    summary = {
        key: _compact(value.get("document"))
        for key, value in nodes.items()
    }
    return _capped(json.dumps(summary, ensure_ascii=False, indent=2))


async def figma_export_image(
    file_key: str,
    node_id: str,
    format: str = "png",
    *,
    project_key: str = "",
) -> str:
    status, data = await _figma_get(
        f"/images/{str(file_key).strip()}",
        {"ids": str(node_id).strip(), "format": str(format), "scale": "2"},
        project_key=project_key,
    )
    if status != 200:
        return f"Figma error {status}: {data}"
    url = (data.get("images") or {}).get(node_id)
    if url:
        return f"Image URL for {node_id}: {url}"
    return f"No image URL returned: {data}"
