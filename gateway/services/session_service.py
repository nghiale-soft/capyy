from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from providers.freebuff import (
    CodebuffAccountLease,
    CodebuffAccountPool,
    CodebuffClient,
    SessionManager,
)
from gateway.core.config import Settings


logger = logging.getLogger("gateway.services.session")


def _normalize_tokens(value: Any) -> tuple[str, ...]:
    """Accept a str (comma separated), list/tuple, or JSON list."""
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return ()
    tokens = tuple(str(item).strip() for item in items if str(item).strip())
    return tokens


class SessionService:
    """Manages sessions and the account pool for the Freebuff provider.

    Tokens come from the dashboard-managed config file
    (`settings.tokens_file`, gitignored, written by the dashboard). The file
    wins over env; when updated via the dashboard, the pool is rebuilt right away.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._reload_lock = asyncio.Lock()
        self._tokens = self._load_tokens()
        self._active_index = self._read_active_index()
        self.pool = self._build_pool(self._tokens)

    # ------------------------------------------------------------------
    # Token persistence
    # ------------------------------------------------------------------

    @property
    def tokens_file(self) -> Path:
        return Path(self.settings.tokens_file)

    def _load_tokens(self) -> tuple[str, ...]:
        file_tokens = self._read_tokens_file()
        if file_tokens:
            logger.info(
                "freebuff tokens loaded source=file count=%s",
                len(file_tokens),
            )
            return file_tokens
        env_tokens = self.settings.codebuff_tokens
        if env_tokens:
            logger.info(
                "freebuff tokens loaded source=env count=%s",
                len(env_tokens),
            )
            return env_tokens
        return ()

    def _read_tokens_file(self) -> tuple[str, ...]:
        path = self.tokens_file
        if not path.exists():
            return ()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("failed to read freebuff tokens file: %s", error)
            return ()
        return _normalize_tokens(raw.get("tokens") if isinstance(raw, dict) else None)

    def _read_active_index(self) -> int:
        try:
            raw = json.loads(self.tokens_file.read_text(encoding="utf-8"))
            value = raw.get("active_index", 0) if isinstance(raw, dict) else 0
            return max(0, int(value))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return 0

    def _write_tokens_file(self, tokens: tuple[str, ...]) -> None:
        path = self.tokens_file
        path.parent.mkdir(parents=True, exist_ok=True)
        if tokens:
            self._active_index %= len(tokens)
        payload = {
            "version": 2,
            "tokens": list(tokens),
            "active_index": self._active_index,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _persist_active_index(self, index: int) -> None:
        """Persist the sticky default without exposing token values in logs."""
        if not self._tokens or self.token_source != "file":
            return
        self._active_index = index % len(self._tokens)
        try:
            self._write_tokens_file(self._tokens)
            logger.info("freebuff default account persisted index=%s", self._active_index)
        except OSError as error:
            logger.warning("failed to persist freebuff default account: %s", error)

    @property
    def token_source(self) -> str:
        if self._read_tokens_file():
            return "file"
        if self.settings.codebuff_token:
            return "env"
        return "none"

    @property
    def tokens(self) -> list[str]:
        """Tokens currently in use (pool order)."""
        return list(self._tokens)

    async def update_tokens(self, tokens: list[str]) -> list[str]:
        """Write tokens to the file and reload the pool (called from API/dashboard).

        An empty list deletes the config file and falls back to env, avoiding an
        empty file contradicting the running pool. Returns the active list.
        """
        normalized = _normalize_tokens(tokens)
        async with self._reload_lock:
            if not normalized:
                await self._remove_tokens_file()
                active = self.settings.codebuff_tokens
            else:
                self._active_index = min(self._active_index, len(normalized) - 1)
                self._write_tokens_file(normalized)
                logger.info("freebuff tokens updated count=%s", len(normalized))
                active = normalized
            await self._swap_pool(active)
        return list(active)

    async def remove_token(self, index: int) -> list[str]:
        """Remove one token by index; returns the remaining list.

        If the file empties, falls back to env (if set).
        """
        current = list(self._tokens)
        if index < 0 or index >= len(current):
            raise IndexError(index)
        current.pop(index)
        remaining = tuple(current)
        async with self._reload_lock:
            if remaining:
                self._active_index = min(self._active_index, len(remaining) - 1)
                self._write_tokens_file(remaining)
            else:
                await self._remove_tokens_file()
                remaining = self.settings.codebuff_tokens
            logger.info("freebuff token removed; remaining=%s", len(remaining))
            await self._swap_pool(remaining)
        return list(remaining)

    async def clear_tokens(self) -> None:
        """Delete the config file, falling back to env (if set)."""
        async with self._reload_lock:
            await self._remove_tokens_file()
            env_tokens = self.settings.codebuff_tokens
            logger.info("freebuff tokens cleared; falling back to env count=%s", len(env_tokens))
            await self._swap_pool(env_tokens)

    async def _remove_tokens_file(self) -> None:
        path = self.tokens_file
        if path.exists():
            try:
                path.unlink()
            except OSError as error:
                logger.warning(
                    "failed to remove freebuff tokens file: %s",
                    error,
                )

    async def _swap_pool(self, tokens: tuple[str, ...]) -> None:
        new_pool = self._build_pool(tokens)
        old_pool = self.pool
        self.pool = new_pool
        self._tokens = tokens
        await old_pool.aclose()

    def _build_pool(self, tokens: tuple[str, ...]) -> CodebuffAccountPool:
        codebuff_token = ",".join(tokens) if tokens else None
        return CodebuffAccountPool(
            replace(self.settings, codebuff_token=codebuff_token),
            default_index=self._active_index,
            on_default_change=self._persist_active_index,
        )

    # ------------------------------------------------------------------
    # Pool delegate
    # ------------------------------------------------------------------

    @property
    def account_count(self) -> int:
        return self.pool.account_count

    @property
    def default_client(self) -> CodebuffClient:
        return self.pool.default_client

    @property
    def default_sessions(self) -> SessionManager:
        return self.pool.default_sessions

    async def acquire_session(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> CodebuffAccountLease:
        return await self.pool.acquire_session(model, messages)

    async def aclose(self) -> None:
        await self.pool.aclose()
