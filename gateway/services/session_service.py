from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from providers.freebuff import (
    CodebuffAccountLease,
    CodebuffAccountPool,
    CodebuffClient,
    FreebuffSession,
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

    Tokens come exclusively from the dashboard-managed config file
    (`settings.tokens_file`, gitignored, written by the dashboard). When it
    changes, the pool is rebuilt right away.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._reload_lock = asyncio.Lock()
        self._tokens = self._load_tokens()
        self._active_index = self._read_active_index()
        self.pool = self._build_pool(self._tokens)
        self._hydrate_cached_sessions(self.pool, self._tokens)

    # ------------------------------------------------------------------
    # Token persistence
    # ------------------------------------------------------------------

    @property
    def tokens_file(self) -> Path:
        return Path(self.settings.tokens_file)

    @property
    def sessions_file(self) -> Path:
        """Runtime-only FreeBuff session cache; it never contains tokens."""
        return Path("data/freebuff-sessions.json")

    @staticmethod
    def _token_fingerprint(token: str | None) -> str | None:
        if not token:
            return None
        return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]

    def _read_session_cache(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.sessions_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("failed to read FreeBuff session cache: %s", error)
            return {}
        sessions = raw.get("sessions") if isinstance(raw, dict) else None
        return sessions if isinstance(sessions, dict) else {}

    def _write_session_cache(self, sessions: dict[str, Any]) -> None:
        path = self.sessions_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": 1, "sessions": sessions}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _hydrate_cached_sessions(
        self,
        pool: CodebuffAccountPool,
        tokens: tuple[str, ...],
    ) -> None:
        cache = self._read_session_cache()
        restored = 0
        now = time.time()
        accounts = getattr(pool, "_accounts", [])
        for index, token in enumerate(tokens):
            if index >= len(accounts):
                break
            fingerprint = self._token_fingerprint(token)
            entry = cache.get(fingerprint or "")
            models = entry.get("models") if isinstance(entry, dict) else None
            if not isinstance(models, dict):
                continue
            for model, item in models.items():
                if not isinstance(model, str) or not isinstance(item, dict):
                    continue
                instance_id = item.get("instance_id")
                remaining_ms = item.get("remaining_ms")
                stored_at = item.get("stored_at")
                if not isinstance(instance_id, str) or not isinstance(remaining_ms, int):
                    continue
                elapsed_ms = max(0, int((now - float(stored_at or now)) * 1000))
                remaining_ms -= elapsed_ms
                if remaining_ms <= 60_000:
                    continue
                accounts[index].sessions._sessions[model] = FreebuffSession(
                    instance_id=instance_id,
                    model=model,
                    expires_at=item.get("expires_at") if isinstance(item.get("expires_at"), str) else None,
                    remaining_ms=remaining_ms,
                )
                restored += 1
        if restored:
            logger.info("restored FreeBuff session cache entries=%s", restored)

    def _persist_cached_session(self, lease: CodebuffAccountLease, model: str) -> None:
        if lease._account_index >= len(self._tokens):
            return
        fingerprint = self._token_fingerprint(self._tokens[lease._account_index])
        session = lease.session
        if fingerprint is None or not isinstance(session.remaining_ms, int):
            return
        cache = self._read_session_cache()
        entry = cache.setdefault(fingerprint, {"models": {}})
        models = entry.setdefault("models", {})
        current = models.get(model)
        if isinstance(current, dict) and current.get("instance_id") == session.instance_id:
            return
        models[model] = {
            "instance_id": session.instance_id,
            "expires_at": session.expires_at,
            "remaining_ms": session.remaining_ms,
            "stored_at": time.time(),
        }
        try:
            self._write_session_cache(cache)
            logger.info("persisted FreeBuff session cache model=%s", model)
        except OSError as error:
            logger.warning("failed to persist FreeBuff session cache: %s", error)

    def _load_tokens(self) -> tuple[str, ...]:
        file_tokens = self._read_tokens_file()
        if file_tokens:
            logger.info(
                "freebuff tokens loaded source=file count=%s",
                len(file_tokens),
            )
            return file_tokens
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
        return "none"

    @property
    def tokens(self) -> list[str]:
        """Tokens currently in use (pool order)."""
        return list(self._tokens)

    async def update_tokens(self, tokens: list[str]) -> list[str]:
        """Write tokens to the file and reload the pool (called from API/dashboard).

        An empty list deletes the config file and clears the active pool.
        Returns the active list.
        """
        normalized = _normalize_tokens(tokens)
        async with self._reload_lock:
            if not normalized:
                await self._remove_tokens_file()
                active = ()
            else:
                self._active_index = min(self._active_index, len(normalized) - 1)
                self._write_tokens_file(normalized)
                logger.info("freebuff tokens updated count=%s", len(normalized))
                active = normalized
            await self._swap_pool(active)
        return list(active)

    async def remove_token(self, index: int) -> list[str]:
        """Remove one token by index; returns the remaining list.

        If the file empties, the active pool becomes empty.
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
                remaining = ()
            logger.info("freebuff token removed; remaining=%s", len(remaining))
            await self._swap_pool(remaining)
        return list(remaining)

    async def clear_tokens(self) -> None:
        """Delete the dashboard token config file and clear the active pool."""
        async with self._reload_lock:
            await self._remove_tokens_file()
            logger.info("freebuff tokens cleared")
            await self._swap_pool(())

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
        self._hydrate_cached_sessions(new_pool, tokens)
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
        lease = await self.pool.acquire_session(model, messages)
        self._persist_cached_session(lease, model)
        return lease

    async def aclose(self) -> None:
        await self.pool.aclose()
