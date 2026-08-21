"""FreeBuff token-pool handoff used before provider-priority fallback."""

from __future__ import annotations

from typing import Any

from providers.freebuff import CodebuffClient, CodebuffError, FreebuffRun


class FreebuffDispatchFailover:
    """Replace one FreeBuff dispatch with the next usable token dispatch.

    This class owns the *token* segment of the protected sequence.  When all
    accounts are unavailable, ``prepare_freebuff_dispatch`` raises and the
    route hands control to the provider-priority policy.
    """

    def __init__(
        self,
        *,
        accounts: Any,
        model_config: Any,
        body: dict[str, Any],
        messages: list[dict[str, Any]],
        settings: Any,
        lease: Any,
        client: CodebuffClient,
        run: FreebuffRun,
        payload: dict[str, Any],
    ) -> None:
        self._accounts = accounts
        self._model_config = model_config
        self._body = body
        self._messages = messages
        self._settings = settings
        self.lease = lease
        self.client = client
        self.run = run
        self.payload = payload

    @staticmethod
    def _overlay_active_request(
        fresh_payload: dict[str, Any], current_payload: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(fresh_payload)
        for key in ("messages", "response_format", "temperature"):
            if key in current_payload:
                merged[key] = current_payload[key]
        return merged

    async def recover_session(self, current_payload: dict[str, Any]) -> dict[str, Any]:
        """Refresh the active account for a 428 without changing accounts."""
        from gateway.services.chat_service import build_payload

        session = await self.lease.refresh_session(self._model_config.session_id)
        fresh_payload = build_payload(
            {**self._body, "messages": self._messages},
            session=session,
            run=self.run,
            client_id=self._settings.client_id,
            upstream_model_id=self._model_config.upstream_id,
            max_tokens_cap=self._settings.max_tokens,
        )
        self.payload = self._overlay_active_request(fresh_payload, current_payload)
        return self.payload

    async def failover(
        self, error: CodebuffError, current_payload: dict[str, Any]
    ) -> tuple[CodebuffClient, dict[str, Any]]:
        """Retire the failed token and prepare the next usable token."""
        from gateway.services.chat_service import (
            prepare_freebuff_dispatch,
            schedule_finalize_run,
        )

        previous_lease = self.lease
        previous_client = self.client
        previous_run = self.run
        previous_lease.mark_rate_limited(self._settings.account_cooldown, error=error)
        await previous_lease.aclose()
        schedule_finalize_run(previous_client, previous_run, None)

        lease, client, run, fresh_payload = await prepare_freebuff_dispatch(
            self._accounts,
            self._model_config,
            self._body,
            self._messages,
            self._settings,
        )
        self.lease = lease
        self.client = client
        self.run = run
        self.payload = self._overlay_active_request(fresh_payload, current_payload)
        return client, self.payload

    async def aclose(self) -> None:
        await self.lease.aclose()
