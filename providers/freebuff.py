from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx

from gateway.core.config import HAR_BROWSER_USER_AGENT, Settings
from gateway.core.logging import redact_headers, render_debug
from gateway.compat.models import agent_validation_payload


logger = logging.getLogger("gateway.providers.freebuff")

CODEBUFF_ACCEPT_ENCODING = "gzip, deflate"
CODEBUFF_JSON_USER_AGENT = "Bun/1.3.11"
FREEBUFF_CLI_USER_AGENT = "Freebuff-CLI/0.0.105"
CHAT_COMPLETIONS_USER_AGENT = (
    "ai-sdk/openai-compatible/0.0.0-test/codebuff "
    "ai-sdk/provider-utils/3.0.25 runtime/browser"
)


class CodebuffError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        *,
        retry_after_ms: int | None = None,
        reset_at: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_ms = retry_after_ms
        self.reset_at = reset_at


def is_waiting_room_required(error: CodebuffError) -> bool:
    """True when upstream reports 428 waiting_room_required.

    Upstream returns this when the chat request carries a session that is no
    longer active (e.g. the session was stolen by another client or expired), so
    the caller should refresh the session and retry once instead of failing.
    """
    return error.status_code == 428 or "waiting_room_required" in str(error)


@dataclass
class FreebuffSession:
    instance_id: str
    model: str
    expires_at: str | None = None
    remaining_ms: int | None = None

    @property
    def is_fresh(self) -> bool:
        return self.remaining_ms is None or self.remaining_ms > 60_000


@dataclass
class FreebuffRun:
    run_id: str
    agent_id: str
    started_at: str
    child_run_id: str | None = None
    chat_run_id: str | None = None
    chat_started_at: str | None = None

    @property
    def payload_run_id(self) -> str:
        return self.chat_run_id or self.run_id


@dataclass
class FreebuffSessionLease:
    session: FreebuffSession
    _lock: asyncio.Lock
    _closed: bool = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._lock.release()


class CodebuffClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout, read=None),
            follow_redirects=True,
            proxy=settings.upstream_proxy_url,
            trust_env=False,
        )
        self._agents_validated = False
        self._validate_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(
        self,
        *,
        json_body: bool = False,
        user_agent: str = CODEBUFF_JSON_USER_AGENT,
        require_auth: bool = True,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        if require_auth and not self.settings.codebuff_token:
            raise CodebuffError(
                "No Freebuff token is configured. Add one in Dashboard → Freebuff Tokens.",
                503,
            )

        headers = {
            "Accept": "*/*",
            "Accept-Encoding": CODEBUFF_ACCEPT_ENCODING,
            "Connection": "keep-alive",
            "Host": _host_header(self.settings.codebuff_api_url),
            "User-Agent": user_agent,
        }
        if require_auth:
            headers["Authorization"] = f"Bearer {self.settings.codebuff_token}"
        if json_body:
            headers["Content-Type"] = "application/json"
        if extra:
            headers.update(extra)
        return headers

    async def _json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.settings.codebuff_api_url}{path}"
        request_headers = headers or self._headers(json_body=body is not None)
        max_attempts = max(1, self.settings.retry_attempts)
        last_error: CodebuffError | None = None
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                await asyncio.sleep(_retry_delay(attempt, self.settings))
            try:
                response = await self._client.request(
                    method,
                    url,
                    json=body,
                    headers=request_headers,
                )
            except httpx.RequestError as error:
                raise _network_error(method, url, error) from error
            if self.settings.debug:
                logger.debug(
                    "upstream json request method=%s url=%s headers=%s body=%s",
                    method,
                    url,
                    redact_headers(request_headers),
                    render_debug(body, self.settings.log_body_chars),
                )
                logger.debug(
                    "upstream json response status=%s body=%s",
                    response.status_code,
                    render_debug(response.text, self.settings.log_body_chars),
                )
            if response.status_code >= 400:
                error = _upstream_error(response)
                if (
                    _retriable_for_method(method, response.status_code)
                    and attempt < max_attempts
                ):
                    logger.warning(
                        "upstream %s %s status=%s retrying attempt=%s/%s: %s",
                        method,
                        path,
                        response.status_code,
                        attempt,
                        max_attempts,
                        error,
                    )
                    last_error = error
                    continue
                raise error
            if not response.content:
                return {}
            return response.json()
        assert last_error is not None
        raise last_error

    async def validate_agents(self) -> None:
        if self._agents_validated:
            return
        async with self._validate_lock:
            if self._agents_validated:
                return
            try:
                data = await self._json(
                    "POST",
                    "/api/agents/validate",
                    body=agent_validation_payload(),
                    headers=self._headers(json_body=True, require_auth=False),
                )
            except CodebuffError:
                logger.warning(
                    "agent validation failed; continuing with server configs",
                    exc_info=self.settings.debug,
                )
                self._agents_validated = True
                return
            error_count = int(data.get("errorCount") or 0)
            if error_count:
                logger.warning(
                    "agent validation returned errors count=%s body=%s",
                    error_count,
                    render_debug(data, self.settings.log_body_chars),
                )
            else:
                logger.info(
                    "agent validation completed configs=%s",
                    len(data.get("configs") or []),
                )
            self._agents_validated = True

    async def health(self) -> dict[str, Any]:
        return await self._json(
            "GET",
            "/api/healthz",
            headers=self._headers(require_auth=False),
        )

    async def get_session(self, instance_id: str | None = None) -> dict[str, Any]:
        headers_extra = {}
        if instance_id:
            headers_extra["x-freebuff-instance-id"] = instance_id
        return await self._json(
            "GET",
            "/api/v1/freebuff/session",
            headers=self._headers(extra=headers_extra),
        )

    async def fetch_free_model_ids(self) -> set[str]:
        """Query upstream for the account's currently available free models.

        `GET /api/v1/freebuff/session` returns `rateLimitsByModel` — the model ids
        this account currently holds a free entitlement for. The always-on baseline
        models (e.g. deepseek-v4-flash, mimo-v2.5) are merged in by
        `models_response`, so callers only see the full catalog filtered to
        availability.
        """
        data = await self.get_session()
        return set(data.get("rateLimitsByModel") or {})

    async def create_session(self, model: str) -> FreebuffSession:
        data = await self._json(
            "POST",
            "/api/v1/freebuff/session",
            headers=self._headers(extra={"x-freebuff-model": model}),
        )
        if data.get("status") == "queued":
            return await self._wait_for_active_session(data, model)
        return self._session_from_data(data, model)

    def _session_from_data(
        self,
        data: dict[str, Any],
        model: str,
        instance_id: str | None = None,
    ) -> FreebuffSession:
        resolved_instance_id = data.get("instanceId") or instance_id
        if data.get("status") != "active" or not resolved_instance_id:
            raise CodebuffError(f"Freebuff session is not active: {data}", 502)
        return FreebuffSession(
            instance_id=resolved_instance_id,
            model=data.get("model") or model,
            expires_at=data.get("expiresAt"),
            remaining_ms=data.get("remainingMs"),
        )

    async def _wait_for_active_session(
        self,
        data: dict[str, Any],
        model: str,
    ) -> FreebuffSession:
        instance_id = data.get("instanceId")
        if not instance_id:
            raise CodebuffError(f"Freebuff queued session id missing: {data}", 502)

        deadline = time.monotonic() + self.settings.request_timeout
        attempts = 0
        while data.get("status") == "queued":
            logger.info(
                "freebuff session queued model=%s instance_id=%s position=%s estimated_wait_ms=%s",
                model,
                instance_id,
                data.get("position"),
                data.get("estimatedWaitMs"),
            )
            if time.monotonic() >= deadline:
                raise CodebuffError(
                    f"Freebuff session did not become active before timeout: {data}",
                    502,
                )
            if attempts:
                await asyncio.sleep(_queue_poll_delay(data.get("estimatedWaitMs")))
            data = await self.get_session(instance_id)
            attempts += 1

        return self._session_from_data(data, model, instance_id=instance_id)

    async def delete_session(self) -> None:
        await self._json(
            "DELETE",
            "/api/v1/freebuff/session",
            headers=self._headers(),
        )
        logger.info("deleted active freebuff session")

    async def get_streak(self) -> dict[str, Any]:
        data = await self._json(
            "GET",
            "/api/v1/freebuff/streak",
            headers=self._headers(),
        )
        logger.info(
            "freebuff streak streak=%s today_used=%s",
            data.get("streak"),
            data.get("todayUsed"),
        )
        return data

    async def request_ads(
        self,
        provider: str,
        messages: list[dict[str, Any]] | None = None,
        surface: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "provider": provider,
            "messages": _ad_messages(messages),
            "sessionId": self.settings.session_id,
            "device": {
                "os": self.settings.os_name,
                "timezone": self.settings.timezone,
                "locale": self.settings.locale,
            },
            "userAgent": HAR_BROWSER_USER_AGENT,
        }
        if surface:
            body["surface"] = surface
        return await self._json(
            "POST",
            "/api/v1/ads",
            body=body,
            headers=self._headers(
                json_body=True,
                user_agent=FREEBUFF_CLI_USER_AGENT,
            ),
        )

    async def request_ad_chain(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        surface: str | None = None,
    ) -> None:
        for provider in self.settings.ad_providers:
            try:
                ads_data = await self.request_ads(
                    provider,
                    messages=messages,
                    surface=surface,
                )
                ads = ads_data.get("ads") or []
                ad = ads[0] if ads else None
                logger.info(
                    "ads provider=%s messages=%s count=%s selected=%s",
                    provider,
                    len(messages or []),
                    len(ads),
                    bool(ad),
                )
                if not ad:
                    continue
                await self.report_zeroclick_impressions(
                    list(ad.get("impressionIds") or [])
                )
                await self.report_codebuff_impression(ad.get("impUrl") or "")
                return
            except CodebuffError as error:
                logger.warning(
                    "ads provider=%s failed; continuing without blocking chat: %s",
                    provider,
                    error,
                    exc_info=self.settings.debug,
                )

    async def report_zeroclick_impressions(self, ids: list[str]) -> None:
        if not ids:
            return
        url = f"{self.settings.zeroclick_api_url}/api/v2/impressions"
        try:
            response = await self._client.post(
                url,
                json={"ids": ids},
                headers={
                    "Content-Type": "application/json",
                    "Accept": "*/*",
                    "User-Agent": CODEBUFF_JSON_USER_AGENT,
                },
            )
        except httpx.RequestError as error:
            raise _network_error("POST", url, error) from error
        if self.settings.debug:
            logger.debug(
                "zeroclick impression ids=%s status=%s body=%s",
                ids,
                response.status_code,
                render_debug(response.text, self.settings.log_body_chars),
            )
        if response.status_code >= 400:
            raise CodebuffError(
                f"Zeroclick impression failed: {response.status_code} {response.text[:500]}",
                502,
            )

    async def report_codebuff_impression(self, imp_url: str) -> None:
        if not imp_url:
            return
        await self._json(
            "POST",
            "/api/v1/ads/impression",
            body={"impUrl": imp_url, "mode": "LITE"},
            headers=self._headers(
                json_body=True,
                user_agent=FREEBUFF_CLI_USER_AGENT,
            ),
        )

    async def start_run(
        self,
        agent_id: str,
        ancestor_run_ids: list[str] | None = None,
    ) -> str:
        data = await self._json(
            "POST",
            "/api/v1/agent-runs",
            body={
                "action": "START",
                "agentId": agent_id,
                "ancestorRunIds": ancestor_run_ids or [],
            },
        )
        run_id = data.get("runId")
        if not run_id:
            raise CodebuffError(f"Codebuff run id missing: {data}", 502)
        logger.info(
            "agent run started agent_id=%s run_id=%s ancestors=%s",
            agent_id,
            run_id,
            ancestor_run_ids or [],
        )
        return run_id

    async def record_run_step(
        self,
        run_id: str,
        *,
        step_number: int,
        message_id: str | None,
        start_time: str,
        child_run_ids: list[str] | None = None,
    ) -> None:
        await self._json(
            "POST",
            f"/api/v1/agent-runs/{run_id}/steps",
            body={
                "stepNumber": step_number,
                "credits": 0,
                "childRunIds": child_run_ids or [],
                "messageId": message_id,
                "status": "completed",
                "startTime": start_time,
            },
        )
        logger.info(
            "agent run step recorded run_id=%s step=%s message_id=%s children=%s",
            run_id,
            step_number,
            message_id,
            child_run_ids or [],
        )

    async def finish_run(self, run_id: str, *, total_steps: int) -> None:
        await self._json(
            "POST",
            "/api/v1/agent-runs",
            body={
                "action": "FINISH",
                "runId": run_id,
                "status": "completed",
                "totalSteps": total_steps,
                "directCredits": 0,
                "totalCredits": 0,
            },
        )
        logger.info("agent run finished run_id=%s total_steps=%s", run_id, total_steps)

    async def chat_events(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        url = f"{self.settings.codebuff_api_url}/api/v1/chat/completions"
        request_headers = self._headers(
            json_body=True,
            user_agent=CHAT_COMPLETIONS_USER_AGENT,
        )
        max_attempts = max(1, self.settings.retry_attempts)
        last_error: CodebuffError | None = None
        current_payload = payload
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                # Only halve max_tokens on 429 (quota); retry 5xx as-is
                if last_error is not None and last_error.status_code == 429:
                    current_payload = _reduce_max_tokens(current_payload)
                delay = _retry_delay(attempt, self.settings)
                logger.info(
                    "chat upstream retry attempt=%s/%s delay=%.1fs",
                    attempt,
                    max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
            try:
                async with self._client.stream(
                    "POST",
                    url,
                    json=current_payload,
                    headers=request_headers,
                ) as response:
                    if self.settings.debug:
                        logger.debug(
                            "chat stream request url=%s headers=%s payload=%s",
                            url,
                            redact_headers(request_headers),
                            render_debug(current_payload, self.settings.log_body_chars),
                        )
                        logger.debug(
                            "chat stream response status=%s headers=%s",
                            response.status_code,
                            redact_headers(dict(response.headers)),
                        )
                    if response.status_code >= 400:
                        text = await response.aread()
                        error = _upstream_error(
                            response,
                            body=text,
                            prefix="Codebuff chat failed",
                        )
                        if (
                            _is_retriable_status(response.status_code)
                            and attempt < max_attempts
                        ):
                            logger.warning(
                                "chat upstream status=%s retrying attempt=%s/%s: %s",
                                response.status_code,
                                attempt,
                                max_attempts,
                                error,
                            )
                            last_error = error
                            continue
                        raise error
                    async for line in response.aiter_lines():
                        if self.settings.debug:
                            logger.debug(
                                "chat stream line=%s",
                                render_debug(line, self.settings.log_body_chars),
                            )
                        yield line
                    return
            except httpx.RequestError as error:
                raise _network_error("POST", url, error) from error
        assert last_error is not None
        raise last_error


class SessionManager:
    def __init__(self, client: CodebuffClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self._sessions: dict[str, FreebuffSession] = {}
        self._lock = asyncio.Lock()

    async def ensure_session(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> FreebuffSession:
        async with self._lock:
            return await self._ensure_session_locked(model, messages)

    async def acquire_session(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> FreebuffSessionLease:
        await self._lock.acquire()
        try:
            session = await self._ensure_session_locked(model, messages)
        except Exception:
            self._lock.release()
            raise
        return FreebuffSessionLease(session=session, _lock=self._lock)

    async def _ensure_session_locked(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> FreebuffSession:
        cached = self._sessions.get(model)
        if cached and cached.is_fresh:
            # FreeBuff's session inspection is routinely slower than a whole
            # terminal turn.  ``remaining_ms`` is already supplied when the
            # session is created/discovered, so reuse it directly while fresh.
            # A genuinely stale session is handled by the existing 428 refresh
            # path during the chat request; probing before every turn only
            # adds a 10–20 second round trip.
            logger.info(
                "reuse cached freebuff session model=%s instance_id=%s remaining_ms=%s",
                model,
                cached.instance_id,
                cached.remaining_ms,
            )
            return cached

        active_session = await self._delete_locked_session(model)
        if active_session:
            return active_session
        await self._request_ads_and_streak(surface="waiting_room")

        try:
            session = await self.client.create_session(model)
        except CodebuffError as error:
            if "model_locked" not in str(error):
                raise
            logger.info(
                "freebuff session locked during create; delete and retry model=%s",
                model,
            )
            await self.client.delete_session()
            self._sessions.clear()
            await self._request_ads_and_streak(surface="waiting_room")
            session = await self.client.create_session(model)
        self._sessions[model] = session
        logger.debug(
            "created freebuff session model=%s instance_id=%s remaining_ms=%s",
            model,
            session.instance_id,
            session.remaining_ms,
        )
        return session

    async def _request_ads_and_streak(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        surface: str | None = None,
    ) -> None:
        for provider in self.settings.ad_providers:
            try:
                ads_data = await self.client.request_ads(
                    provider,
                    messages=messages,
                    surface=surface,
                )
                ads = ads_data.get("ads") or []
                ad = ads[0] if ads else None
                logger.info(
                    "ads provider=%s messages=%s count=%s selected=%s",
                    provider,
                    len(messages or []),
                    len(ads),
                    bool(ad),
                )
                if not ad:
                    continue
                await self.client.get_streak()
                await self.client.report_zeroclick_impressions(
                    list(ad.get("impressionIds") or [])
                )
                await self.client.report_codebuff_impression(ad.get("impUrl") or "")
                return
            except CodebuffError as error:
                logger.warning(
                    "ads provider=%s failed; continuing without blocking chat: %s",
                    provider,
                    error,
                    exc_info=self.settings.debug,
                )

    async def refresh_session(self, model: str) -> FreebuffSession:
        """Create a brand-new upstream session, clearing the cache and locked sessions.

        The caller must already hold the session lock (inside a lease). Called to
        recover from 428 waiting_room_required: the old session is no longer
        active upstream, so we clear the cache, release the locked session (if
        any), then create a new session and cache it.
        """
        self._sessions.clear()
        try:
            await self.client.delete_session()
        except CodebuffError as error:
            logger.debug(
                "could not delete upstream session during refresh: %s",
                error,
                exc_info=self.settings.debug,
            )
        await self._request_ads_and_streak(surface="waiting_room")
        try:
            session = await self.client.create_session(model)
        except CodebuffError as error:
            if "model_locked" not in str(error):
                raise
            logger.info(
                "freebuff session locked during refresh; delete and retry model=%s",
                model,
            )
            await self.client.delete_session()
            self._sessions.clear()
            session = await self.client.create_session(model)
        self._sessions[model] = session
        logger.info(
            "refreshed freebuff session model=%s instance_id=%s remaining_ms=%s",
            model,
            session.instance_id,
            session.remaining_ms,
        )
        return session

    async def _delete_locked_session(
        self,
        requested_model: str,
    ) -> FreebuffSession | None:
        try:
            data = await self.client.get_session()
        except CodebuffError:
            logger.debug(
                "could not inspect active freebuff session before create",
                exc_info=self.settings.debug,
            )
            return None

        if data.get("status") != "active":
            return None

        current_model = data.get("model")
        instance_id = data.get("instanceId")
        if current_model == requested_model and instance_id:
            session = FreebuffSession(
                instance_id=instance_id,
                model=current_model,
                expires_at=data.get("expiresAt"),
                remaining_ms=data.get("remainingMs"),
            )
            self._sessions[requested_model] = session
            logger.info(
                "discovered active freebuff session model=%s instance_id=%s remaining_ms=%s",
                requested_model,
                session.instance_id,
                session.remaining_ms,
            )
            return session

        if not current_model or current_model == requested_model:
            return None

        logger.info(
            "switch freebuff session current_model=%s requested_model=%s instance_id=%s",
            current_model,
            requested_model,
            instance_id,
        )
        await self.client.delete_session()
        self._sessions.clear()
        return None


@dataclass
class CodebuffAccount:
    client: CodebuffClient
    sessions: SessionManager
    busy: bool = False
    cooldown_until: float = 0.0
    reset_at: str | None = None
    quota_reset_epoch: float = 0.0


@dataclass
class CodebuffAccountLease:
    client: CodebuffClient
    session: FreebuffSession
    _session_lease: FreebuffSessionLease
    _pool: "CodebuffAccountPool"
    _account_index: int
    _closed: bool = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._session_lease.aclose()
        await self._pool.release(self._account_index)

    def mark_rate_limited(
        self, duration: float, *, error: CodebuffError | None = None
    ) -> None:
        """Mark this account as 429'd upstream; no new requests for `duration` seconds.

        `error` carries the upstream 429 window (`retryAfterMs` / `resetAt`) so
        the pool honors the real reset time instead of the fixed cooldown.
        """
        self._pool.mark_rate_limited(self._account_index, duration, error=error)

    async def refresh_session(self, model: str) -> FreebuffSession:
        """Create a brand-new upstream session for this account.

        Used to recover when upstream returns 428 waiting_room_required (the old
        session was stolen/expired). The lease still holds the session lock, so
        the account is not released — only the session is swapped in the lease.
        `self.session` is updated so later callers never use the stale instance.
        """
        session = await self._pool.refresh_session(self._account_index, model)
        self.session = session
        return session


class CodebuffAccountPool:
    def __init__(
        self,
        settings: Settings,
        *,
        default_index: int = 0,
        on_default_change: Any | None = None,
    ) -> None:
        tokens = settings.codebuff_tokens or (None,)
        self._accounts: list[CodebuffAccount] = []
        for token in tokens:
            account_settings = replace(settings, codebuff_token=token)
            client = CodebuffClient(account_settings)
            self._accounts.append(
                CodebuffAccount(
                    client=client,
                    sessions=SessionManager(client, account_settings),
                )
            )
        # The active account is sticky. We only move it after that account
        # actually fails, rather than round-robining every normal request.
        self._default_index = default_index % len(self._accounts)
        self._on_default_change = on_default_change
        self._account_cooldown = settings.account_cooldown
        self._condition = asyncio.Condition()

    @property
    def account_count(self) -> int:
        return len(self._accounts)

    @property
    def default_client(self) -> CodebuffClient:
        return self._accounts[0].client

    @property
    def default_sessions(self) -> SessionManager:
        return self._accounts[0].sessions

    async def aclose(self) -> None:
        await asyncio.gather(
            *(account.client.aclose() for account in self._accounts),
            return_exceptions=True,
        )
        # Wake up any request waiting for a free account on the old pool (when the
        # pool is replaced by a token reload) to avoid hanging forever.
        async with self._condition:
            self._condition.notify_all()

    async def acquire_session(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> CodebuffAccountLease:
        # Quota belongs to an account, not the whole pool.  Do not let a 429
        # from token A prevent a healthy token B from being tried.
        if self._all_accounts_rate_limited():
            raise self._all_accounts_rate_limited_error()

        # A 429 can happen while obtaining the upstream Freebuff session,
        # before a CodebuffAccountLease exists. Previously that immediately
        # escaped to the caller, so a newly-added second token was never tried.
        # Mark that account unavailable and try each remaining account once.
        last_rate_limit: CodebuffError | None = None
        for _ in range(self.account_count):
            account_index = await self._reserve_account()
            account = self._accounts[account_index]
            try:
                session_lease = await account.sessions.acquire_session(model, messages)
            except CodebuffError as error:
                await self.release(account_index)
                if error.status_code not in {401, 403, 429}:
                    raise
                self.mark_rate_limited(
                    account_index, self._account_cooldown, error=error
                )
                last_rate_limit = error
                continue
            except Exception:
                await self.release(account_index)
                raise
            return CodebuffAccountLease(
                client=account.client,
                session=session_lease.session,
                _session_lease=session_lease,
                _pool=self,
                _account_index=account_index,
            )
        assert last_rate_limit is not None
        raise last_rate_limit

    async def release(self, account_index: int) -> None:
        async with self._condition:
            self._accounts[account_index].busy = False
            self._condition.notify(1)

    def mark_rate_limited(
        self,
        account_index: int,
        duration: float,
        *,
        error: CodebuffError | None = None,
    ) -> None:
        """Mark this account as 429'd upstream; no new requests until the window passes.

        The upstream 429 body carries the real reset window (`retryAfterMs` /
        `resetAt`); honor that instead of the fixed `settings.account_cooldown`
        so the account stays unavailable until Freebuff actually refills. The
        reset time is kept at pool level too, so later requests fail fast with
        a friendly message instead of retrying into the wall.
        """
        duration = max(0.0, duration)
        if (
            error is not None
            and isinstance(error.retry_after_ms, int)
            and error.retry_after_ms > 0
        ):
            duration = error.retry_after_ms / 1000.0
        account = self._accounts[account_index]
        account.cooldown_until = time.monotonic() + duration
        if error is not None and error.reset_at:
            account.reset_at = error.reset_at
            # Parse the upstream reset window into a wall-clock epoch once, so
            # acquire_session can gate on it directly. `resetAt` is normally
            # hours away while `retryAfterMs` is only ~60s, so the epoch (not
            # cooldown) is what keeps requests from hitting the wall again.
            normalized = (
                error.reset_at[:-1] + "+00:00"
                if error.reset_at.endswith("Z")
                else error.reset_at
            )
            try:
                account.quota_reset_epoch = datetime.fromisoformat(normalized).timestamp()
            except ValueError:
                account.quota_reset_epoch = time.time() + duration
        self._default_index = (account_index + 1) % len(self._accounts)
        if self._on_default_change is not None:
            try:
                self._on_default_change(self._default_index)
            except Exception:
                logger.exception("failed to persist freebuff default account")
        logger.warning(
            "freebuff account=%s unavailable for %.0fs; switched default account=%s",
            account_index,
            duration,
            self._default_index,
        )

    async def refresh_session(self, account_index: int, model: str) -> FreebuffSession:
        return await self._accounts[account_index].sessions.refresh_session(model)

    async def _reserve_account(self) -> int:
        async with self._condition:
            while True:
                account_index = self._next_available_index()
                if account_index is not None:
                    self._accounts[account_index].busy = True
                    return account_index
                await self._condition.wait()

    def _next_available_index(self) -> int | None:
        account_count = len(self._accounts)
        # Prefer the sticky default for every normal request. A different
        # account is only borrowed while the default is busy; that does not
        # alter the default flag.
        for offset in range(account_count):
            account_index = (self._default_index + offset) % account_count
            account = self._accounts[account_index]
            if not account.busy and not self._account_rate_limited(account):
                return account_index
        return None

    @staticmethod
    def _account_rate_limited(account: CodebuffAccount) -> bool:
        return (
            account.cooldown_until > time.monotonic()
            or account.quota_reset_epoch > time.time()
        )

    def _all_accounts_rate_limited(self) -> bool:
        return bool(self._accounts) and all(
            self._account_rate_limited(account) for account in self._accounts
        )

    def _all_accounts_rate_limited_error(self) -> CodebuffError:
        limited = [account for account in self._accounts if self._account_rate_limited(account)]
        reset_account = min(
            (account for account in limited if account.quota_reset_epoch > time.time()),
            key=lambda account: account.quota_reset_epoch,
            default=None,
        )
        if reset_account is not None:
            when = reset_account.reset_at or f"in {int(reset_account.quota_reset_epoch - time.time())}s"
            message = (
                f"All Freebuff tokens have exhausted their quota; the next refill is {when} "
                "(UTC). Add another Freebuff token in Dashboard → Freebuff Tokens "
                "or configure a lower-priority provider."
            )
            return CodebuffError(message, 429, reset_at=reset_account.reset_at)
        return CodebuffError(
            "All Freebuff tokens are temporarily rate-limited. "
            "Configure a lower-priority provider or retry shortly.",
            429,
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def _host_header(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or "www.codebuff.com"


def _queue_poll_delay(estimated_wait_ms: Any) -> float:
    if isinstance(estimated_wait_ms, int | float) and estimated_wait_ms > 0:
        return min(max(float(estimated_wait_ms) / 1000.0, 0.25), 2.0)
    return 0.25


def _network_error(method: str, url: str, error: httpx.RequestError) -> CodebuffError:
    detail = str(error).strip()
    suffix = f": {detail}" if detail else ""
    return CodebuffError(
        f"Codebuff request failed: {method} {url} network error "
        f"({type(error).__name__}){suffix}",
        502,
    )


def _upstream_error(
    response: httpx.Response,
    *,
    body: bytes | None = None,
    prefix: str = "Codebuff request failed",
) -> CodebuffError:
    raw_text = (
        body.decode("utf-8", errors="replace")
        if body is not None
        else response.text
    )
    text = raw_text[:500]
    if response.status_code == 409:
        try:
            data = (
                response.json()
                if body is None
                else httpx.Response(
                    response.status_code,
                    content=body,
                    headers=response.headers,
                ).json()
            )
        except ValueError:
            data = {}
        if data.get("error") == "session_model_mismatch":
            upstream_message = data.get("message") or text
            return CodebuffError(
                "Codebuff 409 session_model_mismatch: "
                f"{upstream_message} The current IP/region is restricted; "
                "please retry from a US server or a US egress IP.",
                409,
            )

    # Keep upstream 4xx (e.g. 429 rate limit); only 5xx is normalized to 502.
    # That way clients (e.g. Claude Code) retry 429 with their own rate-limit logic.
    status_code = 502 if response.status_code >= 500 else response.status_code

    # Parse the upstream rate-limit window (retryAfterMs / resetAt) from the 429
    # body so the pool can honor the real reset time instead of a fixed cooldown.
    retry_after_ms: int | None = None
    reset_at: str | None = None
    if response.status_code == 429:
        try:
            data = (
                response.json()
                if body is None
                else httpx.Response(
                    response.status_code,
                    content=body,
                    headers=response.headers,
                ).json()
            )
        except ValueError:
            data = {}
        retry_after_ms = data.get("retryAfterMs")
        reset_at = data.get("resetAt")
        if not isinstance(retry_after_ms, int):
            retry_after_ms = None
        if not isinstance(reset_at, str) or not reset_at:
            reset_at = None

    return CodebuffError(
        f"{prefix}: {response.status_code} {text}",
        status_code,
        retry_after_ms=retry_after_ms,
        reset_at=reset_at,
    )


def _is_retriable_status(status_code: int) -> bool:
    return status_code == 408 or status_code == 429 or 500 <= status_code < 600


def _retriable_for_method(method: str, status_code: int) -> bool:
    """GET may retry 429/5xx; POST/PUT/DELETE only retry 429
    (a rate limit means the request was not processed, while 5xx may have side effects)."""
    if method in {"GET", "HEAD"}:
        return _is_retriable_status(status_code)
    return status_code == 429


def _retry_delay(attempt: int, settings: Settings) -> float:
    """Exponential backoff. `attempt` is the attempt number (first retry attempt=2)."""
    exponent = max(0, attempt - 2)
    delay = settings.retry_base_delay * (2**exponent)
    return min(delay, settings.retry_max_delay)


def _reduce_max_tokens(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy payload with max_tokens / max_completion_tokens halved."""
    reduced = dict(payload)
    for key in ("max_tokens", "max_completion_tokens"):
        value = reduced.get(key)
        if isinstance(value, int) and value > 512:
            reduced[key] = max(value // 2, 512)
    return reduced


def _ad_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    return [
        {
            "role": _ad_message_role(message.get("role")),
            "content": _ad_message_content(message.get("content")),
        }
        for message in messages or []
    ]


def _ad_message_role(role: Any) -> str:
    if role == "developer":
        return "system"
    return str(role or "user")


def _ad_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = [
            str(part.get("text"))
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return "\n".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return str(content)
