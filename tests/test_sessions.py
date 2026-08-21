import asyncio
import unittest
from datetime import datetime
from unittest.mock import patch

from providers.freebuff import (
    CodebuffAccountPool,
    CodebuffError,
    FreebuffSession,
    SessionManager,
)
from gateway.core.config import Settings


class RefreshClient:
    """delete_session luôn lỗi (không có session active) nhưng create vẫn được."""

    def __init__(self) -> None:
        self.calls = []

    async def delete_session(self) -> None:
        self.calls.append(("delete_session",))
        raise CodebuffError("Codebuff request failed: 404 no active session", 404)

    async def request_ads(self, provider, messages=None, *, surface=None) -> dict:
        self.calls.append(("request_ads", provider, messages or [], surface))
        return {"ads": []}

    async def get_streak(self) -> dict:
        self.calls.append(("get_streak",))
        return {"streak": 0}

    async def report_zeroclick_impressions(self, ids) -> None:
        self.calls.append(("report_zeroclick_impressions", ids))

    async def report_codebuff_impression(self, imp_url) -> None:
        self.calls.append(("report_codebuff_impression", imp_url))

    async def create_session(self, model):
        self.calls.append(("create_session", model))
        return FreebuffSession(
            instance_id="fresh-instance",
            model=model,
            remaining_ms=3_000_000,
        )


class LockedRefreshClient(RefreshClient):
    """delete lần đầu lỗi (không có session), create lần đầu model_locked;
    delete lần hai thành công thì create thành công."""

    def __init__(self) -> None:
        super().__init__()
        self.delete_attempts = 0

    async def delete_session(self) -> None:
        self.delete_attempts += 1
        self.calls.append(("delete_session",))
        if self.delete_attempts == 1:
            raise CodebuffError("Codebuff request failed: 404 no active session", 404)

    async def create_session(self, model):
        self.calls.append(("create_session", model))
        if self.delete_attempts < 2:
            raise CodebuffError(
                'Codebuff request failed: 409 {"status":"model_locked"}',
                502,
            )
        return FreebuffSession(
            instance_id="fresh-instance",
            model=model,
            remaining_ms=3_000_000,
        )


class SwitchModelClient:
    def __init__(self) -> None:
        self.deleted = False
        self.calls = []

    async def get_session(self, instance_id=None):
        self.calls.append(("get_session", instance_id))
        if self.deleted:
            return {"status": "none"}
        return {
            "status": "active",
            "instanceId": "deepseek-instance",
            "model": "deepseek/deepseek-v4-pro",
            "expiresAt": "2026-05-23T15:27:34.581Z",
            "remainingMs": 3_000_000,
        }

    async def delete_session(self) -> None:
        self.calls.append(("delete_session",))
        self.deleted = True

    async def request_ad_chain(self, messages=None, *, surface=None) -> None:
        self.calls.append(("request_ad_chain", messages or [], surface))

    async def request_ads(self, provider, messages=None, *, surface=None) -> dict:
        self.calls.append(("request_ads", provider, messages or [], surface))
        return {"ads": []}

    async def get_streak(self) -> dict:
        self.calls.append(("get_streak",))
        return {"streak": 0}

    async def report_zeroclick_impressions(self, ids) -> None:
        self.calls.append(("report_zeroclick_impressions", ids))

    async def report_codebuff_impression(self, imp_url) -> None:
        self.calls.append(("report_codebuff_impression", imp_url))

    async def create_session(self, model):
        self.calls.append(("create_session", model))
        if not self.deleted:
            raise CodebuffError(
                'Codebuff request failed: 409 {"status":"model_locked"}',
                502,
            )
        return FreebuffSession(
            instance_id="kimi-instance",
            model=model,
            remaining_ms=3_000_000,
        )


class LeaseSwitchModelClient:
    def __init__(self) -> None:
        self.current_model = "deepseek/deepseek-v4-flash"
        self.calls = []

    async def get_session(self, instance_id=None):
        self.calls.append(("get_session", instance_id, self.current_model))
        return {
            "status": "active",
            "instanceId": f"{self.current_model}-instance",
            "model": self.current_model,
            "remainingMs": 3_000_000,
        }

    async def delete_session(self) -> None:
        self.calls.append(("delete_session", self.current_model))
        self.current_model = ""

    async def request_ad_chain(self, messages=None, *, surface=None) -> None:
        self.calls.append(("request_ad_chain", messages or [], surface))

    async def request_ads(self, provider, messages=None, *, surface=None) -> dict:
        self.calls.append(("request_ads", provider, messages or [], surface))
        return {"ads": []}

    async def get_streak(self) -> dict:
        self.calls.append(("get_streak",))
        return {"streak": 0}

    async def report_zeroclick_impressions(self, ids) -> None:
        self.calls.append(("report_zeroclick_impressions", ids))

    async def report_codebuff_impression(self, imp_url) -> None:
        self.calls.append(("report_codebuff_impression", imp_url))

    async def create_session(self, model):
        self.calls.append(("create_session", model))
        self.current_model = model
        return FreebuffSession(
            instance_id=f"{model}-instance",
            model=model,
            remaining_ms=3_000_000,
        )


class PoolClient:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    async def get_session(self, instance_id=None):
        token = self.settings.codebuff_token
        return {
            "status": "active",
            "instanceId": f"{token}-instance",
            "model": "deepseek/deepseek-v4-flash",
            "remainingMs": 3_000_000,
        }


class RateLimitedSessionManager:
    async def acquire_session(self, model, messages=None):
        raise CodebuffError("quota exhausted", 429)


class SessionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_session_forces_fresh_session_when_delete_fails(self):
        client = RefreshClient()
        manager = SessionManager(
            client,
            Settings(codebuff_token="token", local_api_key=None),
        )

        session = await manager.refresh_session("deepseek/deepseek-v4-flash")

        self.assertEqual(session.instance_id, "fresh-instance")
        self.assertEqual(
            client.calls,
            [
                ("delete_session",),
                ("request_ads", "gravity", [], "waiting_room"),
                ("request_ads", "zeroclick", [], "waiting_room"),
                ("create_session", "deepseek/deepseek-v4-flash"),
            ],
        )
        self.assertEqual(
            manager._sessions["deepseek/deepseek-v4-flash"].instance_id,
            "fresh-instance",
        )

    async def test_refresh_session_recovers_from_model_locked(self):
        client = LockedRefreshClient()
        manager = SessionManager(
            client,
            Settings(codebuff_token="token", local_api_key=None),
        )

        session = await manager.refresh_session("moonshotai/kimi-k2.6")

        self.assertEqual(session.instance_id, "fresh-instance")
        self.assertEqual(
            client.calls,
            [
                ("delete_session",),
                ("request_ads", "gravity", [], "waiting_room"),
                ("request_ads", "zeroclick", [], "waiting_room"),
                # lần tạo đầu bị model_locked
                ("create_session", "moonshotai/kimi-k2.6"),
                ("delete_session",),
                ("create_session", "moonshotai/kimi-k2.6"),
            ],
        )

    async def test_switch_model_deletes_active_upstream_session_before_create(self):
        client = SwitchModelClient()
        manager = SessionManager(
            client,
            Settings(codebuff_token="token", local_api_key=None),
        )

        session = await manager.ensure_session("moonshotai/kimi-k2.6")

        self.assertEqual(session.instance_id, "kimi-instance")
        self.assertEqual(session.model, "moonshotai/kimi-k2.6")
        self.assertEqual(
            client.calls,
            [
                ("get_session", None),
                ("delete_session",),
                ("request_ads", "gravity", [], "waiting_room"),
                ("request_ads", "zeroclick", [], "waiting_room"),
                ("create_session", "moonshotai/kimi-k2.6"),
            ],
        )

    async def test_fresh_cached_session_does_not_probe_upstream(self):
        client = SwitchModelClient()
        manager = SessionManager(
            client,
            Settings(codebuff_token="token", local_api_key=None),
        )
        cached = FreebuffSession(
            instance_id="cached-instance",
            model="deepseek/deepseek-v4-flash",
            remaining_ms=120_000,
        )
        manager._sessions[cached.model] = cached

        session = await manager.ensure_session(cached.model)

        self.assertIs(session, cached)
        self.assertEqual(client.calls, [])

    async def test_session_lease_blocks_model_switch_until_chat_releases(self):
        client = LeaseSwitchModelClient()
        manager = SessionManager(
            client,
            Settings(codebuff_token="token", local_api_key=None),
        )

        first = await manager.acquire_session("deepseek/deepseek-v4-flash")
        started = asyncio.Event()

        async def acquire_second():
            started.set()
            return await manager.acquire_session("moonshotai/kimi-k2.6")

        task = asyncio.create_task(acquire_second())
        await started.wait()
        await asyncio.sleep(0.05)

        self.assertFalse(task.done())
        self.assertNotIn(
            ("delete_session", "deepseek/deepseek-v4-flash"),
            client.calls,
        )

        await first.aclose()
        second = await asyncio.wait_for(task, timeout=1)
        try:
            self.assertEqual(second.session.model, "moonshotai/kimi-k2.6")
            self.assertIn(
                ("delete_session", "deepseek/deepseek-v4-flash"),
                client.calls,
            )
        finally:
            await second.aclose()

    async def test_account_pool_uses_next_free_token_for_concurrent_requests(self):
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
        )

        with patch("providers.freebuff.CodebuffClient", PoolClient):
            pool = CodebuffAccountPool(settings)
            first = await pool.acquire_session("deepseek/deepseek-v4-flash")
            second = await pool.acquire_session("deepseek/deepseek-v4-flash")
            try:
                self.assertEqual(first.client.settings.codebuff_token, "token-a")
                self.assertEqual(second.client.settings.codebuff_token, "token-b")
                self.assertNotEqual(
                    first.session.instance_id,
                    second.session.instance_id,
                )
            finally:
                await second.aclose()
                await first.aclose()
                await pool.aclose()

    async def test_account_pool_skips_rate_limited_account(self):
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
        )

        with patch("providers.freebuff.CodebuffClient", PoolClient):
            pool = CodebuffAccountPool(settings)
            first = await pool.acquire_session("deepseek/deepseek-v4-flash")
            first.mark_rate_limited(60)
            await first.aclose()
            # token-a đang cooldown -> request kế tiếp phải vào token-b
            second = await pool.acquire_session("deepseek/deepseek-v4-flash")
            try:
                self.assertEqual(second.client.settings.codebuff_token, "token-b")
            finally:
                await second.aclose()
                await pool.aclose()

    async def test_account_pool_exposes_redacted_runtime_status(self):
        settings = Settings(codebuff_token="token-a,token-b", local_api_key=None)

        with patch("providers.freebuff.CodebuffClient", PoolClient):
            pool = CodebuffAccountPool(settings)
            first = await pool.acquire_session("deepseek/deepseek-v4-flash")
            self.assertEqual(pool.public_account_statuses()[0]["status"], "busy")
            self.assertNotIn("token", pool.public_account_statuses()[0])
            first.mark_rate_limited(60)
            await first.aclose()
            status = pool.public_account_statuses()[0]
            self.assertEqual(status["status"], "rate_limited")
            self.assertEqual(status["last_error_status"], 429)
            await pool.aclose()

    async def test_account_pool_fails_over_when_session_creation_returns_429(self):
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
            account_cooldown=60,
        )

        with patch("providers.freebuff.CodebuffClient", PoolClient):
            pool = CodebuffAccountPool(settings)
            pool._accounts[0].sessions = RateLimitedSessionManager()
            lease = await pool.acquire_session("deepseek/deepseek-v4-flash")
            try:
                self.assertEqual(lease.client.settings.codebuff_token, "token-b")
                self.assertEqual(pool._default_index, 1)
            finally:
                await lease.aclose()
                # The successfully-failed-over token stays default; it does
                # not bounce back to token-a on the following normal request.
                next_lease = await pool.acquire_session("deepseek/deepseek-v4-flash")
                self.assertEqual(next_lease.client.settings.codebuff_token, "token-b")
                await next_lease.aclose()
                await pool.aclose()

    async def test_account_pool_reports_when_all_accounts_rate_limited(self):
        settings = Settings(
            codebuff_token="token-a,token-b",
            local_api_key=None,
        )

        with patch("providers.freebuff.CodebuffClient", PoolClient):
            pool = CodebuffAccountPool(settings)
            first = await pool.acquire_session("deepseek/deepseek-v4-flash")
            first.mark_rate_limited(60)
            await first.aclose()
            second = await pool.acquire_session("deepseek/deepseek-v4-flash")
            second.mark_rate_limited(60)
            await second.aclose()
            # Cả 2 account đang cooldown: không gửi thêm request vào upstream.
            try:
                with self.assertRaisesRegex(CodebuffError, "All Freebuff tokens"):
                    await pool.acquire_session("deepseek/deepseek-v4-flash")
            finally:
                await pool.aclose()

    async def test_quota_error_uses_host_local_time_and_countdown(self):
        settings = Settings(
            codebuff_token="token-a",
            local_api_key=None,
            timezone="Europe/London",  # Must not affect user-facing time.
        )

        with patch("providers.freebuff.CodebuffClient", PoolClient), patch(
            "providers.freebuff.time.time", return_value=1_715_000_000.0
        ):
            pool = CodebuffAccountPool(settings)
            account = pool._accounts[0]
            account.quota_reset_epoch = 1_715_009_150.0  # 2h 32m 30s later
            account.reset_at = "2024-05-16T10:52:30.000Z"
            try:
                error = pool._all_accounts_rate_limited_error()
                expected_local = datetime.fromtimestamp(1_715_009_150.0).astimezone()
                self.assertIn(expected_local.strftime("GMT%z")[:-2] + ":" + expected_local.strftime("GMT%z")[-2:], str(error))
                self.assertIn("(in 2h 32m)", str(error))
                self.assertNotIn("(UTC)", str(error))
            finally:
                await pool.aclose()


if __name__ == "__main__":
    unittest.main()
