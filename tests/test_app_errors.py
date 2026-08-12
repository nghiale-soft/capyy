import json
import unittest

from gateway.deps import error_response
from gateway.services.chat_service import finalize_run
from providers.freebuff import CodebuffError, FreebuffRun
from gateway.core.config import Settings


class FinalizeFailingClient:
    def __init__(self) -> None:
        self.settings = Settings(
            codebuff_token="token",
            local_api_key=None,
            debug=False,
        )

    async def record_run_step(self, *args, **kwargs) -> None:
        raise CodebuffError("network error", 502)

    async def finish_run(self, *args, **kwargs) -> None:
        raise AssertionError("finish_run should not be called")


class AppErrorTests(unittest.TestCase):
    def test_provider_error_creates_metadata_only_contribution(self) -> None:
        captured = []

        class _Contributions:
            def add(self, *args):
                captured.append(args)

        request = type("Request", (), {"app": type("App", (), {"state": type("State", (), {"contributions": _Contributions()})()})()})()
        error_response(CodebuffError("secret upstream detail /Users/me", 429), request)

        self.assertEqual(captured[0][0], "provider")
        self.assertEqual(captured[0][3], {"provider": "freebuff", "status_code": "429"})
        self.assertNotIn("secret", repr(captured))
        self.assertNotIn("/Users", repr(captured))

    def test_codebuff_error_returns_openai_style_json_response(self) -> None:
        response = error_response(CodebuffError("network error", 502))
        body = json.loads(response.body)

        self.assertEqual(response.status_code, 502)
        self.assertEqual(body["error"]["message"], "network error")
        self.assertEqual(body["error"]["type"], "upstream_error")

    def test_finalize_codebuff_error_logs_warning_without_raising(self) -> None:
        client = FinalizeFailingClient()
        run = FreebuffRun(
            run_id="run-1",
            agent_id="agent-1",
            started_at="2026-05-24T00:00:00.000Z",
        )

        with self.assertLogs("gateway.services.chat", level="WARNING") as logs:
            self.asyncio_run(finalize_run(client, run, None))

        self.assertIn("finalize run failed run_id=run-1: network error", logs.output[0])

    def asyncio_run(self, awaitable) -> None:
        import asyncio

        asyncio.run(awaitable)


if __name__ == "__main__":
    unittest.main()
