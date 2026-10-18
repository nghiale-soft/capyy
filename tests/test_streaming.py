import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gateway.services.chat_service import (
    _start_child_chat_run_chain,
    chat_events_with_recovery,
    collect_completion,
    prepare_freebuff_dispatch,
    start_freebuff_run_chain,
    stream_openai_chunks,
)
from providers.freebuff import CodebuffError, FreebuffRun
from gateway.core.config import Settings
from gateway.compat.models import resolve_model


class FakeClient:
    def __init__(self) -> None:
        self.recorded = False
        self.finished = False
        self.calls = []

    async def chat_events(self, payload):
        yield (
            'data: {"id":"chunk-1","object":"chat.completion.chunk",'
            '"created":1,"model":"deepseek/deepseek-v4-flash",'
            '"choices":[{"index":0,"delta":{"content":null,'
            '"reasoning_content":"hello"},"finish_reason":null}]}'
        )
        yield "data: [DONE]"

    async def record_run_step(self, *args, **kwargs) -> None:
        self.recorded = True
        self.calls.append(("step", args, kwargs))
        await asyncio.sleep(0)

    async def finish_run(self, *args, **kwargs) -> None:
        self.finished = True
        self.calls.append(("finish", args, kwargs))
        await asyncio.sleep(0)

    async def start_run(self, agent_id, ancestor_run_ids=None):
        run_id = f"run-{len([call for call in self.calls if call[0] == 'start']) + 1}"
        self.calls.append(("start", agent_id, ancestor_run_ids or [], run_id))
        await asyncio.sleep(0)
        return run_id


class FailingStreamClient(FakeClient):
    async def chat_events(self, payload):
        raise CodebuffError("Codebuff chat failed: 403 hierarchy", 502)
        yield


class SlowCompletionClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.settings = SimpleNamespace(request_timeout=0.01, debug=False)

    async def chat_events(self, payload):
        await asyncio.sleep(1)
        yield "data: [DONE]"


class WaitingRoomClient(FakeClient):
    """Lần chat đầu trả 428 waiting_room_required, lần sau thành công."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    async def chat_events(self, payload):
        self.attempts += 1
        if self.attempts == 1:
            raise CodebuffError(
                'Codebuff chat failed: 428 {"error":"waiting_room_required",'
                '"message":"No active free session. Call POST /api/v1/freebuff/session first."}',
                428,
            )
            yield
        yield (
            'data: {"id":"chunk-1","object":"chat.completion.chunk",'
            '"created":1,"model":"deepseek/deepseek-v4-flash",'
            '"choices":[{"index":0,"delta":{"content":null,'
            '"reasoning_content":"hello"},"finish_reason":null}]}'
        )
        yield "data: [DONE]"


class AlwaysWaitingRoomClient(FakeClient):
    """Luôn trả 428 — recovery chỉ nên thử 1 lần rồi dừng."""

    async def chat_events(self, payload):
        raise CodebuffError(
            'Codebuff chat failed: 428 {"error":"waiting_room_required",'
            '"message":"No active free session."}',
            428,
        )
        yield


class PreparationClient:
    def __init__(self, *, fail=False) -> None:
        self.fail = fail

    async def request_ad_chain(self, messages=None):
        if self.fail:
            raise CodebuffError("upstream unavailable", 502)

    async def validate_agents(self):
        return None


class PreparationLease:
    def __init__(self, index, client) -> None:
        self._account_index = index
        self.client = client
        self.session = object()
        self.marked_error = None
        self.closed = False

    def mark_rate_limited(self, duration, *, error=None):
        self.marked_error = error

    async def aclose(self):
        self.closed = True


class PreparationAccounts:
    account_count = 2

    def __init__(self):
        self.leases = [
            PreparationLease(0, PreparationClient(fail=True)),
            PreparationLease(1, PreparationClient()),
        ]
        self.acquired = 0

    async def acquire_session(self, model, messages=None):
        lease = self.leases[self.acquired]
        self.acquired += 1
        return lease


class RateLimitedStreamClient(FakeClient):
    async def chat_events(self, payload):
        raise CodebuffError("Codebuff chat failed: 429 quota", 429)
        yield


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_preparation_fails_over_to_another_token(self) -> None:
        accounts = PreparationAccounts()
        model = SimpleNamespace(session_id="model", upstream_id="upstream")
        settings = SimpleNamespace(client_id="client", max_tokens=100, account_cooldown=60)

        async def _start_run(_client, _model):
            return FreebuffRun("run", "agent", "started")

        with patch("gateway.services.chat_service.start_freebuff_run_chain", _start_run), patch(
            "gateway.services.chat_service.build_payload",
            return_value={"payload": True},
        ):
            lease, client, run, payload = await prepare_freebuff_dispatch(
                accounts,
                model,
                {"model": "m"},
                [],
                settings,
            )

        self.assertEqual(lease._account_index, 1)
        self.assertIs(client, lease.client)
        self.assertEqual(run.run_id, "run")
        self.assertEqual(payload, {"payload": True})
        self.assertTrue(accounts.leases[0].closed)
        self.assertEqual(accounts.leases[0].marked_error.status_code, 502)
    async def test_stream_forwards_content_before_finalize(self) -> None:
        client = FakeClient()
        settings = Settings(codebuff_token="token", local_api_key=None, debug=False)

        chunks = []
        run = FreebuffRun(
            run_id="run-1",
            agent_id="base2-free-deepseek-flash",
            started_at="2026-05-23T00:00:00.000Z",
        )
        async for chunk in stream_openai_chunks(client, {}, run):
            chunks.append(chunk.decode("utf-8"))

        first_payload = json.loads(chunks[0].removeprefix("data: ").strip())

        delta = first_payload["choices"][0]["delta"]
        self.assertNotIn("content", delta)
        self.assertEqual(delta["reasoning_content"], "hello")
        self.assertEqual(chunks[1], "data: [DONE]\n\n")

        await asyncio.sleep(0.05)
        self.assertTrue(client.recorded)
        self.assertTrue(client.finished)

    async def test_run_chain_matches_freebuff_parent_child_shape(self) -> None:
        client = FakeClient()

        run = await start_freebuff_run_chain(client, "base2-free-kimi")

        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(run.child_run_id, "run-2")
        self.assertEqual(client.calls[0], ("start", "base2-free-kimi", [], "run-1"))
        self.assertEqual(
            client.calls[1],
            ("start", "context-pruner", ["run-1"], "run-2"),
        )
        self.assertEqual(client.calls[2][0], "step")
        self.assertEqual(client.calls[2][1], ("run-2",))
        self.assertEqual(client.calls[2][2]["step_number"], 1)
        self.assertEqual(client.calls[2][2]["child_run_ids"], [])
        self.assertEqual(client.calls[2][2]["message_id"], None)
        self.assertEqual(client.calls[3], ("finish", ("run-2",), {"total_steps": 2}))
        self.assertEqual(
            client.calls[4],
            (
                "step",
                ("run-1",),
                {
                    "step_number": 1,
                    "child_run_ids": ["run-2"],
                    "message_id": None,
                    "start_time": run.started_at,
                },
            ),
        )

    async def test_gemini_thinker_run_chain_uses_child_as_payload_run(self) -> None:
        client = FakeClient()

        run = await start_freebuff_run_chain(
            client,
            resolve_model("google/gemini-3.1-pro-preview"),
        )

        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(run.chat_run_id, "run-2")
        self.assertEqual(run.payload_run_id, "run-2")
        self.assertEqual(client.calls[0], ("start", "base2-free-kimi", [], "run-1"))
        self.assertEqual(
            client.calls[1],
            ("start", "thinker-with-files-gemini", ["run-1"], "run-2"),
        )

    async def test_gemini_flash_lite_run_chain_uses_session_root_parent(self) -> None:
        client = FakeClient()

        run = await start_freebuff_run_chain(
            client,
            resolve_model("google/gemini-2.5-flash-lite"),
        )

        self.assertEqual(run.run_id, "run-1")
        self.assertEqual(run.chat_run_id, "run-2")
        self.assertEqual(run.payload_run_id, "run-2")
        self.assertEqual(
            client.calls[0],
            ("start", "base2-free-mimo", [], "run-1"),
        )
        self.assertEqual(
            client.calls[1],
            ("start", "file-picker", ["run-1"], "run-2"),
        )

    async def test_stream_openai_chunks_calls_rate_limit_hook_and_closes_lease(self) -> None:
        class RateLimitedClient(FakeClient):
            async def chat_events(self, payload):
                raise CodebuffError(
                    "Codebuff chat failed: 429 rate limited",
                    429,
                )
                yield

        client = RateLimitedClient()
        lease_closed = {"closed": False}
        rate_limited = {"called": False}

        class FakeLease:
            async def aclose(self) -> None:
                lease_closed["closed"] = True

        run = FreebuffRun(
            run_id="run-1",
            agent_id="base2-free-deepseek-flash",
            started_at="2026-05-23T00:00:00.000Z",
        )
        with self.assertLogs("gateway.services.chat", level="WARNING"):
            async for _ in stream_openai_chunks(
                client,
                {},
                run,
                account_lease=FakeLease(),
                on_rate_limited=lambda error: rate_limited.__setitem__("called", True),
            ):
                pass

        self.assertTrue(rate_limited["called"])
        self.assertTrue(lease_closed["closed"])

    async def test_streaming_codebuff_error_is_returned_as_sse_error(self) -> None:
        client = FailingStreamClient()
        settings = Settings(codebuff_token="token", local_api_key=None, debug=False)

        chunks = []
        run = FreebuffRun(
            run_id="run-1",
            agent_id="base2-free-deepseek-flash",
            started_at="2026-05-23T00:00:00.000Z",
        )
        with self.assertLogs("gateway.services.chat", level="WARNING"):
            async for chunk in stream_openai_chunks(client, {}, run):
                chunks.append(chunk.decode("utf-8"))

        error_payload = json.loads(chunks[0].removeprefix("data: ").strip())
        self.assertEqual(error_payload["error"]["code"], "codebuff_error")
        self.assertEqual(chunks[1], "data: [DONE]\n\n")

    async def test_stream_recovers_once_from_waiting_room_required(self) -> None:
        client = WaitingRoomClient()
        recovered = {"called": 0, "payload": None}

        async def _recover(_payload: dict) -> dict:
            recovered["called"] += 1
            recovered["payload"] = {
                "codebuff_metadata": {"freebuff_instance_id": "instance-fresh"}
            }
            return recovered["payload"]

        run = FreebuffRun(
            run_id="run-1",
            agent_id="base2-free-deepseek-flash",
            started_at="2026-05-23T00:00:00.000Z",
        )
        chunks = []
        with self.assertLogs("gateway.services.chat", level="WARNING"):
            async for chunk in stream_openai_chunks(
                client,
                {"codebuff_metadata": {"freebuff_instance_id": "instance-stale"}},
                run,
                recover=_recover,
            ):
                chunks.append(chunk.decode("utf-8"))

        self.assertEqual(client.attempts, 2)
        self.assertEqual(recovered["called"], 1)
        self.assertEqual(
            recovered["payload"]["codebuff_metadata"]["freebuff_instance_id"],
            "instance-fresh",
        )
        first_payload = json.loads(chunks[0].removeprefix("data: ").strip())
        self.assertEqual(first_payload["choices"][0]["delta"]["reasoning_content"], "hello")
        self.assertEqual(chunks[1], "data: [DONE]\n\n")

    async def test_stream_recovers_at_most_once_from_waiting_room_required(self) -> None:
        client = AlwaysWaitingRoomClient()
        recovered = {"called": 0}

        async def _recover(_payload: dict) -> dict:
            recovered["called"] += 1
            return {"codebuff_metadata": {"freebuff_instance_id": "instance-fresh"}}

        run = FreebuffRun(
            run_id="run-1",
            agent_id="base2-free-deepseek-flash",
            started_at="2026-05-23T00:00:00.000Z",
        )
        chunks = []
        with self.assertLogs("gateway.services.chat", level="WARNING"):
            async for chunk in stream_openai_chunks(
                client,
                {},
                run,
                recover=_recover,
            ):
                chunks.append(chunk.decode("utf-8"))

        self.assertEqual(recovered["called"], 1)
        error_payload = json.loads(chunks[0].removeprefix("data: ").strip())
        self.assertEqual(error_payload["error"]["code"], "codebuff_error")
        self.assertEqual(chunks[1], "data: [DONE]\n\n")

    async def test_chat_events_with_recovery_reraises_non_428_errors(self) -> None:
        client = FailingStreamClient()

        async def _recover(_payload: dict) -> dict:
            raise AssertionError("recover should not be called for non-428 errors")

        with self.assertRaises(CodebuffError):
            async for _ in chat_events_with_recovery(
                client,
                {},
                recover=_recover,
            ):
                pass

    async def test_chat_events_fails_over_before_any_upstream_output(self) -> None:
        limited = RateLimitedStreamClient()
        healthy = FakeClient()
        retried = {"called": 0, "payload": None}

        async def _failover(error: CodebuffError, payload: dict) -> tuple[FakeClient, dict]:
            self.assertEqual(error.status_code, 429)
            retried["called"] += 1
            retried["payload"] = payload
            return healthy, {"codebuff_metadata": {"freebuff_instance_id": "token-1"}}

        lines = []
        async for line in chat_events_with_recovery(
            limited,
            {"codebuff_metadata": {"freebuff_instance_id": "token-2"}},
            failover=_failover,
        ):
            lines.append(line)

        self.assertEqual(retried["called"], 1)
        self.assertEqual(retried["payload"]["codebuff_metadata"]["freebuff_instance_id"], "token-2")
        self.assertEqual(lines[-1], "data: [DONE]")

    async def test_chat_events_fails_over_on_upstream_server_error(self) -> None:
        unavailable = FailingStreamClient()
        healthy = FakeClient()
        attempts = {"count": 0}

        async def _failover(error: CodebuffError, _payload: dict) -> tuple[FakeClient, dict]:
            self.assertEqual(error.status_code, 502)
            attempts["count"] += 1
            return healthy, {}

        lines = [
            line
            async for line in chat_events_with_recovery(
                unavailable,
                {},
                failover=_failover,
            )
        ]

        self.assertEqual(attempts["count"], 1)
        self.assertEqual(lines[-1], "data: [DONE]")

    async def test_non_stream_completion_times_out_and_returns_gateway_error(self) -> None:
        client = SlowCompletionClient()
        run = FreebuffRun(
            run_id="run-timeout",
            agent_id="base2-free-deepseek-flash",
            started_at="2026-05-23T00:00:00.000Z",
        )
        with patch("gateway.services.chat_service.schedule_finalize_run"):
            with self.assertRaises(CodebuffError) as context:
                await collect_completion(client, {}, run, "deepseek/deepseek-v4-flash")

        self.assertEqual(context.exception.status_code, 504)
        self.assertIn("timed out", str(context.exception))


if __name__ == "__main__":
    unittest.main()
