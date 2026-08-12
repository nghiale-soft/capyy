import asyncio
import json
import unittest
from unittest import mock

from gateway.core.config import Settings
import gateway.routes.messages as messages_module
from gateway.routes.messages import _stream_generic_anthropic, _stream_tool_loop_anthropic


def _chunk(content: str) -> str:
    data = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek/deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": "stop",
            }
        ],
    }
    return f"data: {json.dumps(data)}"


class _ToolPassClient:
    """Fake upstream that emits a DSML bash tool call in one pass."""

    def __init__(self) -> None:
        self.attempts = 0

    async def chat_events(self, payload):
        self.attempts += 1
        bar = "\uff5c"
        yield _chunk(
            "Chạy analyze nha. "
            f"<{bar}DSML{bar}tool_calls>"
            f"<{bar}DSML{bar}invoke name=\"Bash\">"
            f"<{bar}DSML{bar}parameter name=\"command\" string=\"true\">fvm flutter analyze</{bar}DSML{bar}parameter>"
            f"</{bar}DSML{bar}invoke>"
            f"</{bar}DSML{bar}tool_calls>"
        )
        yield "data: [DONE]"


class _FakeLease:
    closed = False

    async def aclose(self) -> None:
        _FakeLease.closed = True


class NativeToolStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_unclassified_tool_response_creates_safe_contribution(self) -> None:
        class InvalidCompilerClient(_ToolPassClient):
            async def chat_events(self, payload):
                self.attempts += 1
                yield _chunk("not a compiler protocol")
                yield "data: [DONE]"

        captured: list[tuple] = []
        client = InvalidCompilerClient()
        settings = Settings(codebuff_token="token", local_api_key=None)
        async for _ in _stream_tool_loop_anthropic(
            client,
            {"model": "deepseek/deepseek-v4-flash", "messages": [{"role": "user", "content": "secret /Users/me/token.txt"}]},
            body={"model": "deepseek/deepseek-v4-flash", "stream": True, "tools": [
                {"name": "Read", "input_schema": {"type": "object"}}
            ]},
            settings=settings,
            model="deepseek/deepseek-v4-flash",
            requested_model="deepseek/deepseek-v4-flash",
            on_contribution=lambda *args: captured.append(args),
        ):
            pass

        self.assertEqual(len(captured), 1)
        kind, title, summary, metadata = captured[0]
        self.assertEqual((kind, title), ("tool-protocol", "Unclassified upstream tool response"))
        self.assertEqual(
            metadata,
            {
                "error_code": "unclassified_tool_response",
                "declared_tool_count": "1",
                "declared_tools": "Read",
                "compiler_passes": "2",
            },
        )
        self.assertNotIn("/Users", repr(captured))
        self.assertNotIn("secret", repr(captured))

    async def test_compiles_vietnamese_plan_to_native_tool_call_privately(self) -> None:
        class PlanThenToolClient(_ToolPassClient):
            payloads = []

            async def chat_events(self, payload):
                self.attempts += 1
                self.payloads.append(payload)
                if self.attempts == 1:
                    yield _chunk("Bắt đầu bằng việc tìm tài nguyên trong workspace và kiểm tra Figma.")
                else:
                    yield _chunk('{"action":"tool_call","name":"read_file","arguments":{"path":".env.example"}}')
                yield "data: [DONE]"

        client = PlanThenToolClient()
        settings = Settings(codebuff_token="token", local_api_key=None)
        events = []
        async for raw in _stream_tool_loop_anthropic(
            client,
            {"model": "deepseek/deepseek-v4-flash", "messages": [{"role": "user", "content": "analyze"}]},
            body={"model": "deepseek/deepseek-v4-flash", "stream": True, "tools": [
                {"name": "Read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}
            ]},
            settings=settings,
            model="deepseek/deepseek-v4-flash",
            requested_model="deepseek/deepseek-v4-flash",
        ):
            events.append(raw.decode("utf-8"))

        self.assertEqual(client.attempts, 2)
        self.assertEqual(len(client.payloads[1]["messages"]), 2)
        self.assertEqual(client.payloads[1]["response_format"], {"type": "json_object"})
        self.assertEqual(client.payloads[1]["temperature"], 0)
        self.assertIn("tool-call compiler", client.payloads[1]["messages"][0]["content"])
        self.assertIn("Draft to compile", client.payloads[1]["messages"][1]["content"])
        self.assertTrue(any('"name":"Read"' in event for event in events))

    async def test_tool_bearing_turn_uses_protocol_final_classifier(self) -> None:
        class FinalClassifierClient(_ToolPassClient):
            async def chat_events(self, payload):
                self.attempts += 1
                if self.attempts == 2:
                    # The private compiler, not a language regex, releases the
                    # original answer as final.
                    yield _chunk("<<<FINAL>>>")
                else:
                    yield _chunk("Kết quả đã đầy đủ, không cần gọi tool.")
                yield "data: [DONE]"

        client = FinalClassifierClient()
        settings = Settings(codebuff_token="token", local_api_key=None)
        events = []
        async for raw in _stream_tool_loop_anthropic(
            client,
            {"model": "deepseek/deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]},
            body={"model": "deepseek/deepseek-v4-flash", "stream": True, "tools": [
                {"name": "Read", "input_schema": {"type": "object"}}
            ]},
            settings=settings,
            model="deepseek/deepseek-v4-flash",
            requested_model="deepseek/deepseek-v4-flash",
        ):
            events.append(raw.decode("utf-8"))

        self.assertEqual(client.attempts, 2)
        self.assertTrue(any("Kết quả đã đầy đủ" in event for event in events))

    async def test_generic_provider_keeps_native_tool_calls_for_client(self) -> None:
        """A non-FreeBuff provider receives/returns native tools unchanged."""

        class GenericGateway:
            calls = []

            async def stream_chat(self, provider_id, payload, *, real_model=None):
                self.calls.append((provider_id, payload, real_model))
                yield 'data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Read","arguments":"{\\\"path\\\":\\\"a.txt\\\"}"}}]},"finish_reason":"tool_calls"}]}'
                yield "data: [DONE]"

        gateway = GenericGateway()
        events = []
        async for raw in _stream_generic_anthropic(
            gateway,
            "priority-one",
            {
                "model": "gpt",
                "messages": [{"role": "user", "content": "read a.txt"}],
                "tools": [{"type": "function", "function": {"name": "Read"}}],
            },
            real_model="gpt",
            requested_model="claude-test",
        ):
            text = raw.decode("utf-8")
            event = next(
                (line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")),
                "",
            )
            data = next(
                (line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")),
                "{}",
            )
            events.append((event, json.loads(data)))

        self.assertEqual(gateway.calls[0][0], "priority-one")
        self.assertIn("tools", gateway.calls[0][1])
        tool_block = next(
            data["content_block"]
            for event, data in events
            if event == "content_block_start" and data.get("content_block", {}).get("type") == "tool_use"
        )
        self.assertEqual(tool_block["name"], "Read")
        self.assertEqual(tool_block["input"], {})

    async def test_streams_message_start_then_tool_use_with_stop_reason(self) -> None:
        client = _ToolPassClient()
        settings = Settings(codebuff_token="token", local_api_key=None)
        payload = {
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "chạy analyze"}],
        }
        body = {"model": "deepseek/deepseek-v4-flash", "messages": [], "stream": True, "tools": [
            {"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
        ]}

        _FakeLease.closed = False
        events: list[dict] = []
        async for raw in _stream_tool_loop_anthropic(
            client,
            payload,
            body=body,
            settings=settings,
            model="deepseek/deepseek-v4-flash",
            requested_model="deepseek/deepseek-v4-flash",
            account_lease=_FakeLease(),
        ):
            text = raw.decode("utf-8")
            event_line = next(
                (line for line in text.splitlines() if line.startswith("event: ")),
                "",
            )
            data_line = next(
                (line for line in text.splitlines() if line.startswith("data: ")),
                "",
            )
            events.append(
                {
                    "event": event_line.removeprefix("event: "),
                    "data": json.loads(data_line.removeprefix("data: ")),
                }
            )

        event_types = [event["event"] for event in events]
        # message_start must come FIRST so the client never thinks it hung.
        self.assertEqual(event_types[0], "message_start")
        self.assertIn("content_block_start", event_types)
        self.assertIn("content_block_delta", event_types)
        self.assertIn("content_block_stop", event_types)
        self.assertIn("message_delta", event_types)
        self.assertEqual(event_types[-1], "message_stop")

        # The tool_use block must carry the CLIENT tool name + native params.
        tool_block = next(
            event["data"]
            for event in events
            if event["event"] == "content_block_start"
            and event["data"].get("content_block", {}).get("type") == "tool_use"
        )
        self.assertEqual(tool_block["content_block"]["name"], "Bash")
        self.assertEqual(tool_block["content_block"]["input"], {})

        message_delta = next(
            event["data"] for event in events if event["event"] == "message_delta"
        )
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

        # The lease is always closed when the stream ends.
        self.assertTrue(_FakeLease.closed)

    async def test_ping_heartbeat_while_pass_is_slow(self) -> None:
        class SlowClient(_ToolPassClient):
            async def chat_events(self, payload):
                self.attempts += 1
                await asyncio.sleep(0.05)
                yield _chunk("Chờ lâu nè.")
                yield "data: [DONE]"

        client = SlowClient()
        settings = Settings(codebuff_token="token", local_api_key=None)
        payload = {
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
        }
        body = {"model": "deepseek/deepseek-v4-flash", "messages": [], "stream": True, "tools": []}

        _FakeLease.closed = False
        event_types = []
        # Tiny ping interval so the slow fake upstream triggers a heartbeat.
        with mock.patch.object(messages_module, "STREAM_PING_SECONDS", 0.01):
            async for raw in _stream_tool_loop_anthropic(
                client,
                payload,
                body=body,
                settings=settings,
                model="deepseek/deepseek-v4-flash",
                requested_model="deepseek/deepseek-v4-flash",
                account_lease=_FakeLease(),
            ):
                text = raw.decode("utf-8")
                event_line = next(
                    (line for line in text.splitlines() if line.startswith("event: ")),
                    "",
                )
                event_types.append(event_line.removeprefix("event: "))

        self.assertIn("ping", event_types)
        self.assertEqual(event_types[0], "message_start")
        self.assertEqual(event_types[-1], "message_stop")

    async def test_plain_answer_streams_without_tool_use(self) -> None:
        class PlainClient(_ToolPassClient):
            async def chat_events(self, payload):
                self.attempts += 1
                yield _chunk("Không cần tool.")
                yield "data: [DONE]"

        client = PlainClient()
        settings = Settings(codebuff_token="token", local_api_key=None)
        payload = {
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
        }
        body = {"model": "deepseek/deepseek-v4-flash", "messages": [], "stream": True, "tools": []}

        _FakeLease.closed = False
        event_types = []
        async for raw in _stream_tool_loop_anthropic(
            client,
            payload,
            body=body,
            settings=settings,
            model="deepseek/deepseek-v4-flash",
            requested_model="deepseek/deepseek-v4-flash",
            account_lease=_FakeLease(),
        ):
            text = raw.decode("utf-8")
            event_line = next(
                (line for line in text.splitlines() if line.startswith("event: ")),
                "",
            )
            data_line = next(
                (line for line in text.splitlines() if line.startswith("data: ")),
                "",
            )
            event_types.append(event_line.removeprefix("event: "))
            if event_line.startswith("event: content_block_start"):
                block = json.loads(data_line.removeprefix("data: ")).get("content_block")
                self.assertNotEqual(block.get("type"), "tool_use")

        # A plain answer has text blocks but NO tool_use block.
        self.assertIn("content_block_start", event_types)
        self.assertEqual(event_types[-1], "message_stop")
        self.assertIn("message_delta", event_types)


if __name__ == "__main__":
    unittest.main()
