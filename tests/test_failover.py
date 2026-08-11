import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.services.gateway_service import GatewayService
from providers.openai_compatible import GatewayProviderError
from registry import ProviderRegistry
from router import Router


def _run(coro):
    return asyncio.run(coro)


def _collect(agen):
    """Consume an async generator and return its items as a list."""
    return asyncio.run(_collect_async(agen))


async def _collect_async(agen):
    return [item async for item in agen]


class _Provider:
    # Generic OpenAI-compatible providers expose _chat_url; freebuff does not.
    _chat_url = "/chat/completions"

    def __init__(self, pid, *, fail=False, empty_stream=False):
        self.id = pid
        self._fail = fail
        self._empty_stream = empty_stream

    async def chat(self, payload):
        if self._fail:
            raise GatewayProviderError(f"{self.id} boom", 502)
        return {"choices": [{"message": {"content": f"from-{self.id}"}}]}

    async def stream_chat(self, payload):
        if self._fail:
            raise GatewayProviderError(f"{self.id} boom", 502)
        if self._empty_stream:
            return
        yield 'data: {"x": 1}'
        yield "data: [DONE]"

    async def models(self):
        return [self.id + "-model"]


class _FreebuffProvider:
    """Freebuff providers are async-native; used to prove they are skipped."""

    def __init__(self, pid):
        self.id = pid

    async def models(self):
        return [self.id + "-model"]


class StreamErrorSseTests(unittest.TestCase):
    def test_safe_stream_emits_valid_sse_error(self):
        """Failover exhaustion inside a stream must emit a `data:` SSE frame."""
        import json as _json
        from gateway.core.sse import decode_sse_data
        from gateway.routes.chat import _safe_stream

        async def _boom():
            raise GatewayProviderError("all down", 502)
            yield  # pragma: no cover

        async def _collect():
            out = []
            async for chunk in _safe_stream(_boom()):
                out.append(chunk)
            return out

        chunks = asyncio.run(_collect())
        self.assertEqual(len(chunks), 1)
        line = chunks[0].decode()
        self.assertTrue(line.startswith("data: "), line)
        self.assertIn("provider_unavailable", line)
        parsed = decode_sse_data(line)
        self.assertEqual(parsed["error"]["code"], "provider_unavailable")


class GatewayFailoverTests(unittest.TestCase):
    def _make_gateway(self, providers, *, default="a"):
        registry = ProviderRegistry()
        for pid, prov in providers:
            registry.register(pid, prov, default=(pid == default))
        router = Router(registry, mode="fallback")
        return GatewayService(registry, router)

    def test_chat_falls_back_to_next_provider(self):
        gw = self._make_gateway(
            [("a", _Provider("a", fail=True)), ("b", _Provider("b"))]
        )
        result = _run(gw.chat("a", {"model": "x", "messages": []}))
        self.assertEqual(result["choices"][0]["message"]["content"], "from-b")

    def test_chat_uses_first_provider_when_healthy(self):
        gw = self._make_gateway([("a", _Provider("a")), ("b", _Provider("b"))])
        result = _run(gw.chat("a", {"model": "x", "messages": []}))
        self.assertEqual(result["choices"][0]["message"]["content"], "from-a")

    def test_chat_raises_when_all_providers_fail(self):
        gw = self._make_gateway(
            [("a", _Provider("a", fail=True)), ("b", _Provider("b", fail=True))]
        )
        with self.assertRaises(GatewayProviderError):
            _run(gw.chat("a", {"model": "x", "messages": []}))

    def test_stream_falls_back_before_first_chunk(self):
        gw = self._make_gateway(
            [("a", _Provider("a", fail=True)), ("b", _Provider("b"))]
        )
        chunks = _collect(gw.stream_chat("a", {"model": "x", "messages": []}))
        joined = b"".join(chunks).decode()
        self.assertIn("[DONE]", joined)

    def test_freebuff_providers_are_not_used_as_failover_targets(self):
        # Freebuff has its own session logic; it must not receive generic payloads.
        gw = self._make_gateway(
            [("freebuff", _FreebuffProvider("freebuff")), ("a", _Provider("a"))],
            default="freebuff",
        )
        result = _run(gw.chat("freebuff", {"model": "x", "messages": []}))
        self.assertEqual(result["choices"][0]["message"]["content"], "from-a")

    def test_failover_preserves_model_for_generic_target(self):
        calls = []

        class _Recorder(_Provider):
            def __init__(self, pid, **kw):
                super().__init__(pid, **kw)

            async def chat(self, payload):
                calls.append((self.id, payload.get("model")))
                return await super().chat(payload)

        gw = self._make_gateway([("a", _Recorder("a", fail=True)), ("b", _Recorder("b"))])
        _run(gw.chat("a", {"model": "my-model", "messages": []}))
        self.assertEqual(calls, [("a", "my-model"), ("b", "my-model")])

    def test_failover_passes_resolved_model_to_fallback(self):
        """Prefixed requests reach fallback providers with the resolved model,
        not the raw "provider/model" prefix string."""
        calls = []

        class _Recorder(_Provider):
            async def chat(self, payload):
                calls.append((self.id, payload.get("model")))
                return await super().chat(payload)

        gw = self._make_gateway([("a", _Recorder("a", fail=True)), ("b", _Recorder("b"))])
        _run(gw.chat("a", {"model": "a/llama3", "messages": []}, real_model="llama3"))
        self.assertEqual(calls, [("a", "llama3"), ("b", "llama3")])


if __name__ == "__main__":
    unittest.main()
