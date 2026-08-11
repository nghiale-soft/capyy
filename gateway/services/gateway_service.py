from __future__ import annotations

"""Unified gateway orchestration.

Coordinates every provider (freebuff + generic openai-compatible) through the
Registry + Router. Routes only call here and never know the concrete provider.
"""

import logging
import uuid
from typing import Any, AsyncIterator

from providers.openai_compatible import GatewayProviderError
from registry import ProviderRegistry
from router import Router

from ..compat.openai import (
    CompletionAccumulator,
    normalize_chat_messages,
    sanitize_stream_chunk,
)
from ..core.sse import decode_sse_data, encode_sse


logger = logging.getLogger("gateway.services.gateway")


class GatewayService:
    def __init__(
        self,
        registry: ProviderRegistry,
        router: Router,
        *,
        freebuff_chat: Any = None,
    ) -> None:
        """freebuff_chat: callable xử lý chat qua Freebuff (nếu có).

        freebuff_chat(request, body, model_config) -> StreamingResponse | JSONResponse
        """
        self.registry = registry
        self.router = router
        self.freebuff_chat = freebuff_chat

    def _provider_for(self, provider_id: str) -> Any:
        provider = self.registry.get(provider_id)
        if provider is None:
            raise GatewayProviderError(f"provider '{provider_id}' not found", 404)
        return provider

    def resolve(self, model: str | None) -> tuple[str, str | None]:
        """Return (provider_id, real_model)."""
        return self.router.resolve(model)

    def _failover_candidates(
        self,
        provider_id: str,
        body: dict[str, Any],
        real_model: str | None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Ordered list of (provider_id, payload) to try in failover order.

        The resolved provider is tried first; on failure the gateway falls back
        to the next registered provider in priority order. Payloads are rebuilt
        per provider so the model name is rewritten (freebuff models have no
        prefix, generic providers keep their own ids).
        """
        # The model is sent consistently to every candidate: a prefixed request
        # ("ollama/llama3") reaches fallbacks with the same resolved model, and
        # unprefixed models stay as-is.
        candidates = [(provider_id, self._build_payload(body, real_model or body.get("model")))]
        for pid in self.registry.ordered_ids():
            if pid == provider_id:
                continue
            provider = self.registry.get(pid)
            if provider is None:
                continue
            candidates.append((pid, self._build_payload(body, real_model or body.get("model"))))
        return candidates

    def is_freebuff(self, provider_id: str) -> bool:
        provider = self.registry.get(provider_id)
        # Freebuff provider is not an OpenAICompatibleProvider (async .chat)
        return provider is not None and not hasattr(provider, "_chat_url")

    async def chat(
        self,
        provider_id: str,
        body: dict[str, Any],
        *,
        real_model: str | None = None,
        freebuff_handler: Any = None,
    ) -> dict[str, Any]:
        """Non-stream chat through a generic provider, with priority failover."""
        last_error: Exception | None = None
        for pid, payload in self._failover_candidates(provider_id, body, real_model):
            provider = self.registry.get(pid)
            if provider is None or self.is_freebuff(pid):
                continue
            try:
                return await provider.chat(payload)
            except GatewayProviderError as error:
                logger.warning(
                    "provider %s failed, trying next: %s",
                    pid,
                    error,
                )
                last_error = error
        if last_error is not None:
            raise last_error
        raise GatewayProviderError(
            "no provider available for the request", 502
        )

    async def stream_chat(
        self,
        provider_id: str,
        body: dict[str, Any],
        *,
        real_model: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream chat through a generic provider, with priority failover.

        Falls back to the next provider only if the upstream fails before any
        SSE chunk is emitted (a mid-stream error cannot be replayed safely).
        """
        last_error: Exception | None = None
        for pid, payload in self._failover_candidates(provider_id, body, real_model):
            provider = self.registry.get(pid)
            if provider is None or self.is_freebuff(pid):
                continue
            try:
                emitted = False
                async for line in provider.stream_chat(payload):
                    emitted = True
                    yield encode_sse(line) if isinstance(line, str) else line
                return
            except GatewayProviderError as error:
                logger.warning(
                    "provider %s stream failed before chunks, trying next: %s",
                    pid,
                    error,
                )
                last_error = error
                continue
        if last_error is not None:
            raise last_error
        raise GatewayProviderError(
            "no provider available for the request", 502
        )

    def _build_payload(
        self,
        body: dict[str, Any],
        real_model: str | None,
    ) -> dict[str, Any]:
        payload = dict(body)
        messages = normalize_chat_messages(payload.get("messages"))
        payload["messages"] = messages
        if real_model:
            payload["model"] = real_model
        return payload

    async def list_models(self) -> dict[str, Any]:
        """Aggregate model list from all providers.

        Returned model ids are the provider model ids (freebuff model ids have
        no prefix). Clients use these ids to route.
        """
        data: list[dict[str, Any]] = []
        for provider_id, provider in self.registry.all().items():
            try:
                models = await provider.models()
            except Exception:
                logger.exception("failed to list models for provider %s", provider_id)
                models = []
            for model in models:
                data.append(
                    {
                        "id": model,
                        "object": "model",
                        "created": 0,
                        "owned_by": provider_id,
                    }
                )
        return {"object": "list", "data": data}

    async def health(self, provider_id: str) -> bool:
        provider = self._provider_for(provider_id)
        return await provider.health()
