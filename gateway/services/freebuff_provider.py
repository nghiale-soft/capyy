from __future__ import annotations

"""Adapter wrapping the Freebuff provider for Registry registration.

Freebuff has its own logic (session/ads/agent-runs), so OpenAICompatibleProvider
cannot be reused. This adapter exposes the Provider interface but routes
chat/stream through SessionService (CodebuffClient).
"""

import logging
from typing import Any

from providers.base import Provider
from providers.freebuff import CodebuffError

from ..compat.models import models_response
from ..services.provider_crud import ProviderConfig


logger = logging.getLogger("gateway.services.freebuff_provider")


class FreebuffProviderAdapter(Provider):
    def __init__(self, cfg: ProviderConfig, session_service: Any) -> None:
        self.id = cfg.id
        self.cfg = cfg
        self.session_service = session_service

    @property
    def client(self) -> Any:
        # Grab the client from the live pool: if tokens are updated via the
        # dashboard, the adapter uses the new client without re-registering.
        return self.session_service.default_client

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise CodebuffError("use /v1/chat/completions router for freebuff", 400)

    async def stream_chat(self, payload: dict[str, Any]) -> Any:
        raise CodebuffError("use /v1/chat/completions router for freebuff", 400)

    async def models(self) -> list[str]:
        try:
            upstream = await self.client.fetch_free_model_ids()
        except CodebuffError as error:
            logger.warning("freebuff model refresh failed: %s", error)
            upstream = None
        response = models_response(upstream)
        # Freebuff models trong catalog không có prefix.
        # Khi gọi /v1/models, trả về danh sách model id để client có thể route.
        return [item["id"] for item in response["data"]]

    async def health(self) -> bool:
        try:
            await self.client.health()
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        pass
