from __future__ import annotations

import logging
from typing import Any

from providers.freebuff import CodebuffClient, CodebuffError
from gateway.compat.models import FreebuffModel, models_response


logger = logging.getLogger("gateway.services.provider")


class ProviderService:
    """Coordinates providers through the Registry + Router.

    This class bridges routes (controllers) and concrete providers. Once the
    Registry/Router are fully implemented, provider selection lives here.
    """

    def __init__(self, client: CodebuffClient) -> None:
        self.client = client

    async def list_models(self) -> dict[str, Any]:
        try:
            upstream_model_ids = await self.client.fetch_free_model_ids()
        except CodebuffError as error:
            logger.warning(
                "model list refresh failed; falling back to static catalog: %s",
                error,
            )
            upstream_model_ids = None
        return models_response(upstream_model_ids)

    def resolve_model(self, requested: str | None) -> FreebuffModel:
        from gateway.compat.models import resolve_model

        return resolve_model(requested)
