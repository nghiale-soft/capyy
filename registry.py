from __future__ import annotations

"""Registry — đăng ký và quản lý các provider.

Mỗi provider được đăng ký với một id duy nhất (vd: "freebuff", "claude",
"blackbox", "openrouter", ...). Registry quản lý vòng đời provider:
  - openai-compatible: tạo instance OpenAICompatibleProvider từ config
  - freebuff: instance FreebuffClient + SessionService
"""

import logging
from typing import Any

from providers.base import Provider
from providers.openai_compatible import OpenAICompatibleProvider


logger = logging.getLogger("gateway.registry")


class ProviderRegistry:
    """Registry giữ map provider_id -> Provider instance."""

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}
        self._default: str | None = None

    def register(self, provider_id: str, provider: Any, *, default: bool = False) -> None:
        self._providers[provider_id] = provider
        # ``default`` is kept only for backwards-compatible direct callers.
        # Configured providers are registered in priority order below, so the
        # first enabled provider is always the routing default.
        if default or self._default is None:
            self._default = provider_id
        logger.info(
            "registered provider id=%s default=%s",
            provider_id,
            default,
        )

    def get(self, provider_id: str) -> Any:
        return self._providers.get(provider_id)

    def get_default(self) -> Any:
        if self._default:
            return self._providers.get(self._default)
        for provider in self._providers.values():
            return provider
        raise KeyError("no provider registered")

    def all(self) -> dict[str, Any]:
        return dict(self._providers)

    def ordered_ids(self) -> list[str]:
        """Provider ids in failover priority order (registration order)."""
        return list(self._providers.keys())

    def __len__(self) -> int:
        return len(self._providers)

    def __contains__(self, provider_id: str) -> bool:
        return provider_id in self._providers

    def build_from_config(
        self,
        configs: list[Any],
        *,
        freebuff_factory: Any = None,
    ) -> None:
        """Build registry từ danh sách ProviderConfig.

        freebuff_factory: callable nhận ProviderConfig -> provider instance
        (ví dụ tạo CodebuffClient/SessionService). Nếu None, bỏ qua freebuff.
        Providers are registered in failover priority order (lowest priority
        value first) so Router/GatewayService can fail over down the list.
        """
        for cfg in sorted(
            configs,
            key=lambda c: getattr(c, "priority", 0),
        ):
            if not getattr(cfg, "enabled", True):
                logger.info("skip disabled provider '%s'", cfg.id)
                continue
            provider_type = getattr(cfg, "type", "openai-compatible")
            source = getattr(cfg, "source", "url")
            command = getattr(cfg, "command", "")

            # Command-based providers other than freebuff are saved as config
            # but have no runner yet (local CLI executors not implemented).
            if source == "command" and command and command != "freebuff":
                logger.warning(
                    "skip command provider '%s' (command=%s): CLI runner not implemented yet",
                    cfg.id,
                    command,
                )
                continue

            if provider_type == "freebuff" or command == "freebuff":
                if freebuff_factory is not None:
                    provider = freebuff_factory(cfg)
                    self.register(cfg.id, provider, default=not self._providers)
                else:
                    logger.warning(
                        "skip freebuff provider '%s': no factory",
                        cfg.id,
                    )
                continue

            provider = OpenAICompatibleProvider(
                cfg.id,
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                models=cfg.models,
            )
            self.register(cfg.id, provider, default=not self._providers)

    def reload(
        self,
        configs: list[Any],
        *,
        freebuff_factory: Any = None,
    ) -> None:
        """Rebuild the registry from configs in place (used after CRUD changes).

        Keeps the same registry object identity so Router/GatewayService hold
        a valid reference; the provider map is swapped atomically.
        """
        fresh = ProviderRegistry()
        fresh.build_from_config(configs, freebuff_factory=freebuff_factory)
        self._providers = fresh._providers
        self._default = fresh._default
        logger.info(
            "registry reloaded providers=%s default=%s",
            list(self._providers.keys()),
            self._default,
        )

    async def aclose(self) -> None:
        for provider in self._providers.values():
            close = getattr(provider, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    logger.exception("failed to close provider")
