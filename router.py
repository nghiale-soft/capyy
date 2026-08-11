from __future__ import annotations

"""Router — chọn provider phù hợp cho mỗi request.

Hỗ trợ 3 chế độ:
  - strict: bắt buộc model có prefix provider (`claude/...`, `blackbox/...`)
  - fallback: model không có prefix sẽ về provider default
  - auto: (mở rộng Phase 2) dựa trên rule/health/quota

Cách route theo model:
  - `claude/claude-3-5-sonnet`     -> provider_id="claude", model="claude-3-5-sonnet"
  - `blackbox/blackbox-gpt-4o`     -> provider_id="blackbox", model="blackbox-gpt-4o"
  - `gpt-4o` (không prefix)        -> provider default (nếu có)
"""

import logging
from typing import Any

from registry import ProviderRegistry


logger = logging.getLogger("gateway.router")


class Router:
    """Chọn provider dựa trên model prefix."""

    def __init__(self, registry: ProviderRegistry, mode: str = "fallback") -> None:
        self.registry = registry
        self.mode = mode

    def resolve(self, model: str | None) -> tuple[str, str | None]:
        """Trả về (provider_id, real_model).

        - Model có prefix provider -> (provider_id, model_after_prefix)
        - Model không prefix -> (default_provider_id, model)
        - model None -> (default_provider_id, None)
        """
        if not model:
            default = self.registry.get_default()
            if default is None:
                raise ValueError("no provider available")
            return self._provider_id_for(default), None

        if "/" in model:
            prefix, _, rest = model.partition("/")
            if prefix in self.registry:
                return prefix, rest

        if self.mode == "strict":
            raise ValueError(
                f"model '{model}' has no provider prefix and strict mode "
                f"requires one (e.g. 'claude/...', 'blackbox/...')"
            )

        # fallback: về provider default
        default = self.registry.get_default()
        if default is None:
            raise ValueError(f"no default provider for model '{model}'")
        return self._provider_id_for(default), model

    def _provider_id_for(self, provider: Any) -> str:
        provider_id = getattr(provider, "id", None)
        if provider_id:
            return str(provider_id)
        # fallback: tìm theo identity
        for pid, p in self.registry.all().items():
            if p is provider:
                return pid
        raise ValueError("cannot determine provider id")

    def route(self, model: str) -> str:
        """Trả về provider_id cho model (compat wrapper)."""
        provider_id, _ = self.resolve(model)
        return provider_id
