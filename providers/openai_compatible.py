from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

from gateway.core.sse import decode_sse_data, encode_sse
from .base import Provider


logger = logging.getLogger("providers.openai_compatible")


class OpenAICompatibleProvider(Provider):
    """Provider generic cho mọi API OpenAI-compatible.

    Hỗ trợ các provider như Claude (qua Anthropic-compatible gateway),
    Blackbox, OpenRouter, Ollama, LM Studio... chỉ cần:
      - base_url: endpoint gốc (vd "https://openrouter.ai/api/v1")
      - api_key: Bearer token (có thể rỗng với Ollama/LM Studio local)
      - models: danh sách model id provider hỗ trợ
    """

    def __init__(
        self,
        provider_id: str,
        *,
        base_url: str,
        api_key: str | None = None,
        models: list[str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.id = provider_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # Stored as _models so it does not shadow the ``models()`` method
        # (gateway_service.list_models calls ``await provider.models()``).
        self._models = models or []
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, read=None),
            follow_redirects=True,
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _chat_url(self) -> str:
        # Cho phép base_url trỏ thẳng tới /chat/completions hoặc gốc
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _models_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url.rsplit("/chat/completions", 1)[0] + "/models"
        return f"{self.base_url}/models"

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Gọi chat completions non-stream."""
        url = self._chat_url()
        body = dict(payload)
        body["stream"] = False
        try:
            response = await self._client.post(
                url,
                json=body,
                headers=self._headers(json_body=True),
            )
        except httpx.RequestError as error:
            raise GatewayProviderError(
                f"provider {self.id} request failed: {error}"
            ) from error

        if response.status_code >= 400:
            raise GatewayProviderError(
                f"provider {self.id} error {response.status_code}: "
                f"{response.text[:500]}"
            )
        return response.json()

    async def stream_chat(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Gọi chat completions stream, trả về từng dòng SSE."""
        url = self._chat_url()
        body = dict(payload)
        body["stream"] = True
        try:
            async with self._client.stream(
                "POST",
                url,
                json=body,
                headers=self._headers(json_body=True),
            ) as response:
                if response.status_code >= 400:
                    text = (await response.aread()).decode(
                        "utf-8", errors="replace"
                    )
                    raise GatewayProviderError(
                        f"provider {self.id} error {response.status_code}: "
                        f"{text[:500]}"
                    )
                async for line in response.aiter_lines():
                    if line:
                        yield line
        except httpx.RequestError as error:
            raise GatewayProviderError(
                f"provider {self.id} request failed: {error}"
            ) from error

    async def models(self) -> list[str]:
        """Return the model list from config (and try fetching from the API if possible)."""
        if self._models:
            return list(self._models)
        # Fallback: fetch từ /models endpoint nếu có
        try:
            response = await self._client.get(
                self._models_url(),
                headers=self._headers(),
            )
            if response.status_code >= 400:
                return []
            data = response.json()
            ids = []
            for item in data.get("data") or []:
                if isinstance(item, dict) and item.get("id"):
                    ids.append(str(item["id"]))
            return ids
        except Exception:
            return []

    async def health(self) -> bool:
        """Kiểm tra sức khỏe provider bằng GET /models (hoặc base_url)."""
        try:
            response = await self._client.get(
                self._models_url(),
                headers=self._headers(),
                timeout=10.0,
            )
            return response.status_code < 500
        except Exception:
            return False


class GatewayProviderError(RuntimeError):
    """Lỗi provider từ generic adapter."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code
