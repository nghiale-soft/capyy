from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class Provider(ABC):
    """Interface cho mọi AI provider.

    A new provider only needs to implement this interface and register in the Registry.
    Tuân theo nguyên tắc: không hardcode provider, không phụ thuộc HTTP hay CLI.
    """

    @abstractmethod
    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Gọi chat completions (non-stream)."""
        raise NotImplementedError

    @abstractmethod
    async def stream_chat(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """Gọi chat completions (stream), trả về SSE lines."""
        raise NotImplementedError

    @abstractmethod
    def models(self) -> list[str]:
        """Models supported by this provider."""
        raise NotImplementedError

    @abstractmethod
    def health(self) -> bool:
        """Kiểm tra sức khỏe provider."""
        raise NotImplementedError
