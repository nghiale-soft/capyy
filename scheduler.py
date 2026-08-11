from __future__ import annotations

"""Scheduler — lên lịch health check, quota refresh, bảo trì.

Trạng thái: skeleton (Phase 2). Chưa implement đầy đủ.
"""


class Scheduler:
    """Background tasks: health check, quota refresh, circuit breaker reset."""

    def __init__(self) -> None:
        self._tasks: list[Any] = []

    async def start(self) -> None:
        # TODO(Phase 2): start periodic health check / quota refresh
        raise NotImplementedError

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
