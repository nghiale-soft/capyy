from __future__ import annotations

"""Judge — đánh giá chất lượng trả lời để tối ưu chi phí/chuyển hướng.

Trạng thái: skeleton (Phase 3). Chưa implement đầy đủ.
"""

from typing import Any


class Judge:
    """Dùng model nhỏ/AI để đánh giá output, học chọn provider tốt hơn."""

    def __init__(self) -> None:
        self._scores: dict[str, list[float]] = {}

    def score(self, provider_id: str, good: bool) -> None:
        # TODO(Phase 3): accumulate scores, feed learning router
        self._scores.setdefault(provider_id, []).append(1.0 if good else 0.0)
