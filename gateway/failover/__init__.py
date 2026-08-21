"""Protected provider/token failover business flow.

Do not encode provider ordering or token exhaustion decisions in API routes.
"""

from .freebuff import FreebuffDispatchFailover
from .policy import (
    FREEBUFF_ACCOUNT_FAILURE_STATUSES,
    has_next_provider,
    should_fallback_to_next_provider,
)

__all__ = [
    "FREEBUFF_ACCOUNT_FAILURE_STATUSES",
    "FreebuffDispatchFailover",
    "has_next_provider",
    "should_fallback_to_next_provider",
]
