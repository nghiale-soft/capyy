"""The immutable business policy for provider and token failover.

Order is intentionally fixed:
``provider 1 -> every usable token of provider 1 -> provider 2 -> ...``.
Routes must not skip directly from one token to another provider, nor return a
quota error while a lower-priority enabled provider remains available.
"""

from __future__ import annotations

from typing import Any


FREEBUFF_ACCOUNT_FAILURE_STATUSES = frozenset({401, 403, 429})


def is_freebuff_account_failure(error: Exception) -> bool:
    """Whether the current FreeBuff token must be retired for this request."""
    return getattr(error, "status_code", None) in FREEBUFF_ACCOUNT_FAILURE_STATUSES


def should_fallback_to_next_provider(gateway: Any, error: Exception) -> bool:
    """Return true only after the FreeBuff token pool has failed this request.

    The account-pool layer already consumes every usable token before it emits
    its terminal error.  Only then may the provider-priority layer proceed to
    its next enabled generic provider.
    """
    return is_freebuff_account_failure(error) and gateway.has_generic_fallback("freebuff")


def has_next_provider(gateway: Any) -> bool:
    """Whether an enabled lower-priority non-FreeBuff provider exists."""
    return gateway.has_generic_fallback("freebuff")
