"""Protected entry point for native client-tool protocol handling.

The implementation remains in ``gateway.services.toolkit`` during the
compatibility migration; new callers must import from this boundary.
"""

from gateway.services.toolkit import (
    adapt_client_tool_call,
    client_tool_call,
    coerce_client_tool_call_arguments,
    declared_client_tool_names,
    detect_tool_markers,
    parse_tool_call,
    validate_client_tool_call,
)

__all__ = [
    "adapt_client_tool_call",
    "client_tool_call",
    "coerce_client_tool_call_arguments",
    "declared_client_tool_names",
    "detect_tool_markers",
    "parse_tool_call",
    "validate_client_tool_call",
]
