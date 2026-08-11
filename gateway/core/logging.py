from __future__ import annotations

import json
import logging
import sys
from typing import Any

from fastapi import Request

from .config import Settings

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[35m",
}

CURL_BODY_LIMIT = 20000


def log_curl(request: Request, body: Any, *, logger: logging.Logger) -> None:
    """Log the incoming API request as a curl command for easy debugging.

    The output is a single-line curl command that can be copy-pasted into a
    terminal to reproduce the exact request. Sensitive headers (authorization,
    cookie, x-api-key) are masked. The body is capped at ``CURL_BODY_LIMIT``
    chars and truncated with a note if longer.

    Logged at DEBUG level because request bodies can contain private source
    code and long agent transcripts.
    """
    url = str(request.url)
    method = request.method

    headers_to_log: list[tuple[str, str]] = []
    for key, value in request.headers.items():
        if key.lower() in {"authorization", "cookie", "set-cookie", "x-api-key"}:
            headers_to_log.append((key, "<redacted>"))
        else:
            headers_to_log.append((key, value))

    raw_body = json.dumps(body, ensure_ascii=False, default=str)
    if len(raw_body) > CURL_BODY_LIMIT:
        body_log = raw_body[:CURL_BODY_LIMIT] + "... <TRUNCATED>"
        body_note = f" (body truncated from {len(raw_body)} to {CURL_BODY_LIMIT} chars)"
    else:
        body_log = raw_body
        body_note = ""

    curl_line = f"curl -s -X {method} '{url}'"
    for key, value in headers_to_log:
        curl_line += f" -H '{key}: {value}'"
    escaped_body = body_log.replace("'", "'\\''")
    curl_line += f" --data-raw '{escaped_body}'"

    logger.debug("CURL: %s%s", curl_line, body_note)


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = COLORS.get(record.levelno)
        if not color:
            return message
        return f"{color}{message}{RESET}"


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter_cls = ColorFormatter if settings.log_color else logging.Formatter
    handler.setFormatter(formatter_cls(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    logging.getLogger("httpx").setLevel(logging.DEBUG if settings.debug else logging.WARNING)
    logging.getLogger("gateway").debug(
        "logging configured debug=%s level=%s body_chars=%s color=%s",
        settings.debug,
        settings.log_level,
        settings.log_body_chars,
        settings.log_color,
    )


def render_debug(value: Any, limit: int) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)

    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted = {}
    for key, value in headers.items():
        if key.lower() in {"authorization", "cookie", "set-cookie"}:
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted
