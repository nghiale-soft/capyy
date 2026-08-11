from __future__ import annotations

import json
from typing import Any


def encode_sse(data: dict[str, Any] | str) -> bytes:
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"data: {payload}\n\n".encode("utf-8")


def decode_sse_data(line: str) -> dict[str, Any] | str | None:
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data:
        return None
    if data == "[DONE]":
        return data
    return json.loads(data)
