# Architecture Patterns — AI Gateway

> Architecture patterns.

- owner: Tech Lead
- status: draft
- last_verified: TBD

## Patterns

| Pattern | Description | Used in |
|---|---|---|
| Plugin interface | Provider implements interface `chat()`, `stream_chat()`, `models()`, `health()` | Providers |
| Registry | Registers provider metadata at startup | Registry |
| Router | Selects provider based on Registry | Router |
| Adapter | Protocol conversion (OpenAI/Anthropic) | API Layer |
| Proxy | Forwards requests to upstream | Gateway |
| Circuit breaker | Protects when a provider fails | Scheduler |
| Fallback chain | Primary → Secondary → Third | Fallback |

## Reuse from freebuff2api

- `openai_compat.py` — OpenAI protocol conversion.
- `anthropic_compat.py` — Anthropic protocol conversion.
- `codebuff.py` — Codebuff adapter (becomes one provider).
- `sse.py` — SSE encode/decode.
- `logging_config.py` — logging, secret redaction.

## Unconfirmed (TBD)

- Detailed scheduler/judge patterns.
- Specific reuse rules when splitting modules.
