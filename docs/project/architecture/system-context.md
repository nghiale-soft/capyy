# System Context — AI Gateway

> System boundaries and context.

- owner: SA
- status: source-reviewed implementation
- last_verified: 2026-08-10

## Actors

| Actor | Role |
|---|---|
| Client | Calls the gateway API (chat/completions, messages, models) |
| Developer / operator | Configures providers and management features through API or dashboard |
| Upstream AI providers | OpenAI, Anthropic, Claude, Gemini, OpenRouter, Ollama, Blackbox, LM Studio, Freebuff, Codex CLI, Claude Code CLI |

## System context

```text
+---------+     +------------+     +--------------------+
| Client  | --> | API :1221  | --> | Upstream providers |
+---------+     +------------+     +--------------------+
                      |
                      v
                 (Registry / Router /
                  file-backed services)
```

## Boundaries

- **Inside**: API layer, router, registry, provider plugins, persisted config,
  history, tokens, tool approval, and dashboard proxy.
- **Outside**: upstream providers and API clients. The dashboard is implemented
  inside the system boundary and listens on port `2222` by default.

## Unconfirmed (TBD)

- Per-upstream integration details.
- Authentication protocol between client and gateway.
