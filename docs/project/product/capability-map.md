# Capability Map — AI Gateway

> Product capability map.

- owner: PO
- status: draft
- last_verified: TBD

## Capabilities

| ID | Capability | Description | Link |
|---|---|---|---|
| CAP-001 | Unified API | `POST /v1/chat/completions`, `POST /v1/messages`, `GET /v1/models` | OBJ-001 |
| CAP-002 | Multi-provider | OpenAI, Anthropic, Claude, Gemini, OpenRouter, Ollama, Blackbox, LM Studio, Freebuff, Codex CLI, Claude Code CLI | OBJ-001 |
| CAP-003 | Registry | Registers metadata: model, capability, cost, latency, quota, health, context, tool support | OBJ-001 |
| CAP-004 | Router | Routes based only on Registry; modes: strict, fallback, auto | OBJ-001 |
| CAP-005 | Management dashboard | Provider, token, history, and tool-approval management on port 2222 | OBJ-001 |
| CAP-006 | Local-agent support | Tool loop, tool approvals, browser and Figma capabilities | OBJ-001 |
| CAP-007 | History | Per-project JSONL retention, recall, and local history import | OBJ-001 |
| CAP-008 | Persisted provider config | JSON-backed provider configuration and immediate registry reload | OBJ-001 |

## Capability principles

- Never hardcode providers.
- No HTTP or CLI dependency.
- All routing goes through the Router.
- A new provider only needs to implement the interface.

## Unconfirmed (TBD)

- Priority of each capability.
- Technical details of each capability.
