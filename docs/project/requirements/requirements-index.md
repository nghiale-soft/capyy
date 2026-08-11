# Requirements Index — AI Gateway

> Requirements navigation hub.

- owner: BA
- status: draft
- last_verified: TBD

## Requirements

| ID | Requirement | Link | Status |
|---|---|---|---|
| REQ-001 | Support `POST /v1/chat/completions` | CAP-001 | unverified |
| REQ-002 | Support `POST /v1/messages` | CAP-001 | unverified |
| REQ-003 | Support `GET /v1/models` | CAP-001 | unverified |
| REQ-004 | Support multiple providers (OpenAI, Anthropic, Claude, Gemini, OpenRouter, Ollama, Blackbox, LM Studio, Freebuff, Codex CLI, Claude Code CLI) | CAP-002 | unverified |
| REQ-005 | Registry registers provider metadata | CAP-003 | unverified |
| REQ-006 | Router routes per Registry with strict/fallback/auto modes | CAP-004 | unverified |
| REQ-007 | Scheduler: health, quota, circuit breaker | CAP-005 | unverified |
| REQ-008 | Judge abstraction | CAP-006 | unverified |
| REQ-009 | Fallback policy Primary/Secondary/Third | CAP-007 | unverified |
| REQ-010 | Provider configuration persisted in JSON by default | CAP-008 | source-reviewed |
| REQ-011 | Dashboard management API and UI | CAP-005 | source-reviewed |
| REQ-012 | Per-project history and local tool approval workflows | CAP-006, CAP-007 | source-reviewed |

## Related files

- `requirements/srs.md` — software requirements specification.
- `requirements/business-rules.md` — business rules.
- `requirements/use-cases.md` — use cases.
- `requirements/screen-specifications.md` — screen specifications (UI).

## Unconfirmed (TBD)

- Acceptance criteria per REQ.
- Priority (MoSCoW) per requirement.
