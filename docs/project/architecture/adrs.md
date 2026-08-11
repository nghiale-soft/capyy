# ADRs — AI Gateway

> Architecture Decision Records.

- owner: SA
- status: draft
- last_verified: TBD

## ADR-001 — Multi-provider gateway architecture

- **Status**: draft
- **Decision**: Use Client → API Layer → Router → Registry → Providers.
- **Context**: Need a unified gateway for many AI providers.
- **Consequences**: A new provider only implements an interface; routing goes through the Registry.
- **Source**: `ai-gateway-provider` README.

## ADR-002 — Provider persistence is file-backed, not hardcoded

- **Status**: draft
- **Decision**: Providers are managed through the API/dashboard and persisted
  in `config/providers.json` by default, not hardcoded.
- **Context**: Need easy provider extensibility.
- **Consequences**: Provider mutations update persistent JSON and rebuild the
  registry immediately; configuration requires filesystem protection.

## ADR-003 — No HTTP or CLI dependency

- **Status**: draft
- **Decision**: Providers do not depend on a specific HTTP client or CLI.
- **Context**: Ensure portability and separation of concerns.
- **Consequences**: Providers implement a pure interface.

## ADR-004 — OpenAI/Anthropic-compatible protocol

- **Status**: draft
- **Decision**: The API layer normalizes to OpenAI/Anthropic protocols.
- **Context**: Clients configure Base URL / API Key / Model.
- **Consequences**: Reuse `openai_compat.py`, `anthropic_compat.py` from freebuff2api.

## Unconfirmed (TBD)

- Additional ADRs that arise while implementing phases 1-4.
