# Vision & Scope — AI Gateway

> Product vision and scope.

- owner: PO
- status: draft
- last_verified: TBD

## Vision

**AI Gateway** is a unified gateway layer for any AI provider. Clients only configure
**Base URL, API Key, and Model**. The gateway decides routing to the right provider,
or falls back when needed.

> Source: `ai-gateway-provider` README.md.

## Core value

- **One API, many providers**: clients do not change integration when switching providers.
- **Automation**: routing, fallback, health, quota, circuit breaker.
- **Easy extension**: a new provider only implements the interface.
- **No hardcoding**: provider definitions are persisted and routing goes through
  the Router.

## Product scope

### Users
- Clients calling the API (applications, CLIs, AI tools).
- Developers extending providers.

### Problems solved
- Integrating many separate AI providers is complex.
- No unified routing/fallback mechanism.
- Hard to add new providers.

### Problems not yet established as complete
- Metrics and multi-node operation.
- Real AI Judge integration (phase 3).

## Unconfirmed (TBD)

- Detailed user personas.
- Specific acceptance criteria.
- Success metrics.
