# Use Cases — AI Gateway

> Use case descriptions.

- owner: BA
- status: draft
- last_verified: TBD

## Actors

| Actor | Description |
|---|---|
| Client | Application/CLI/tool calling the gateway API |
| Developer | Developer extending providers |
| Extension package | New provider plugin |

## Use cases

### UC-001 — Send chat completion
- **Actor**: Client
- **Description**: Client sends `POST /v1/chat/completions` with a model. Gateway routes to the right provider.
- **Precondition**: Provider registered in the Registry.
- **Postcondition**: OpenAI-compatible response returned.
- **Status**: unverified

### UC-002 — Send messages (Anthropic)
- **Actor**: Client
- **Description**: Client sends `POST /v1/messages`. Gateway routes to a provider.
- **Status**: unverified

### UC-003 — List models
- **Actor**: Client
- **Description**: Client calls `GET /v1/models`, receives the model list from the Registry.
- **Status**: unverified

### UC-004 — Add a new provider
- **Actor**: Developer
- **Description**: Developer implements `chat()`, `stream_chat()`, `models()`, `health()` and declares YAML.
- **Status**: unverified

### UC-005 — Fallback routing
- **Actor**: Client
- **Description**: When the Primary provider fails, the Router switches to Secondary/Third per policy.
- **Status**: unverified

## Unconfirmed (TBD)

- Detailed flow scenarios per use case.
- Exception scenarios.
- Acceptance conditions.
