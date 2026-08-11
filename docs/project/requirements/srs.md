# SRS — AI Gateway

> Software Requirements Specification.

- owner: BA
- status: draft
- last_verified: TBD

## 1. Introduction

AI Gateway is a unified gateway layer for any AI provider. Clients configure Base URL,
API Key, and Model; the gateway routes to the right provider or falls back.

## 2. General description

### 2.1 Context
Extension of `freebuff2api` (OpenAI-compatible adapter for Codebuff Freebuff).

### 2.2 Users
- Clients calling the API.
- Developers extending providers.

## 3. Functional requirements

### 3.1 API endpoints
- `POST /v1/chat/completions` — chat completions (OpenAI-compatible).
- `POST /v1/messages` — messages (Anthropic-compatible).
- `GET /v1/models` — model list.
- `GET /healthz` — liveness response.

### 3.2 Multi-provider
- Support: OpenAI, Anthropic, Claude, Gemini, OpenRouter, Ollama, Blackbox, LM Studio, Freebuff, Codex CLI, Claude Code CLI.
- Provider plugins implement the interface: `chat()`, `stream_chat()`, `models()`, `health()`.

### 3.3 Registry
- Scans all providers at startup, registers metadata: model, capability, cost, latency, quota, health, context, tool support.

### 3.4 Router
- Routes based only on the Registry.
- Modes: strict, fallback, auto.

### 3.5 Management workflows
- Dashboard on a separate port proxies management APIs for providers, FreeBuff
  and Figma tokens, history, and tool approvals.

### 3.6 Local capabilities
- Per-project chat history supports retention, recall, and Claude Code/Codex
  import.
- Tool requests can use a local agent loop with persisted allow/ask/deny modes.

### 3.7 Config
- Providers persist to JSON (`config/providers.json` by default); runtime paths
  are configurable through environment variables.

## 4. Non-functional requirements

- Never hardcode providers.
- All routing goes through the Router.
- Never log secrets/tokens.
- Async, manage resource lifecycles.

## 5. Unconfirmed (TBD)

- Detailed payload/response schemas per endpoint.
- Exact provider YAML format.
- Standard error semantics and codes.
- Operational limits (timeout, payload size).
