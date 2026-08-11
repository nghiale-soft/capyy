# System Overview — AI Gateway

> System overview.

- owner: SA
- status: source-reviewed implementation
- last_verified: 2026-08-10

## Overview

AI Gateway is a unified gateway layer for any AI provider. Clients configure
Base URL, API Key, and Model. The gateway routes to the right provider or falls back.

## High-level architecture

```text
Client → API (1221) → Router/Registry → Providers
Dashboard (2222) → management API proxy → API (1221)
```

## Main flows

1. Client calls the API (chat/completions, messages, models).
2. API Layer receives and normalizes the request.
3. Router decides the provider based on the Registry.
4. Registry provides provider metadata (model, capability, cost, latency, quota, health, context, tool support).
5. Provider processes and returns the response.

## Components

| Component | Description |
|---|---|
| API Layer | Standardized endpoints (OpenAI/Anthropic-compatible) |
| Router | Routes based on Registry; modes strict/fallback/auto |
| Registry | Registers provider metadata at startup |
| Providers | Provider plugins implementing the interface |
| Dashboard | Separate management web app for providers, tokens, history, and approvals |
| Local services | Token pool, chat history/import, tool approvals, Figma and browser tools |
| Scheduler / Judge | Extension modules; advanced policy features are not established as complete |

## Origin

Extension of `freebuff2api` (OpenAI-compatible adapter for Codebuff Freebuff).

## Unconfirmed (TBD)

- Production topology, metrics, and multi-node behavior.
