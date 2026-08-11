# User Flows — AI Gateway

> User flows.

- owner: Design
- status: source-reviewed implemented flows
- last_verified: 2026-08-10

## Actors

- Client (calls the API).
- Developer (extends providers).
- Admin (manages the gateway through the dashboard).

## Planned flows

### Flow 1 — Configure a provider
- Admin adds a provider: enter base_url, API key, model.
- Save the JSON-backed configuration through the management API.
- Registry reloads immediately.

### Flow 2 — Use the API
- Client sends a chat completion with a model.
- Gateway routes + (fallback if needed).
- Client receives the response.

### Flow 3 — Manage local gateway
- Admin uses the dashboard to manage tokens, inspect history, and resolve
  pending tool approvals.

## Current UI implementation

- `tool/web/templates/dashboard.html` — management dashboard.
