# API Design — AI Gateway

- owner: Tech Lead
- status: implemented routes documented; detailed protocol schemas remain external contracts
- last_verified: 2026-08-10 (route source review)

## API service — port 1221

All API routes use the optional local bearer authentication configured by
`FREEBUFF_API_KEY`. When it is unset, local authentication is disabled.

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness response: `{"status":"ok"}` |
| GET | `/v1/models` | Models currently exposed by the gateway registry |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions, including streaming when requested |
| POST | `/v1/messages` | Anthropic-compatible messages |

## Management API — port 1221, proxied by dashboard

The dashboard on port `2222` proxies `/api/*` to the API service, so these
endpoints can be called through either local service port when appropriate.

| Area | Routes |
|---|---|
| Providers | `GET/POST /api/providers`, `GET/PUT/DELETE /api/providers/{provider_id}`, `POST /api/providers/fetch-models`, `PUT /api/providers/order`, `POST /api/providers/{provider_id}/test` |
| FreeBuff | `GET /api/freebuff/models`; `GET/POST/PUT/DELETE /api/freebuff/tokens`; `DELETE /api/freebuff/tokens/{index}` |
| Figma | `GET/POST/PUT/DELETE /api/figma/tokens`; `DELETE /api/figma/tokens/{index}` |
| History | `GET /api/history`, `POST /api/history/scan`, `GET/DELETE /api/history/{project_key}` |
| Tool approval | `GET/PUT /api/tools/permissions`, `GET /api/tools/pending`, `POST /api/tools/pending/{approval_id}/approve|deny` |
| Browser runtime | `GET /api/browser/runtime`, `POST /api/browser/install` |

## Contract notes

- The compatibility schemas are implemented in `gateway/compat/` and the
  route handlers, rather than duplicated here.
- Management endpoints return JSON and use FastAPI `400`, `404`, and `409`
  responses for validation/not-found/conflict cases where applicable.
- Payload limits, complete error semantics, and third-party provider contracts
  are not yet a versioned API specification.
