# API Design — AI Gateway

- owner: Tech Lead
- status: implemented routes documented; detailed protocol schemas remain external contracts
- last_verified: 2026-10-20 (image translation/repair route source review)

## API service — port 1221

All API routes use the optional local bearer authentication configured by
`FREEBUFF_API_KEY`. When it is unset, local authentication is disabled.

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness response: `{"status":"ok"}` |
| GET | `/v1/models` | Models currently exposed by the gateway registry |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions, including streaming when requested |
| POST | `/v1/messages` | Anthropic-compatible messages |
| POST | `/v1/images/translate` | Translate text in an image; returns JPEG binary by default or JSON with `image_base64` |
| POST | `/v1/images/{image_id}/repair` | User-initiated repair of one or more translated regions |
| GET | `/v1/images/{image_id}` | Session metadata and version history |
| GET | `/v1/images/{image_id}/content` | Rendered JPEG for the current or requested version |
| DELETE | `/v1/images/{image_id}` | Delete a session and all rendered versions |

### Image translation and repair contract

`POST /v1/images/translate` keeps the previous request fields (`image_base64`
or `image_url`, `target_language`, `instruction`, `model`, `provider`,
`response_format`, `enable_qc`). JSON responses now add `id`, `version`, and
`expires_at`, and every region includes a stable `region_id` assigned by the
gateway. Binary responses expose the same identifiers in
`X-Capyy-Image-Id`, `X-Capyy-Image-Version`, and `X-Capyy-Image-Expires-At`.

`POST /v1/images/{image_id}/repair` accepts JSON:

| Field | Required | Meaning |
|---|---|---|
| `instruction` | yes | User description of the defect (max 2000 chars) |
| `region_ids` | no | Stable ids from the translate response; at least one target is required |
| `bbox_2d` | no | `[ymin, xmin, ymax, xmax]` in 0–1000 coordinates; selects overlapping regions when `region_ids` is omitted |
| `base` | no | Only `original` is supported in v1; re-rendering from a previous output would duplicate text |
| `response_format` | no | `json` (default) or `image` |
| `enable_qc` | no | Run the optional LLM inspection in addition to deterministic QA |
| `idempotency_key` | no | Replays the stored version for the same key without another model call |

Repair re-renders from the original image with the full merged region state.
Only target regions may change; non-target changes returned by the model are
ignored. The response includes the new `version`, `parent_version`, `regions`,
and a `qc_report` with a `repair_check` entry.

### Deterministic QA

Every translation runs a token-free deterministic check for source-language ink
that survived inpainting. The report keeps the existing
`approved`/`qc_score`/`feedback` fields and adds `deterministic: true` and
region-level `issues`. The old LLM acceptance pass remains available through
`enable_qc`; it is an opt-in add-on, not the default.

### Image session storage

Sessions are file-backed under `CAPYY_IMAGE_STORE_DIR` (default
`data/image_sessions`). Each session stores the original image, every rendered
version, and `state.json`. Defaults:

- Retention: `CAPYY_IMAGE_STORE_TTL_HOURS=168` (7 days)
- Per-file limit: `CAPYY_IMAGE_STORE_MAX_BYTES=26214400` (25 MiB)
- IDs are unguessable `img_<uuid4hex>` values and are validated before any
  filesystem access.
- Expired sessions are removed lazily on access and during startup cleanup.
- The current implementation is single-process and local-disk; the store
  interface is intentionally swappable for object storage or a database.

## Management API — port 1221, proxied by dashboard

The dashboard on port `2222` proxies `/api/*` to the API service, so these
endpoints can be called through either local service port when appropriate.

| Area | Routes |
|---|---|
| Providers | `GET/POST /api/providers`, `GET/PUT/DELETE /api/providers/{provider_id}`, `POST /api/providers/fetch-models`, `POST /api/providers/test-config` (non-destructive connectivity test for unsaved form values; FreeBuff tests do not create a session), `PUT /api/providers/order`, `POST /api/providers/{provider_id}/test` |
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
