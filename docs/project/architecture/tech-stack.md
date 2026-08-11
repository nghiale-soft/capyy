# Tech Stack — AI Gateway

- owner: SA
- status: implemented stack
- last_verified: 2026-08-10 (`pyproject.toml` and Dockerfile review)

| Technology | Version constraint | Purpose |
|---|---:|---|
| Python | >=3.13 | Application runtime |
| FastAPI | >=0.115.0 | API and dashboard web applications |
| Uvicorn | >=0.34.0 | ASGI server |
| httpx with SOCKS extras | >=0.28.0 | Upstream requests and dashboard reverse proxy |
| python-dotenv | >=1.0.0 | `.env` loading |
| Playwright | >=1.45.0 | Browser tools; Chromium is installed in the image |
| uv | lockfile-backed dependency management and container startup |
| Docker Compose | Local container orchestration and persistent volumes |

## Runtime configuration

- API: `FREEBUFF_HOST` / `FREEBUFF_PORT` (default `0.0.0.0:1221`).
- Dashboard: `FREEBUFF_DASHBOARD_ENABLED`, `FREEBUFF_DASHBOARD_HOST`, and
  `FREEBUFF_DASHBOARD_PORT` (default `0.0.0.0:2222`).
- Provider configuration: `AI_GATEWAY_PROVIDERS_FILE`, default
  `config/providers.json`.
- Persistent FreeBuff tokens: `FREEBUFF_TOKENS_FILE`, default
  `config/freebuff-tokens.json`.
- Chat history: `FREEBUFF_HISTORY_*`; tool behavior: `FREEBUFF_TOOL_*`.

`config/providers/freebuff.yaml` is a seed/example artifact, not the default
runtime persistence format.
