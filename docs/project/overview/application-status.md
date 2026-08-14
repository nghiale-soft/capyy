# Application Status — AI Gateway

- owner: Tech Lead
- status: implemented; verification status varies by capability
- last_verified: 2026-08-10 (source and configuration review)

## Current implementation

The repository contains a running FastAPI gateway, its management dashboard,
Docker deployment files, and a pytest suite. The API defaults to port `1221`;
the dashboard defaults to port `2222`.

| Area | Current state |
|---|---|
| OpenAI-compatible chat and models | Implemented: `/v1/chat/completions`, `/v1/models` |
| Anthropic-compatible messages | Implemented: `/v1/messages` |
| Provider routing and failover | Implemented in Registry/Router; behavior covered by repository tests |
| Provider management | Implemented through `/api/providers/*`; configuration is persisted in `config/providers.json` by default |
| FreeBuff account pool | Implemented with file-backed tokens, round-robin, cooldown, retry, and dashboard management |
| Dashboard | Implemented as a separate FastAPI app on port `2222` |
| Chat history | Implemented as per-project/per-session JSONL storage, bounded current-session recall, read-only virtual browse tools with audit logs, and Claude Code/Codex import |
| Local tool loop and approval | Implemented; tool modes and pending approvals are managed through `/api/tools/*` |
| Figma tokens/tools | Implemented with default and per-project token support |
| Optional Chromium | Dashboard can download the Playwright Chromium runtime to the persistent data volume |
| Scheduler/Judge | Present as project modules; advanced health/quota/circuit-breaker and AI judging remain incomplete or unverified |

## Evidence and limits

- Source of truth for this status: `gateway/`, `providers/`, `registry.py`,
  `router.py`, `main.py`, `docker-compose.yml`, and `tests/`.
- This document records source review only. It does not claim a live Docker
  deployment or real upstream-provider verification on 2026-08-10.
