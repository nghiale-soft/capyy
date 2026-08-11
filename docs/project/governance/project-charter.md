# Project Charter — AI Gateway

> Project charter: goals, scope, milestones, dependencies, risks.

- owner: PM
- status: draft
- last_verified: TBD

## Context

The **AI Gateway** project is an extension of `freebuff2api` — an OpenAI-compatible
adapter for Codebuff Freebuff. The goal is to upgrade it into a **unified gateway layer**
for many AI providers, letting clients only configure Base URL, API Key, and Model.

> Origin: `ai-gateway-provider` README.md (AI Gateway description) + `freebuff2api` source.

## Goals (S.M.A.R.T.)

- **S**: Provide a single gateway API routing to many AI providers.
- **M**: Support providers: OpenAI, Anthropic, Claude, Gemini, OpenRouter, Ollama, Blackbox, LM Studio, Freebuff, Codex CLI, Claude Code CLI.
- **A**: Clients only configure Base URL, API Key, Model; the gateway routes/falls back automatically.
- **R**: Reuse existing adapter code from `freebuff2api`.
- **T**: 4-phase roadmap framework (see `product/product-roadmap.md`).

## Scope

### In scope
- Unified API layer: `POST /v1/chat/completions`, `POST /v1/messages`, `GET /v1/models`.
- Architecture: Client → API Layer → Router → Registry → Providers.
- Provider plugins implementing the interface: `chat()`, `stream_chat()`, `models()`, `health()`.
- Registry with metadata: model, capability, cost, latency, quota, health, context, tool support.
- Router with modes: strict, fallback, auto.
- Scheduler: health, quota, circuit breaker.
- Judge abstraction (not yet AI-integrated).
- File-backed provider configuration managed through the API/dashboard.

### Not established as complete
- Real AI Judge integration.
- Metrics and multi-node operation.

## Planned milestones

| Phase | Content | Status |
|---|---|---|
| Phase 1 | Refactor + Registry + Router | TBD |
| Phase 2 | Rule Engine + Health + Quota + Circuit Breaker | TBD |
| Phase 3 | AI Judge + Cost Optimizer + Learning Router | TBD |
| Phase 4 | Metrics + Multi-node (dashboard already exists) | TBD |

## Dependencies

- Python 3.13+ (from this repository's `pyproject.toml`).
- FastAPI, httpx[socks], uvicorn, python-dotenv.
- Future health/rate-limit/circuit-breaker capabilities may need additional
  dependencies after design approval.

## Main risks

- Public API / contract change risk (from `requirements-index`).
- Security risk: storing API keys/tokens of many providers.
- Backward-compatibility risk with current freebuff2api behavior.

## Unconfirmed (TBD)

- Official PM/PO/SA roles.
- Budget, timeline, resources.
- Per-phase Definition of Done criteria.
