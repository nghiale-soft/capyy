# TODO — AI Gateway

- owner: Dev
- status: active backlog; priorities unconfirmed
- last_verified: 2026-08-10 (source review)

## Implemented baseline

- [x] API gateway, Docker/Compose deployment, and dashboard.
- [x] Provider registry, routing, persisted provider management, and
  OpenAI/Anthropic compatibility routes.
- [x] FreeBuff token pool/retry/cooldown, history/import, local tools and tool
  approvals, Figma/browser support.
- [x] Repository test suite under `tests/`.

## Backlog requiring product or architecture decisions

- [ ] Define provider support and certification matrix.
- [ ] Specify metrics and operational health model.
- [ ] Specify and, if needed, implement quota/circuit-breaker policy.
- [ ] Decide scope and contract for AI judge/cost optimization/learning router.
- [ ] Design multi-node persistence, coordination, deployment, and security.
