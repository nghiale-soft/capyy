# Product Roadmap — AI Gateway

> Phase-based product roadmap.

- owner: PO
- status: draft
- last_verified: TBD

## Roadmap

| Phase | Content | Goal | Status |
|---|---|---|---|
| Phase 1 | Refactor + Registry + Router | Multi-provider gateway foundation | TBD |
| Phase 2 | Rule Engine + Health + Quota + Circuit Breaker | Reliability, resource control | TBD |
| Phase 3 | AI Judge + Cost Optimizer + Learning Router | Automatic cost/route optimization | TBD |
| Phase 4 | Dashboard + Metrics + Multi-node | Monitoring, scaling | TBD |

> Source: `ai-gateway-provider` README.md.

## Phase details

### Phase 1 — Refactor + Registry + Router
- Split provider logic out of `freebuff2api`.
- Build a Registry registering provider metadata.
- Build a Router with strict/fallback/auto modes.

### Phase 2 — Rule Engine + Health + Quota + Circuit Breaker
- Rule engine for routing.
- Health check, quota tracking, circuit breaker.

### Phase 3 — AI Judge + Cost Optimizer + Learning Router
- Integrate an AI judge.
- Optimize cost and learn routing.

### Phase 4 — Dashboard + Metrics + Multi-node
- Monitoring dashboard.
- Metrics and multi-node deployment.

## Unconfirmed (TBD)

- Specific timeline per phase.
- Deliverables and completion criteria per phase.
- Priorities and dependencies between phases.
