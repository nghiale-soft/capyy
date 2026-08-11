# Architecture — AI Gateway

- owner: SA
- status: source-reviewed implementation
- last_verified: 2026-08-10

```text
Clients ──> API (1221) ──> compatibility routes ──> GatewayService
                         │                         │
Dashboard (2222) ─proxy──┘                         v
                                             Router ──> Registry ──> providers
                                                │
                                config, account pool, history, tool approval
```

## Implemented components

| Component | Implementation |
|---|---|
| API and lifecycle | `gateway.app`; state is initialized in FastAPI lifespan |
| Protocol compatibility | `gateway.routes.chat`, `messages`, `models`, and `gateway.compat` |
| Provider management | `ProviderCrudService`, persisted `ProviderConfig`, and `/api/providers` |
| Routing | `Router` and `ProviderRegistry`; registry rebuilds after provider mutation |
| Providers | FreeBuff adapter plus OpenAI-compatible provider implementation |
| Dashboard | `gateway.webapp` serves HTML/static assets and proxies management APIs |
| Local capabilities | Per-project history, import scanner, tool loop/approval, browser and Figma services |

## Boundaries and incomplete areas

- Provider configuration persists as JSON by default; it is not dynamically
  discovered from YAML at startup.
- Scheduler and Judge files exist, but this review does not establish a
  complete production rule-engine, circuit-breaker, cost-optimizer, or AI-judge
  implementation.
- The dashboard is a local management UI, not an independent API backend.
