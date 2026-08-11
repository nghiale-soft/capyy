# Risk Register — AI Gateway

> Project risk register.

- owner: PM
- status: draft
- last_verified: TBD

## Risk table

| ID | Group | Risk | Impact | Likelihood | Level | Mitigation | Owner |
|---|---|---|---|---|---|---|---|
| RSK-001 | Contract | Changing public API `/v1/chat/completions`, `/v1/messages`, `/v1/models` affects consumers | High | Medium | High | Contract-first gate, versioning | TBD |
| RSK-002 | Security | Leak of API keys/tokens for many providers | High | Medium | High | No hardcoded secrets, encryption, no secret logging | TBD |
| RSK-003 | Compatibility | Losing existing Freebuff adapter behavior during refactor | Medium | High | Medium | Keep scope, regression tests | TBD |
| RSK-004 | Architecture | Hardcoding providers makes extension difficult | Medium | Medium | Medium | Provider interface + Registry + Router | TBD |
| RSK-005 | Operations | Upstream providers change API/rate-limits | Medium | High | Medium | Health checks, quota, circuit breaker | TBD |
| RSK-006 | Performance | Streaming/backpressure across many providers | Medium | Low | Low | Async, stream lifecycle management | TBD |

## Unconfirmed (TBD)

- Detailed probabilities/impacts need real data.
- Owner for each risk.
- Risk acceptance thresholds.
