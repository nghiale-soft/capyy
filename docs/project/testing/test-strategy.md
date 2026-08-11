# Test Strategy — AI Gateway

> Test strategy.

- owner: Tester
- status: draft
- last_verified: TBD

## Goals

- Verify gateway behavior (routing, fallback, protocol conversion).
- Protect backward compatibility with OpenAI/Anthropic standards.
- Guard against regressions during refactors.

## Test levels

| Level | Description | Tool |
|---|---|---|
| Unit | Test each module (registry, router, provider, conversion) | pytest |
| Integration | Test provider/upstream integration | pytest |
| E2E | Test user flows (API workflow) | pytest |
| Business verification | Mandatory for business-code changes | TBD |

## References

- `tests/` — repository test suite for configuration, compatibility, streaming,
  failover, history, provider forms, sessions, tool approvals, and web app.
- `docs/ai-read-first/quality/mandatory/API-SERVICE-WORKFLOW.md`.

## Unconfirmed (TBD)

- Per-phase detailed test cases.
- Coverage targets.
- Test environment (mock upstream).
