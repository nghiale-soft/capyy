# E2E — AI Gateway

> End-to-end testing.

- owner: Tester
- status: draft
- last_verified: TBD

## E2E scenarios

- Send a chat completion through the gateway → upstream provider → response.
- Send messages (Anthropic) through the gateway.
- Streaming.
- Fallback when a provider fails.

## References

- `docs/ai-read-first/quality/mandatory/API-SERVICE-WORKFLOW.md`.
- `tests/` — current automated test suite; real-upstream E2E remains a separate
  environment-dependent verification need.

## Unconfirmed (TBD)

- E2E environment (mock vs real).
- Test data.
