# Test Cases — AI Gateway

> Test case scenarios.

- owner: Tester
- status: draft
- last_verified: TBD

## Test cases

| ID | Scenario | Expected result | Status |
|---|---|---|---|
| TC-001 | `POST /v1/chat/completions` with a valid model | OpenAI-compatible response | unverified |
| TC-002 | `POST /v1/messages` with a valid model | Anthropic-compatible response | unverified |
| TC-003 | `GET /v1/models` | Model list from Registry | unverified |
| TC-004 | Primary provider fails | Fallback to Secondary/Third | unverified |
| TC-005 | Model not in Registry | Clear error | unverified |
| TC-006 | Streaming chat completion | Streams chunks | unverified |
| TC-007 | Provider without health | Registry handles it | unverified |

## References

- `tests/` — executable repository tests for the listed gateway areas.

## Unconfirmed (TBD)

- Per-phase detailed test cases.
- Test data (mock providers).
