# Business Rules — AI Gateway

> Business rules.

- owner: BA
- status: draft
- last_verified: TBD

## Rules verified from README

| ID | Rule | Source |
|---|---|---|
| BR-001 | Never hardcode providers. | README |
| BR-002 | API and dashboard run on separate configurable ports (1221/2222 by default). | `main.py`, settings |
| BR-003 | All gateway request routing goes through the Router and Registry. | `gateway/app.py` |
| BR-004 | Provider changes reload the Registry without a container restart. | `gateway/app.py`, providers route |
| BR-005 | FreeBuff tokens are read only from the Dashboard-managed token file. | session service, README |
| BR-006 | Local tool execution uses allow/ask/deny permissions and times out to denial. | tool approval service |
| BR-007 | Provider configuration persists to `config/providers.json` unless overridden. | settings, provider CRUD service |

## Rules from freebuff2api source (reference)

- Never log authorization headers / secrets (from `logging_config.py` `redact_headers`).
- Handle multiple token accounts via round-robin (from `codebuff.py` `CodebuffAccountPool`).

## Unconfirmed (TBD)

- Gateway authentication/authorization rules.
- Detailed quota/rate-limit rules.
- Specific fallback rules (when to switch Primary → Secondary → Third).
