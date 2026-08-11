# Business Verification — AI Gateway

> Business verification.

- owner: QA
- status: draft
- last_verified: TBD

## Requirement

Business verification is mandatory for business-code changes. It must be either:

- E2E for a user-visible workflow, or
- an executable domain workflow test proving the affected behavior.

## Plan

- For every business change, run E2E or a domain workflow test.
- Record evidence: command, result, date, source revision.

## Unconfirmed (TBD)

- Specific business verification scenarios per phase.
- Official test commands.
