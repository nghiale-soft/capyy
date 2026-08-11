# Security Scanning — AI Gateway

> Security scanning.

- owner: QA
- status: draft
- last_verified: TBD

## Status

- CVE scanning: disabled (optional capability).
- Sonar: disabled (optional capability).

## Security principles

- Never hardcode secrets/tokens/keys.
- Never log authorization headers or secrets.
- Validate input at the trust boundary.
- Enforce authorization on the server.

## References

- `docs/ai-read-first/core/SECURITY-BASELINE.md`.
- `docs/ai-read-first/quality/optional/CVE-SCANNING.md`.

## Unconfirmed (TBD)

- Enabling CVE/Sonar.
- Scanning tools and process.
