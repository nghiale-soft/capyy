# Rules — AI Gateway

> Project-specific rules (complementing `docs/ai-read-first`).

- owner: Tech Lead
- status: draft
- last_verified: TBD

## Main rules

- **Scope**: Only modify/own within `ai-gateway-provider`. Do not touch `freebuff2api` source (another party's project).
- **Never hardcode providers**: every provider is declared in YAML and routed through the Router.
- **Mandatory interface**: providers implement `chat()`, `stream_chat()`, `models()`, `health()`.
- **Security**: never log secrets/tokens; never hardcode secrets.
- **Contract-first**: public API changes require establishing the contract first.

## Sources

- Project `README.md` (root).
- `docs/ai-read-first/core/` — general AI rules.
- `docs/ai-read-first/bootstrap/` — documentation standards.

## Unconfirmed (TBD)

- Detailed coding conventions (format, lint) for the project.
- Review/approval process.
