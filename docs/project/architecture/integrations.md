# Integrations — AI Gateway

- owner: SA
- status: source-reviewed
- last_verified: 2026-08-10

| Integration | Current implementation |
|---|---|
| FreeBuff / Codebuff | Native adapter and multi-token account pool; configured through env or persisted token file |
| OpenAI-compatible upstreams | URL/key/model provider configuration managed by `/api/providers` |
| OpenAI and Anthropic clients | Compatibility endpoints exposed on the gateway API |
| Claude Code and Codex local history | Read-only Docker mounts and `/api/history/scan` import support |
| Figma | Token management and Figma tools, with default/per-project token lookup |
| Browser automation | Playwright-based browser tools; Docker image installs Chromium |

The README lists additional provider targets. This document only marks the
integration mechanisms visible in the current source; it does not certify that
every listed third-party provider has been configured or verified.
