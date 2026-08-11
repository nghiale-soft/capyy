# Development Standards — AI Gateway

> Development standards.

- owner: Tech Lead
- status: draft
- last_verified: TBD

## Python

- Use `uv` (sync, lockfile).
- Format: `ruff format` (planned).
- Lint: `ruff check` (planned).
- Compile: `python -m compileall <affected-path>`.
- Test: `pytest`.

## General rules

- Never hardcode secrets/tokens.
- Never log authorization headers or secrets.
- No HTTP or CLI dependency inside providers.
- Contract-first for public APIs.
- Async, manage resource lifecycles.
- Do not invent requirements or architecture.

## References

- `docs/ai-read-first/core/ENGINEERING-STANDARDS.md`.
- `docs/ai-read-first/tools/backend/PYTHON.md`.

## Unconfirmed (TBD)

- ruff configuration in AI Gateway's `pyproject.toml`.
- CI/CD pipeline.
- Review/approval process.
