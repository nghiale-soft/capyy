# Data Model — AI Gateway

> Data model.

- owner: Tech Lead
- status: draft
- last_verified: TBD

## Provider data

Each provider needs configuration and metadata:

- id / name
- base_url
- api_key (sensitive — never logged)
- model list
- capability
- cost
- latency
- quota
- health
- context
- tool support

## Data sources

- Provider config: JSON file at `config/providers.json` by default, controlled
  by `AI_GATEWAY_PROVIDERS_FILE`.
- Registry metadata: loaded from config + health checks.

## freebuff2api references

- `freebuff2api/config.py` — `Settings` dataclass.
- `freebuff2api/models.py` — `FreebuffModel` (model catalog).
- `freebuff2api/codebuff.py` — `FreebuffSession`, `FreebuffRun`, `CodebuffAccount`.

## Unconfirmed (TBD)

- Official provider YAML schema.
- API key storage (env, file, vault?).
- Whether a database is needed (phase 4 dashboard).
