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

- Provider config: JSON file at `config/providers.json` in local mode, or the
  `capyy.providers` and `capyy.provider_models` tables in shared Docker mode.
- Provider credentials are individual rows in `capyy.ai_tokens`, keyed by
  provider and position (for example, `freebuff`, `claude`, or `codex`); each
  row records `created_at` and whether it is the provider's currently selected
  token (`is_active`). The active index is also kept in `capyy.app_settings`
  to preserve the ordered token-pool behavior.
- Tool permissions are individual rows in `capyy.tool_permissions`.
- Registry metadata: loaded from config + health checks.

## freebuff2api references

- `freebuff2api/config.py` — `Settings` dataclass.
- `freebuff2api/models.py` — `FreebuffModel` (model catalog).
- `freebuff2api/codebuff.py` — `FreebuffSession`, `FreebuffRun`, `CodebuffAccount`.

## Unconfirmed (TBD)

- Official provider YAML schema.
- API key storage (env, file, vault?).
- Whether a database is needed (phase 4 dashboard).
