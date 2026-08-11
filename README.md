# Capyy

> A calm bridge between AI agents, local tools, MCP servers, and AI providers.

## Goal

AI Gateway is a unified gateway layer for any AI provider. Clients only need
to configure a Base URL, API Key, and Model. The gateway decides how to route
to the right provider, with fallback when needed.

## Supported

-   OpenAI API
-   Anthropic API
-   Freebuff
-   OpenAI
-   Claude
-   Gemini
-   OpenRouter
-   Ollama
-   Blackbox API
-   LM Studio
-   Codex CLI
-   Claude Code CLI

## API

-   POST /v1/chat/completions
-   POST /v1/messages
-   GET /v1/models
-   GET /healthz

The API listens on port `1221` by default. The management dashboard is served
separately on port `2222`; see [Dashboard web](#dashboard-web-separate-port).

## Architecture

Client → API Layer → Router → Registry → Providers

## Layout

    gateway/
    providers/
    registry/
    router/
    scheduler/
    judge/
    config/providers/

### Provider

Each provider must implement:

-   chat()
-   stream_chat()
-   models()
-   health()

### Registry

Scans all providers at startup and registers:

-   model
-   capability
-   cost
-   latency
-   quota
-   health
-   context
-   tool support

### Router

Routes based only on the Registry.

Modes:

-   strict
-   fallback
-   auto

### Scheduler

-   Health
-   Quota
-   Circuit Breaker

### Judge

Currently only an abstraction; not yet AI-integrated.

## Freebuff configuration (reduce daily-quota 429s)

- **Multiple accounts:** `FREEBUFF_TOKEN=token-a,token-b,token-c` (comma
  separated). The gateway builds an account pool, distributes round-robin, and
  when an account gets a 429 from upstream
  (`free-models-per-day-high-balance`) it automatically **cooldowns** that
  account and switches to another one.
- `FREEBUFF_MAX_TOKENS=8192` — cap `max_tokens` within the daily quota (Claude
  Code asks for 32000 by default, which often triggers 429). `0` = no cap.
- `FREEBUFF_RETRY_ATTEMPTS=4`, `FREEBUFF_RETRY_BASE_DELAY=1.0`,
  `FREEBUFF_RETRY_MAX_DELAY=10.0` — auto-retry 429/5xx; retried 429s also step
  down `max_tokens` (32000→16000→8000…).
- `FREEBUFF_ACCOUNT_COOLDOWN=60.0` — cooldown (seconds) for a 429'd account.
- `FREEBUFF_AD_PROVIDERS=gravity,carbon` — valid upstream ad providers.
- Upstream 4xx (e.g. 429) is passed through to the client as-is, so Claude
  Code retries with its own rate-limit logic instead of seeing a generic 502.

**Configure tokens from the dashboard (no `.env` editing):**

- The dashboard web runs on its **own port** (default **2222**), not on a path
  of the API port: open `http://localhost:2222/`. The *🔑 Freebuff Tokens*
  screen shows the configured accounts (masked) and lets you **add one token
  at a time** or delete any account. Tokens are written to
  `config/freebuff-tokens.json` (gitignored) and the account pool reloads
  immediately.
- **Priority:** the `freebuff-tokens.json` file wins over the
  `FREEBUFF_TOKEN` env. To fall back to env, delete the file (or
  `DELETE /api/freebuff/tokens`).
- API: `GET /api/freebuff/tokens` (status + masked tokens),
  `POST /api/freebuff/tokens` `{"token": "..."}` (add one),
  `PUT /api/freebuff/tokens` `{"tokens": ["a","b"]}` (replace),
  `DELETE /api/freebuff/tokens/{index}` (remove one),
  `DELETE /api/freebuff/tokens` (clear, fall back to env).
- `FREEBUFF_TOKENS_FILE=config/freebuff-tokens.json` — override the file path
  if needed. In Docker use `docker compose up -d` — volumes are auto-mounted
  (see *Docker Compose* below), tokens survive recreate.
- ⚠️ When tokens change mid-flight, the account pool is rebuilt immediately —
  requests/streams running on an old account may be interrupted (the trade-off
  of hot reload). Best done during low traffic.

### Local tool execution (Claude Code / Codex / tool-calling clients)

Freebuff free models **reject any request that carries `tools`** with a 429
(`free-models-per-day-high-balance`) — which is why the Claude VSCode
extension (always sends `read_file`/`bash`/`glob`) failed with 429 while
Postman worked. Two fixes are built in:

1. **Strip tools upstream:** `tools`, `tool_choice` and `parallel_tool_calls`
   are removed from the payload forwarded to Freebuff, so the request is
   always accepted (429 gone).
2. **Local agent loop:** the gateway executes the tools itself. When a
   request carries `tools`, the gateway runs an agent loop: ask the model →
   parse its `<<<TOOL_CALL>>>` → execute the tool locally → feed the result
   back → repeat until the model answers without tools. The final answer is
   streamed to the client in the normal OpenAI/Anthropic SSE format.

**Available tools** (sandboxed to `FREEBUFF_TOOL_WORKDIR`, default `.`):

Filesystem & shell:

- `read_file(path)` / `read_file_lines(path, start, end?)` — read a file / a line range
- `write_file(path, content)` / `edit_file(path, old_string, new_string, replace_all?)` — write / targeted edit (both ask approval)
- `list_dir(path)` / `glob(pattern)` — directory listing / glob
- `grep(pattern, path?)` — regex search
- `bash(command)` — shell command (opt-out with `FREEBUFF_TOOL_BASH_ENABLED=false`; asks approval by default)
- `git_status()` / `git_diff(path?)` — git state / diff

Pure helpers (always allowed):

- `http_get(url)` — fetch a URL (asks approval; network access)
- `base64_encode(text)` / `base64_decode(text)` / `url_encode(text)` / `url_decode(text)`
- `uuid()` / `timestamp()` / `json_parse(text)`

Browser automation is intentionally delegated to an MCP server running in the
client (for example Chrome DevTools MCP in Claude). This keeps Chromium and its
Linux dependencies out of the gateway image; the resulting DOM text or image
tool result is sent back through the client conversation.

Figma (configured on the Dashboard, no file editing needed):

- `figma_get_file(file_key)` / `figma_get_node(file_key, node_id)` / `figma_export_image(file_key, node_id)` — read design files (ask approval)
- **Tokens:** Dashboard → *Settings → Figma tokens* — set a **default** token,
  or a **per-project** token (each project/workspace can carry its own).
  Stored in `config/figma-tokens.json`
  (`{"default": "...", "projects": {"<project-key>": "..."}}`, gitignored).
  Resolution order: project token → default → legacy `config/figma-token.json`
  → `FIGMA_TOKEN` env. The figma tools automatically pick the token of the
  current project (chat-history project key).

**Tool permissions** apply to every tool above — new browser/figma tools default
to **Ask**, helpers to **Allow** — and the Dashboard Settings list updates
automatically from the backend labels.

**Security notes:**

- Paths are resolved against `FREEBUFF_TOOL_WORKDIR`; `..` traversal outside
  the workdir is blocked and reported back to the model as an error.
- `FREEBUFF_TOOL_MAX_ITERATIONS=8` — max model↔tool round trips (safety
  bound against infinite loops).
- `FREEBUFF_TOOL_COMMAND_TIMEOUT=30.0` — per-command timeout (seconds).
- `FREEBUFF_TOOL_OUTPUT_CAP=50000` / `FREEBUFF_TOOL_FILE_CAP=100000` —
  output/read caps in characters; long results are truncated with a notice.

**Tool approvals (Dashboard → Settings):**

Each tool has a mode — **Allow** (run immediately), **Ask** (pause and wait
for your approval on the Dashboard), **Deny** (block). Defaults: read-only
tools (read_file/list_dir/glob/grep) = Allow; write_file and bash = Ask.

- Modes are persisted in `config/tool-permissions.json` (gitignored) and are
  editable in the Dashboard → *Settings → Tool permissions*.
- When a tool in "Ask" mode is called, the agent loop pauses and a **pending
  approval** appears on the Dashboard (sidebar badge with a red counter; list
  with Approve/Deny buttons). The write_file approval previews the first line
  of the content so you know what will be written.
- `FREEBUFF_TOOL_APPROVAL_TIMEOUT=120.0` — seconds to wait before the tool
  call is auto-denied (timeout message is fed back to the model).
- API: `GET/PUT /api/tools/permissions`, `GET /api/tools/pending`,
  `POST /api/tools/pending/{id}/approve|deny`.

### Dashboard web (separate port)

- Dashboard runs on a **separate port** from the API:
  `FREEBUFF_DASHBOARD_PORT=2222` (default 2222); set
  `FREEBUFF_DASHBOARD_ENABLED=false` to disable.
- The API port (1221) no longer serves UI on paths.
- The dashboard proxies every `/api/*` to the gateway — no CORS needed, no
  secrets leak into HTML.
- Docker: publish `-p 2222:2222`.

### Docker Compose (auto-mount, no manual steps)

```bash
cd capyy
docker compose up -d --build
```

Volumes are declared in `docker-compose.yml`, **no manual `-v` mounting**:

| Volume | Mount | Content |
|---|---|---|
| `gateway-config` | `/app/config` | `freebuff-tokens.json`, `providers.json` |
| `gateway-data` | `/app/data` | per-project chat history (`data/chat_history`) |

- Container runs as user `appuser` (uid 1000); volumes inherit ownership from
  the image so the gateway can write immediately — no permission errors.
- Data lives in named volumes and survives recreate/upgrade. Full reset:
  `docker compose down -v`.
- Configure via `.env` (`FREEBUFF_TOKEN=...`); if `.env` is missing, compose
  still runs (add tokens later through the dashboard).
- `~/.claude` and `~/.codex` are mounted **read-only** so the dashboard scan
  feature can read your local AI history inside the container.

### Per-project chat history

The gateway stores conversations (user/assistant) per project in
`data/chat_history/chats/<project-key>.jsonl` and **remembers past
conversations** when you ask questions like *"do you remember the 429 issue
last time?"*:

- **Full fidelity like Claude/Codex:** each line stores **content**,
  **thinking** (model reasoning) and **tool calls** (name + arguments) — all
  viewable on the Dashboard.
- **Project mapping:** one file per project. The project key is stable, based
  on the **git remote URL** when available, otherwise the folder name. The
  index `data/chat_history/projects.json` records every path ever seen, so
  when a **folder is renamed or moved**, old history is still merged back into
  the same project (nothing is lost).
- **Clients declare the project:** send header `x-project-path:
  /path/to/project` (or `x-project-id`) on chat requests; otherwise the
  gateway looks in `metadata` of the body (cwd/project_path/workspace). If
  nothing is found, it falls back to the `default` project.
- **Recall:** when a question hints at the past ("bạn có nhớ", "lần trước",
  "hôm qua", "remember", "last time"...), the gateway injects relevant older
  exchanges into the context (filtered by specific keywords in the question,
  e.g. "429").
- **Retention: 1 year max:** rows older than
  `FREEBUFF_HISTORY_MAX_AGE_DAYS=365` are pruned automatically.
- Config: `FREEBUFF_HISTORY_DIR=data/chat_history`,
  `FREEBUFF_HISTORY_INJECT_MODE=memory_only` (inject when asked about the
  past), `always` (inject on every request) or `off`,
  `FREEBUFF_HISTORY_CONTEXT_MAX_CHARS=4000`.

**Source of each exchange (2 dimensions):** every record carries
`meta.source` — the **client** you talk through — plus `meta.provider` — the
**AI backend** that actually answers — plus `meta.via` (`gateway` when the
request passed through the gateway). Shown as badges on the Dashboard:

| Scenario | source (client) | provider (backend) | via |
|---|---|---|---|
| Claude extension → Gateway → FreeBuff | `claude` (Anthropic route) | `FreeBuff` | `gateway` |
| Cursor/Postman → Gateway → FreeBuff | `api` (OpenAI route) | `FreeBuff` | `gateway` |
| Scanned Claude Code local history | `claude` | (from history) | — |
| Scanned Codex CLI local history | `codex` | (from history) | — |

**View history on the Dashboard:** open `http://localhost:2222/` → *💬 Chat
history* — project list with source badges; click a project to open the
conversation in a **modal** (with thinking and tool calls), delete a project.

**History API:**

- `GET /api/history` — project list + stats
- `GET /api/history/{project}` — messages (query: `limit`, `offset`)
- `DELETE /api/history/{project}` — delete a project

**Auto-scan local AI history (no copy-paste):**

`POST /api/history/scan` — or press *🔍 Scan from other AI* on the Dashboard:

- **Claude Code** (`~/.claude/projects/**/*.jsonl`) — keeps thinking,
  tool_use, tool_result, timestamps; maps to the right project via `cwd`
- **Codex CLI** (`~/.codex/sessions/**/*.jsonl`) — `session_meta` → project
  via `cwd`
- Idempotent: each session is tagged `meta.session_id`; re-scanning never
  duplicates.
- Only records within the 365-day window are imported.
- In Docker both directories are auto-mounted (read-only), so scanning runs
  inside the container with no extra setup.

## Fallback

Each policy has Primary, Secondary, Third.

## Config

Provider definitions are managed through the dashboard/API and persist to
`config/providers.json` by default (`AI_GATEWAY_PROVIDERS_FILE` can override
the path). `config/providers/freebuff.yaml` remains a seed/example file.

## Principles

-   Never hardcode providers.
-   No HTTP or CLI dependency.
-   All routing goes through the Router.
-   A new provider only needs to implement the interface.

## Roadmap

Phase 1: Refactor + Registry + Router

Phase 2: Rule Engine + Health + Quota + Circuit Breaker

Phase 3: AI Judge + Cost Optimizer + Learning Router

Phase 4: Dashboard + Metrics + Multi-node
