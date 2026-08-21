# Runtime Flows — Capyy

- owner: SA
- status: source-reviewed implementation
- last_verified: 2026-08-10

## Startup

1. `main.py` loads settings and starts the dashboard thread when enabled.
2. API lifespan loads persisted provider configuration, seeds a FreeBuff entry
   if empty, builds the Registry and Router, and initializes account, history,
   and tool-approval services.
3. The API listens on port `1221` by default; the dashboard listens on `2222`.

## Chat request

1. A client calls the OpenAI or Anthropic compatibility route.
2. The route resolves the requested model, identifies project context, and
   delegates to `GatewayService`. The Anthropic Messages route follows the
   same Registry selection for OpenAI-compatible providers.
3. Router selects the enabled provider with the lowest priority value. Before
   a stream emits its first chunk, `GatewayService` can try the next eligible
   OpenAI-compatible provider when the selected one fails.
   For FreeBuff, a 401/403/429 before its first upstream chunk marks the active
   token unavailable, releases its session/run, and retries the same request
   with the next usable token. The API reports quota exhaustion only after the
   token pool has no usable account; it never retries after output has started
   because that could duplicate assistant content.
   Each FreeBuff token/account has an exclusive active lease. Concurrent
   requests use another idle healthy token when one exists; they wait when the
   only healthy token is busy. Lower-priority providers are used after token
   exhaustion/failure, not as load-balancing targets for busy healthy tokens.
4. The provider returns a normal or streaming response. Compatibility code
   converts it to the client protocol. Native tool schemas and tool calls are
   preserved for OpenAI-compatible providers.
5. The history service stores the resulting exchange; optional recall can add
   relevant previous context before provider execution.

## Management and dashboard

1. Dashboard requests `/api/*` on port `2222`.
2. `gateway.webapp` forwards the method, body, query, and relevant headers to
   the API service on port `1221`.
3. Provider/token/permission/history changes update their file-backed service;
   provider changes rebuild the Registry immediately.
4. When authorized, the dashboard can download the fixed Playwright Chromium
   runtime to the data volume; it does not execute arbitrary install commands.

## Client-owned tools

Claude Code, Codex, Cline, and client-local MCP servers own tool approval and
execution. The gateway never executes a requested client tool in the normal
chat flow. OpenAI-compatible providers receive native tool schemas unchanged.
Because the FreeBuff upstream rejects `tools`, the gateway converts its
tool-request text protocol back into native tool calls; the client executes the
call and sends its result on the following request. If FreeBuff only narrates
an intended action, the gateway makes one private compiler pass containing only
the draft plus declared tool contract. It must emit a protocol call or is
discarded; it is never saved to chat history and never executes the tool.
