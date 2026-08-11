# Source Tree — AI Gateway

- owner: Tech Lead
- status: implemented tree
- last_verified: 2026-08-10 (filesystem review)

```text
ai-gateway-provider/
├── main.py                     # starts API and dashboard servers
├── registry.py, router.py       # provider registry and routing/failover
├── scheduler.py, judge.py       # project extension points
├── gateway/
│   ├── app.py                  # FastAPI API composition and lifespan state
│   ├── webapp.py               # dashboard server and /api proxy
│   ├── compat/                 # OpenAI/Anthropic conversions and models
│   ├── core/                   # settings, logging, SSE utilities
│   ├── routes/                 # API and management route handlers
│   └── services/               # chat, providers, sessions, history, tools, Figma
├── providers/                  # base, FreeBuff, and OpenAI-compatible providers
├── config/                     # seed provider YAML and runtime JSON files
├── tool/web/                   # dashboard HTML, CSS, and static assets
├── tests/                      # pytest coverage for gateway behavior
├── Dockerfile
└── docker-compose.yml
```

Runtime state is intentionally outside source control: provider configuration,
token files, tool permissions, and chat history are written under `config/` or
`data/` and mounted through Docker named volumes.
