# Screen Specifications — AI Gateway

- owner: BA
- status: implemented screens inventoried; detailed UX contract pending
- last_verified: 2026-08-10

The dashboard is implemented at `http://localhost:2222/` and is served from
`tool/web/templates/dashboard.html` with static assets in `tool/web/static/`.

| Area | Supported management task |
|---|---|
| Providers | List, add, edit, delete, reorder, test unsaved configuration, and fetch models. FreeBuff supplies a maintained suggestion list while the model field still accepts manual IDs for entitled or newly released models. |
| FreeBuff tokens | List/mask/reveal, add, replace, remove, and clear token pool; live per-token state (`available`, `busy`, `rate limited`, or `error`) refreshes while the Providers screen is open |
| Figma tokens | Manage the Figma token pool |
| Chat history | View project history and import Claude Code/Codex records |
| Tool approval | Inspect permission modes and approve or deny pending actions |

The current dashboard is functional implementation evidence, not an approved
visual specification. Responsive behavior, accessibility acceptance criteria,
and design-token ownership remain to be defined.
