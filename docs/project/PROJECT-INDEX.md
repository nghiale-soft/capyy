# PROJECT-INDEX

> Navigation hub for the **AI Gateway** project documentation.
> AI agents and developers should read this file first, then load the exact artifacts needed for a specific task.

- owner: Tech Lead
- status: maintained implementation documentation
- last_verified: 2026-08-10 (source and configuration review)

## Project

| Item | Value |
|---|---|
| Project name | AI Gateway |
| Documentation language | en (fallback: vi) |
| Status | implemented gateway; some advanced roadmap items remain unverified |
| Origin | Extension of a FreeBuff/Codebuff-compatible adapter |
| Main references | `README.md`, `gateway/`, `providers/`, `tests/`, deployment configuration |

## Documentation structure

```text
docs/project/
├── PROJECT-INDEX.md              ← this file
├── project-profile.yaml
├── governance/
│   ├── documentation-ownership.md
│   ├── project-charter.md
│   └── risk-register.md
├── product/
│   ├── vision-and-scope.md
│   ├── capability-map.md
│   └── product-roadmap.md
├── requirements/
│   ├── requirements-index.md
│   ├── srs.md
│   ├── business-rules.md
│   ├── use-cases.md
│   └── screen-specifications.md
├── rules/
│   └── README.md
├── overview/
│   ├── system-overview.md
│   └── application-status.md
├── architecture/
│   ├── system-context.md
│   ├── architecture.md
│   ├── tech-stack.md
│   ├── runtime-flows.md
│   ├── integrations.md
│   └── adrs.md
├── engineering/
│   ├── source-tree.md
│   ├── architecture-patterns.md
│   ├── api-design.md
│   ├── data-model.md
│   └── development-standards.md
├── ux/
│   ├── user-flows.md
│   ├── screen-designs.md
│   ├── design-system.md
│   └── accessibility.md
├── planning/
│   ├── current-task.md
│   ├── implementation-plan.md
│   ├── roadmap.md
│   ├── decision-log.md
│   └── todo.md
├── testing/
│   ├── test-strategy.md
│   ├── test-cases.md
│   ├── automation-plan.md
│   ├── e2e.md
│   └── business-verification.md
├── quality/
│   ├── quality-plan.md
│   ├── traceability-matrix.md
│   ├── security-scanning.md
│   ├── performance-and-load.md
│   └── release-readiness.md
└── release/
    └── changelog.md
```

## Traceability map

![objective → capability → requirement → design → task → test → release]

| Objective | Capability | Requirement | Design | Task | Test |
|---|---|---|---|---|---|
| OBJ-001 | CAP-001 | REQ-001 | ADR-001 | TASK-001 | TC-001 |
| OBJ-001 | CAP-002 | REQ-002 | ADR-002 | TASK-002 | TC-002 |

> Detailed identifiers will be filled in once business requirements and designs are confirmed.

## Reading guide

- **AI agent**: start from `docs/ai-read-first/START-HERE.md` (mandatory entry point), then load the files from this index relevant to the task.
- **Developer**: start from `docs/project/overview/system-overview.md` and `docs/project/architecture/architecture.md`.
- **PM/PO**: start from `docs/project/product/vision-and-scope.md` and `docs/project/product/product-roadmap.md`.

## Unconfirmed (TBD)

- Roles and specific artifact owners (see `governance/documentation-ownership.md`).
- Phase-by-phase roadmap details (see `product/product-roadmap.md`).
- Official traceability identifiers.
