# Documentation Ownership

> Responsibility and approval mapping for each documentation artifact.
> Per the role model in `docs/ai-read-first/bootstrap/PROJECT-DOCUMENTATION-STANDARDS.md`.

- owner: Tech Lead
- status: draft
- last_verified: TBD

## Role model

| Role | Primary responsibility | Typical artifacts |
|---|---|---|
| PM | Delivery goals, scope, milestones, dependencies, risks | project charter, delivery plan, risk register |
| PO | Product vision, stakeholders, capability, priorities, backlog | product vision, capability map, roadmap, backlog |
| BA | Business process, rules, actors, use cases, functional requirements | SRS, process flows, use cases, screen specs |
| SA | System architecture, technology, integrations, runtime flows, deployment | architecture, tech stack, integration map, ADR |
| Tech Lead | Source structure, modules, patterns, coding conventions, dependencies | source tree, implementation patterns, engineering standards |
| Design | User journey, interaction states, visual system, responsive | UX flows, screen designs, design tokens |
| Tester | Test strategy, scenarios, cases, automation, regression, E2E | test plan, test cases, automation map, evidence |
| QA | Quality governance, traceability, quality gates, security, release | quality plan, traceability matrix, security/load gates |
| Dev | Implementation plan, code, technical tests, migration, evidence | current task, implementation plan, technical design |

## Current owners

> Roles are not yet assigned to specific people. Needs user confirmation.

| Artifact | Primary role | Current owner |
|---|---|---|
| PROJECT-INDEX.md | Tech Lead | TBD — user confirmation required |
| project-profile.yaml | Tech Lead | TBD — user confirmation required |
| project-charter.md | PM | TBD — user confirmation required |
| risk-register.md | PM | TBD — user confirmation required |
| vision-and-scope.md | PO | TBD — user confirmation required |
| capability-map.md | PO | TBD — user confirmation required |
| product-roadmap.md | PO | TBD — user confirmation required |
| srs.md | BA | TBD — user confirmation required |
| business-rules.md | BA | TBD — user confirmation required |
| use-cases.md | BA | TBD — user confirmation required |
| screen-specifications.md | BA | TBD — user confirmation required |
| architecture.md | SA | TBD — user confirmation required |
| tech-stack.md | SA | TBD — user confirmation required |
| source-tree.md | Tech Lead | TBD — user confirmation required |
| api-design.md | Tech Lead | TBD — user confirmation required |
| user-flows.md | Design | TBD — user confirmation required |
| screen-designs.md | Design | TBD — user confirmation required |
| test-strategy.md | Tester | TBD — user confirmation required |
| test-cases.md | Tester | TBD — user confirmation required |
| quality-plan.md | QA | TBD — user confirmation required |
| traceability-matrix.md | QA | TBD — user confirmation required |
| current-task.md | Dev | TBD — user confirmation required |
| implementation-plan.md | Dev | TBD — user confirmation required |
| changelog.md | Dev | TBD — user confirmation required |

## Metadata status

Each important documentation artifact should record:

```yaml
owner: <role>
contributors: []
status: draft | in-review | accepted | superseded | unverified
last_verified: YYYY-MM-DD | TBD
evidence:
  - <source path, decision, command, test, or user confirmation>
```

AI-generated content must not be labeled `accepted` without user confirmation or repository evidence.
