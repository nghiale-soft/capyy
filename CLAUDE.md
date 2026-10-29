# AI Agent Instructions

Read and follow:

1. `docs/ai-read-first/START-HERE.md`
2. `docs/CAPPY_COMMERCIALIZATION_AND_CREDIT_INTEGRATION.md` (Mandatory for Cappy Image/Video localization and Pricing Gateway credit metering tasks)

before analysis, planning, code changes, or command execution.

## Mandatory completion gate

For every code or runtime change, do not report completion until all applicable
checks have succeeded:

1. Run the relevant automated tests.
2. Rebuild/restart the affected Docker service when the running application is
   changed.
3. Use `curl` against the affected live endpoint and inspect the response.

If any check fails, report it as incomplete and continue diagnosis; never claim
the task is done from a successful build alone.
