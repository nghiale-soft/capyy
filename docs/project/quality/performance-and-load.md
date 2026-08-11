# Performance & Load — AI Gateway

> Performance and load.

- owner: QA
- status: draft
- last_verified: TBD

## Goals

- Gateway handles concurrent requests across many providers.
- Smooth streaming with good backpressure.
- Resource management (connections, threads).

## Considerations

- Async (`httpx.AsyncClient`, `asyncio`).
- Account pool / connection pool.
- Timeout, retry, circuit breaker.

## Unconfirmed (TBD)

- Benchmark targets (rps, latency).
- Load tests.
- Quota/rate-limit thresholds.
