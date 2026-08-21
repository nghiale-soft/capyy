# Streaming contract

An upstream response may be retried or moved to a lower-priority provider only
before its first upstream output chunk. Once output has begun, the gateway must
preserve the stream, emit a protocol-valid error if needed, and release the
active run/lease. Pings and gateway progress are liveness UI only; they never
become persisted assistant content.

Changes require coverage for OpenAI SSE, Anthropic SSE conversion, cancellation,
run/lease cleanup, and pre-output failover.
