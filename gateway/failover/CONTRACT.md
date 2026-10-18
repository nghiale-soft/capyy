# Failover contract — protected business flow

This folder owns the mandatory order:

```text
Provider priority 1
  -> token 1
  -> token 2
  -> every remaining usable token
Provider priority 2
  -> every usable token when that provider has an account pool
Provider priority N
```

Each account has one exclusive active lease: concurrent requests may borrow
another idle, healthy account in the same provider, but never share a session.
If every healthy account is busy, a request waits for a lease up to the
configured timeout; it must not wait forever.

FreeBuff account failures include auth/quota responses (`401`, `403`, `429`),
timeouts, and upstream `5xx` errors. The account-pool must try every usable
FreeBuff token before the provider layer is allowed to try a lower-priority
enabled provider. If no token works, return an explicit error. A stream may
change account/provider only before an upstream output chunk; otherwise
replaying it could duplicate user-visible assistant content.

API routes must call this folder's policy rather than duplicating status-code
or provider-order conditions. Changes require a regression test proving the
sequence above.
