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
If every healthy account is busy, a request waits; this condition alone must
not skip to a lower-priority provider. Provider fallback starts only after the
current provider has no usable account because of an auth/quota failure.

FreeBuff account failures are `401`, `403`, and `429`. The account-pool must
exhaust usable FreeBuff tokens before the provider layer is allowed to try a
lower-priority enabled provider. A stream may change provider only before an
upstream output chunk; otherwise replaying it could duplicate user-visible
assistant content.

API routes must call this folder's policy rather than duplicating status-code
or provider-order conditions. Changes require a regression test proving the
sequence above.
