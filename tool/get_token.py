from __future__ import annotations

import json
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

BASE_FREEBUFF = "https://freebuff.com"
BASE_CODEBUFF = "https://www.codebuff.com"

VERIFY_URL = "https://www.codebuff.com/api/v1/freebuff/session"

POLL_INTERVAL = 2.0
POLL_TIMEOUT = 5 * 60


def _endpoints(mode: str) -> tuple[str, str]:
    base = BASE_CODEBUFF if mode == "codebuff" else BASE_FREEBUFF
    return f"{base}/api/auth/cli/code", f"{base}/api/auth/cli/status"


def request_code(fingerprint_id: str, code_url: str) -> dict:
    body = json.dumps({"fingerprintId": fingerprint_id}).encode()
    req = urllib.request.Request(
        code_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Bun/1.3.11",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def poll_status(
    fingerprint_id: str,
    fingerprint_hash: str,
    expires_at: int,
    status_url: str,
) -> dict:
    qs = urllib.parse.urlencode(
        {
            "fingerprintId": fingerprint_id,
            "fingerprintHash": fingerprint_hash,
            "expiresAt": str(expires_at),
        }
    )
    url = f"{status_url}?{qs}"
    deadline = time.monotonic() + POLL_TIMEOUT
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Bun/1.3.11",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                user = data.get("user")
                if user and user.get("authToken"):
                    return user
        except urllib.error.HTTPError as e:
            if e.code == 401:
                pass
            else:
                print(f"  [warn] status HTTP {e.code}: {e.read()[:200].decode(errors='replace')}")

        print(f"  [{attempt:>3}] pending... (will retry in {POLL_INTERVAL:.0f}s)")
        time.sleep(POLL_INTERVAL)

    raise TimeoutError("login was not completed within 5 minutes")


def verify_token(token: str) -> tuple[bool, str]:
    req = urllib.request.Request(
        VERIFY_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "*/*",
            "User-Agent": "Bun/1.3.11",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        body = e.read()[:200].decode(errors="replace")
        if e.code in (401, 403):
            return False, f"HTTP {e.code} (token rejected): {body}"
        return True, f"HTTP {e.code} (auth ok, endpoint returned: {body})"
    except urllib.error.URLError as e:
        return False, f"network error: {e}"


def write_env(token: str) -> None:
    env_path = Path(__file__).parent / ".env"
    lines: list[str] = []
    found = False
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("FREEBUFF_TOKEN="):
                lines.append(f"FREEBUFF_TOKEN={token}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"FREEBUFF_TOKEN={token}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[ok] wrote FREEBUFF_TOKEN to {env_path}")


def main() -> int:
    mode = "codebuff" if "--codebuff" in sys.argv else "freebuff"
    code_url, status_url = _endpoints(mode)
    print(f"[mode] {mode}  ({code_url.rsplit('/api', 1)[0]})")

    fingerprint_id = f"fb-{secrets.token_hex(8)}"
    print(f"[1/3] requesting auth code  fingerprintId={fingerprint_id}")
    try:
        code = request_code(fingerprint_id, code_url)
    except urllib.error.HTTPError as e:
        print(f"[err] code endpoint failed: HTTP {e.code} {e.read()[:200].decode(errors='replace')}")
        return 1

    login_url = code["loginUrl"]
    fingerprint_hash = code["fingerprintHash"]
    expires_at = code["expiresAt"]
    print(f"      fingerprintHash={fingerprint_hash[:16]}...  expiresAt={expires_at}")

    print("\n[2/3] open this URL in your browser and login:")
    print(f"      {login_url}\n")
    try:
        webbrowser.open(login_url)
    except Exception:
        pass

    print("[3/3] polling for authToken...")
    try:
        user = poll_status(fingerprint_id, fingerprint_hash, expires_at, status_url)
    except TimeoutError as e:
        print(f"[err] {e}")
        return 2

    token = user["authToken"]
    print("\n=== success ===")
    print(f"  id     : {user.get('id')}")
    print(f"  name   : {user.get('name')}")
    print(f"  email  : {user.get('email')}")
    print(f"  token  : {token}")

    print("\n[verify] testing token against codebuff.com/api/v1/freebuff/session ...")
    ok, info = verify_token(token)
    print(f"         {'OK' if ok else 'FAIL'} — {info}")
    if not ok:
        print("         token did NOT authenticate against codebuff.com — do not use.")
        return 3

    if "--write-env" in sys.argv:
        write_env(token)
    else:
        print("\n(tip: rerun with --write-env to auto-update .env)")
        print("(tip: use --codebuff to login codebuff.com)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
