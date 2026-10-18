from __future__ import annotations

import fcntl
import hashlib
import json
import os
import plistlib
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


KEYCHAIN_SERVICE = "freebuff-cli"
DEFAULT_CREDENTIALS = Path.home() / ".config/manicode/credentials.json"
DEFAULT_API_URL = "http://127.0.0.1:2222/api/freebuff/tokens"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = PROJECT_ROOT / "data/app/freebuff-key-sync.json"
LOCK_FILE = PROJECT_ROOT / "data/app/freebuff-key-sync.lock"


class SyncError(RuntimeError):
    pass


def read_cli_token(
    credentials_path: Path = DEFAULT_CREDENTIALS,
    security_command: str = "/usr/bin/security",
) -> str | None:
    """Read Freebuff CLI's token from macOS Keychain without displaying it."""
    try:
        metadata = json.loads(credentials_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        raise SyncError("Could not read Freebuff CLI credential metadata.") from error

    default = metadata.get("default") if isinstance(metadata, dict) else None
    if not isinstance(default, dict) or default.get("tokenStore") != "keychain":
        return None

    try:
        result = subprocess.run(
            [
                security_command,
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                str(credentials_path),
                "-w",
            ],
            check=False,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SyncError("Could not access the macOS Keychain.") from error

    if result.returncode != 0:
        raise SyncError(
            "Freebuff CLI Keychain item was not readable; check Keychain access."
        )
    try:
        token = result.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise SyncError("Freebuff CLI Keychain item was not valid UTF-8.") from error
    return token or None


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as error:
        raise SyncError("Could not read the Freebuff sync state.") from error
    return value if isinstance(value, dict) else {}


def _write_state(path: Path, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"version": 1, "token_sha256": digest},
        ensure_ascii=True,
        indent=2,
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            temp.write(payload)
            temp.write("\n")
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    except OSError as error:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise SyncError("Could not save the Freebuff sync state.") from error


def _add_token(api_url: str, token: str, api_key: str | None) -> bool:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        api_url,
        data=json.dumps({"token": token}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if not 200 <= response.status < 300:
                raise SyncError(
                    f"Capyy token endpoint returned HTTP {response.status}."
                )
            return True
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read())
        except (OSError, json.JSONDecodeError):
            body = {}
        detail = body.get("detail") if isinstance(body, dict) else None
        if error.code == 400 and detail == "token already exists":
            return False
        if error.code == 401:
            raise SyncError(
                "Capyy rejected the local API request; configure "
                "CAPYY_LOCAL_API_KEY for the LaunchAgent."
            ) from error
        raise SyncError(f"Capyy token endpoint returned HTTP {error.code}.") from error
    except (OSError, TimeoutError) as error:
        raise SyncError("Could not reach Capyy at the local dashboard API.") from error


def sync_token(
    token: str | None,
    *,
    api_url: str = DEFAULT_API_URL,
    state_path: Path = STATE_FILE,
    api_key: str | None = None,
) -> str:
    if not token:
        return "No Freebuff CLI token is currently available."

    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if _read_state(state_path).get("token_sha256") == digest:
        return "Freebuff CLI token is already synced."

    added = _add_token(api_url, token, api_key)
    _write_state(state_path, digest)
    if added:
        return "Imported a new Freebuff CLI token into Capyy."
    return "Freebuff CLI token was already present in Capyy."


def main() -> int:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        try:
            token = read_cli_token()
            message = sync_token(
                token,
                api_key=os.environ.get("CAPYY_LOCAL_API_KEY"),
            )
        except SyncError as error:
            print(f"Freebuff key sync failed: {error}", file=sys.stderr)
            return 1
        print(message)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
