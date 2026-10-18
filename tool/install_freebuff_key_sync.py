from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path


LABEL = "com.capyy.freebuff-key-sync"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = PROJECT_ROOT / "tool/sync_freebuff_key.py"
PLIST_PATH = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
LOG_DIR = Path.home() / "Library/Logs/capyy"


def main() -> int:
    if sys.platform != "darwin":
        print("This installer only supports macOS LaunchAgents.", file=sys.stderr)
        return 1

    python = shutil.which("python3")
    if not python:
        print("python3 was not found in PATH.", file=sys.stderr)
        return 1

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    launch_agent: dict[str, object] = {
        "Label": LABEL,
        "ProgramArguments": [python, str(SYNC_SCRIPT)],
        "WorkingDirectory": str(PROJECT_ROOT),
        "RunAtLoad": True,
        "StartInterval": 300,
        "StandardOutPath": str(LOG_DIR / "freebuff-key-sync.log"),
        "StandardErrorPath": str(LOG_DIR / "freebuff-key-sync.log"),
    }
    api_key = os.environ.get("CAPYY_LOCAL_API_KEY")
    if api_key:
        launch_agent["EnvironmentVariables"] = {
            "CAPYY_LOCAL_API_KEY": api_key,
        }

    with PLIST_PATH.open("wb") as stream:
        plistlib.dump(launch_agent, stream, sort_keys=True)
    os.chmod(PLIST_PATH, 0o600)

    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["launchctl", "bootout", f"{domain}/{LABEL}"],
        check=False,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["launchctl", "bootstrap", domain, str(PLIST_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(
            f"Could not load LaunchAgent: {result.stderr.strip() or result.stdout.strip()}",
            file=sys.stderr,
        )
        return 1

    kickstart = subprocess.run(
        ["launchctl", "kickstart", "-k", f"{domain}/{LABEL}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if kickstart.returncode:
        print(
            f"LaunchAgent loaded but initial run failed: "
            f"{kickstart.stderr.strip() or kickstart.stdout.strip()}",
            file=sys.stderr,
        )
        return 1

    print(f"Installed {LABEL}; it runs at login and every 5 minutes.")
    print(f"Log: {LOG_DIR / 'freebuff-key-sync.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
