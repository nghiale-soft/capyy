from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tool.sync_freebuff_key import read_cli_token, sync_token


class FreebuffKeySyncTests(unittest.TestCase):
    def test_reads_keychain_token_using_cli_credentials_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            credentials = Path(directory) / "credentials.json"
            credentials.write_text(
                json.dumps({"default": {"tokenStore": "keychain"}}),
                encoding="utf-8",
            )
            result = subprocess.CompletedProcess(
                args=["security"],
                returncode=0,
                stdout=b"secret-token\n",
                stderr=b"",
            )
            with patch(
                "tool.sync_freebuff_key.subprocess.run",
                return_value=result,
            ) as run:
                token = read_cli_token(credentials, "/usr/bin/security")

        self.assertEqual(token, "secret-token")
        self.assertEqual(
            run.call_args.args[0],
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                "freebuff-cli",
                "-a",
                str(credentials),
                "-w",
            ],
        )

    def test_skips_credentials_that_are_not_in_the_keychain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            credentials = Path(directory) / "credentials.json"
            credentials.write_text(
                json.dumps({"default": {"authToken": "file-backed-token"}}),
                encoding="utf-8",
            )
            with patch("tool.sync_freebuff_key.subprocess.run") as run:
                token = read_cli_token(credentials)

        self.assertIsNone(token)
        run.assert_not_called()

    def test_syncs_new_token_once_and_persists_only_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "sync-state.json"
            with patch("tool.sync_freebuff_key._add_token", return_value=True) as add:
                first = sync_token("secret-token", state_path=state)
                second = sync_token("secret-token", state_path=state)

            saved_state = json.loads(state.read_text(encoding="utf-8"))
            state_text = state.read_text(encoding="utf-8")

        self.assertEqual(first, "Imported a new Freebuff CLI token into Capyy.")
        self.assertEqual(second, "Freebuff CLI token is already synced.")
        add.assert_called_once()
        self.assertEqual(
            saved_state["token_sha256"],
            hashlib.sha256(b"secret-token").hexdigest(),
        )
        self.assertNotIn("secret-token", state_text)

    def test_marks_duplicate_token_as_synced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "sync-state.json"
            with patch("tool.sync_freebuff_key._add_token", return_value=False):
                message = sync_token("secret-token", state_path=state)

        self.assertEqual(message, "Freebuff CLI token was already present in Capyy.")


if __name__ == "__main__":
    unittest.main()
