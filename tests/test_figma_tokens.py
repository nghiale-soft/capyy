from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gateway.services.figma import FigmaTokenStore


class FigmaTokenStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "figma-tokens.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def store(self) -> FigmaTokenStore:
        return FigmaTokenStore(self.path)

    def test_empty_store_returns_empty(self) -> None:
        store = self.store()
        self.assertEqual(store.token_for("any-project"), "")
        self.assertEqual(store.status()["tokens"], [])
        self.assertFalse(store.status()["configured"])

    def test_add_and_status_mask_reveal(self) -> None:
        store = self.store()
        store.add_token("figd_default_123456789")
        store.add_token("figd_project_a_987654321")
        status = store.status()
        self.assertEqual(len(status["tokens"]), 2)
        self.assertIn("*", status["tokens"][0]["masked"])
        self.assertNotIn("value", status["tokens"][0])
        revealed = store.status(reveal=True)
        self.assertEqual(revealed["tokens"][0]["value"], "figd_default_123456789")
        self.assertEqual(revealed["tokens"][1]["value"], "figd_project_a_987654321")
        self.assertEqual(revealed["fallback"], [])

    def test_fallback_reported_separately_not_editable(self) -> None:
        import os
        os.environ["FIGMA_TOKEN"] = "figd_env_fallback_999"
        try:
            store = self.store()
            store.add_token("figd_file_token_111")
            status = store.status(reveal=True)
            # Only the editable file token has an index/remove button.
            self.assertEqual([t["value"] for t in status["tokens"]], ["figd_file_token_111"])
            self.assertEqual([f["value"] for f in status["fallback"]], ["figd_env_fallback_999"])
            # token_for still round-robins over both.
            self.assertIn(store.token_for(), ("figd_file_token_111", "figd_env_fallback_999"))
        finally:
            os.environ.pop("FIGMA_TOKEN", None)

    def test_round_robin_picks_all_tokens(self) -> None:
        store = self.store()
        store.add_token("figd_a")
        store.add_token("figd_b")
        picked = {store.token_for() for _ in range(6)}
        self.assertEqual(picked, {"figd_a", "figd_b"})

    def test_remove_token_by_index(self) -> None:
        store = self.store()
        store.add_token("figd_a")
        store.add_token("figd_b")
        tokens = store.remove_token(0)
        self.assertEqual(tokens, ["figd_b"])
        self.assertEqual(store.token_for(), "figd_b")

    def test_remove_token_out_of_range(self) -> None:
        store = self.store()
        store.add_token("figd_a")
        with self.assertRaises(IndexError):
            store.remove_token(5)

    def test_replace_tokens(self) -> None:
        store = self.store()
        store.replace_tokens(["figd_x", "figd_y"])
        self.assertEqual(store.token_for() in ("figd_x", "figd_y"), True)
        store.replace_tokens(["figd_z"])
        self.assertEqual(store.token_for(), "figd_z")

    def test_add_duplicate_raises(self) -> None:
        store = self.store()
        store.add_token("figd_same")
        with self.assertRaises(ValueError):
            store.add_token("figd_same")

    def test_legacy_file_fallback(self) -> None:
        legacy = Path(self.tmp.name) / "figma-token.json"
        legacy.write_text(json.dumps({"token": "figd_legacy_42"}), encoding="utf-8")
        import gateway.services.figma as figma_mod

        original = figma_mod.LEGACY_TOKEN_FILE
        figma_mod.LEGACY_TOKEN_FILE = str(legacy)
        try:
            self.assertEqual(self.store().token_for("x"), "figd_legacy_42")
        finally:
            figma_mod.LEGACY_TOKEN_FILE = original

    def test_clear_all(self) -> None:
        store = self.store()
        store.add_token("figd_a")
        store.clear_all()
        self.assertEqual(store.status()["tokens"], [])

    def test_backwards_compat_default_shape(self) -> None:
        self.path.write_text(json.dumps({"default": "figd_old"}), encoding="utf-8")
        self.assertEqual(self.store().token_for(), "figd_old")


if __name__ == "__main__":
    unittest.main()
