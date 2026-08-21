import unittest

from gateway.routes import freebuff, history


def _route_signatures(router) -> set[tuple[str, str]]:
    return {
        (method.upper(), route.path)
        for route in router.routes
        for method in getattr(route, "methods", []) or []
    }


class HistoryRoutesTests(unittest.TestCase):
    def test_manual_import_route_removed(self) -> None:
        """POST /api/history/import must not be registered anymore."""
        signatures = _route_signatures(history.router)
        self.assertNotIn(("POST", "/api/history/import"), signatures)
        # Scan / list / get / delete still present
        self.assertIn(("POST", "/api/history/scan"), signatures)
        self.assertIn(("GET", "/api/history"), signatures)
        self.assertIn(("GET", "/api/history/{project_key}"), signatures)
        self.assertIn(("DELETE", "/api/history/{project_key}"), signatures)

    def test_manual_import_not_importable(self) -> None:
        """The import_history/convert helpers are gone from history_import."""
        import gateway.services.history_import as module

        self.assertFalse(hasattr(module, "import_history"))
        self.assertFalse(hasattr(module, "convert"))
        # Session parsers used by the scan feature remain
        self.assertTrue(hasattr(module, "claude_code_session"))
        self.assertTrue(hasattr(module, "codex_session"))


class FreebuffTokenRoutesTests(unittest.TestCase):
    def test_token_routes_registered(self) -> None:
        signatures = _route_signatures(freebuff.router)
        self.assertIn(("GET", "/api/freebuff/tokens"), signatures)
        self.assertIn(("POST", "/api/freebuff/tokens"), signatures)
        self.assertIn(("PUT", "/api/freebuff/tokens"), signatures)
        self.assertIn(("DELETE", "/api/freebuff/tokens"), signatures)
        self.assertIn(("DELETE", "/api/freebuff/tokens/{index}"), signatures)

    def test_mask_token_edge_cases(self) -> None:
        from gateway.routes.freebuff import _mask_token

        # Long token: first 3 + stars + last 4, never leaks the middle
        masked = _mask_token("abcdefghij1234")
        self.assertIn("abc", masked)
        self.assertIn("1234", masked)
        self.assertNotIn("fghij", masked)
        # Short token never longer than the original
        self.assertEqual(len(_mask_token("shorttoken")), len("shorttoken") + 5)
        tiny = _mask_token("ab")
        self.assertEqual(tiny, "**")
        eight = _mask_token("12345678")
        self.assertIn("5678", eight)
        self.assertNotIn("1234", eight)

    def test_get_tokens_hides_values_by_default_reveals_on_demand(self) -> None:
        import asyncio

        from gateway.routes.freebuff import get_tokens

        class _FakeAccounts:
            token_source = "file"
            account_count = 1
            tokens = ["sk-secret-token-1234"]

            def token_statuses(self):
                return [{
                    "index": 0,
                    "status": "busy",
                    "is_default": True,
                    "retry_at": None,
                    "last_error_status": None,
                }]

        class _FakeRequest:
            app = type(
                "App",
                (),
                {"state": type("State", (), {"accounts": _FakeAccounts()})()},
            )()

        async def _run():
            hidden = await get_tokens(_FakeRequest(), _=None)
            revealed = await get_tokens(_FakeRequest(), reveal=1, _=None)
            return hidden, revealed

        hidden, revealed = asyncio.run(_run())
        self.assertEqual(hidden["configured"], True)
        self.assertNotIn("value", hidden["tokens"][0])
        self.assertEqual(hidden["tokens"][0]["status"], "busy")
        self.assertTrue(hidden["tokens"][0]["is_default"])
        self.assertEqual(revealed["tokens"][0]["value"], "sk-secret-token-1234")


if __name__ == "__main__":
    unittest.main()
