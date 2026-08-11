import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from gateway import webapp


class _FakeResponse:
    def __init__(self, status_code=200, content=b'{"ok": true}', headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"content-type": "application/json"}


class _FakeClient:
    """Async context manager mô phỏng httpx.AsyncClient, ghi lại request."""

    def __init__(self, *args, **kwargs):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _FakeResponse()


class WebappTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(webapp.app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)

    def test_index_serves_dashboard_html(self) -> None:
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("/static/style.css", resp.text)
        self.assertIn("/static/dashboard.css", resp.text)
        # New menu structure: Providers / Chat history on top, Settings / About below
        self.assertIn('id="providersNav"', resp.text)
        self.assertIn('id="historyNav"', resp.text)
        self.assertIn('id="settingsNav"', resp.text)
        self.assertIn('id="aboutNav"', resp.text)
        # Provider form: source select + command select + single model + fetch + priority
        self.assertIn('id="f_source"', resp.text)
        self.assertIn('id="f_command"', resp.text)
        self.assertIn('id="f_model"', resp.text)
        self.assertIn('id="fetchModelsBtn"', resp.text)
        self.assertIn('id="priorityBtn"', resp.text)
        self.assertNotIn('id="f_default"', resp.text)
        self.assertIn('figmaTokenSecrets = Array.isArray(data.tokens)', resp.text)
        # Tokens now live inside the provider form, not a separate menu item
        self.assertNotIn('id="tokensNav"', resp.text)
        self.assertNotIn('id="tokensSection"', resp.text)
        self.assertNotIn('id="toolPermsList"', resp.text)
        self.assertNotIn('id="settingsPendingBadge"', resp.text)
        self.assertIn('<span class="f-label">Author</span>', resp.text)
        self.assertIn('<span class="f-label">Donate</span>', resp.text)

    def test_index_sets_no_cache_headers(self) -> None:
        resp = self.client.get("/")
        self.assertIn("no-store", resp.headers.get("cache-control", ""))

    def test_static_css_is_served(self) -> None:
        resp = self.client.get("/static/dashboard.css")
        self.assertEqual(resp.status_code, 200)
        # Keelor-style palette token
        self.assertIn("--brand", resp.text)

    def test_proxy_forwards_request_to_gateway(self) -> None:
        fake = _FakeClient()
        with patch("gateway.webapp.httpx.AsyncClient", return_value=fake):
            resp = self.client.get(
                "/api/freebuff/tokens",
                headers={"Authorization": "Bearer abc123", "X-Custom": "yes"},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})
        method, url, kwargs = fake.calls[0]
        self.assertEqual(method, "GET")
        self.assertTrue(url.endswith("/api/freebuff/tokens"))
        # host/content-length bị bỏ; auth + custom header được forward
        headers = kwargs["headers"]
        lowered = {k.lower(): v for k, v in headers.items()}
        self.assertEqual(lowered["authorization"], "Bearer abc123")
        self.assertEqual(lowered["x-custom"], "yes")
        self.assertNotIn("host", lowered)
        self.assertNotIn("content-length", lowered)

    def test_proxy_forwards_query_params(self) -> None:
        fake = _FakeClient()
        with patch("gateway.webapp.httpx.AsyncClient", return_value=fake):
            resp = self.client.get("/api/freebuff/tokens?reveal=1")

        self.assertEqual(resp.status_code, 200)
        method, url, kwargs = fake.calls[0]
        self.assertEqual(method, "GET")
        # Query string reaches the upstream gateway (needed for token reveal).
        self.assertEqual(kwargs["params"], {"reveal": "1"})

    def test_proxy_passes_through_put_body(self) -> None:
        fake = _FakeClient()
        with patch("gateway.webapp.httpx.AsyncClient", return_value=fake):
            resp = self.client.put(
                "/api/freebuff/tokens",
                json={"tokens": ["a", "b"]},
            )

        self.assertEqual(resp.status_code, 200)
        method, url, kwargs = fake.calls[0]
        self.assertEqual(method, "PUT")
        self.assertIn(b'"tokens"', kwargs["content"])

    def test_proxy_returns_502_when_gateway_unreachable(self) -> None:
        class _DownClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def request(self, *args, **kwargs):
                import httpx

                raise httpx.ConnectError("connection refused")

        with patch("gateway.webapp.httpx.AsyncClient", return_value=_DownClient()):
            resp = self.client.get("/api/freebuff/tokens")

        self.assertEqual(resp.status_code, 502)


if __name__ == "__main__":
    unittest.main()
