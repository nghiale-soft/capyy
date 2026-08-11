import unittest

from gateway.core.config import Settings
from providers.freebuff import CodebuffClient, CodebuffError


class FreebuffTokenSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_dashboard_token_has_actionable_message(self) -> None:
        client = CodebuffClient(Settings(codebuff_token=None, local_api_key=None))
        try:
            with self.assertRaises(CodebuffError) as context:
                client._headers()
        finally:
            await client.aclose()

        self.assertEqual(context.exception.status_code, 503)
        self.assertIn("Dashboard", str(context.exception))
        self.assertNotIn("FREEBUFF_TOKEN", str(context.exception))
