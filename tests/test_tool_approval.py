import asyncio
import tempfile
import unittest

from gateway.core.config import Settings
from gateway.services.tool_approval import DEFAULT_PERMISSIONS, ToolApprovalService


def _settings(**overrides) -> Settings:
    base = dict(
        codebuff_token="token",
        local_api_key=None,
        tool_permissions_file=tempfile.mkdtemp(prefix="tools-") + "/perms.json",
        tool_approval_timeout=2.0,
    )
    base.update(overrides)
    return Settings(**base)


class ToolApprovalTests(unittest.IsolatedAsyncioTestCase):
    def test_defaults_read_allow_write_bash_ask(self):
        service = ToolApprovalService(_settings())
        self.assertEqual(service.mode_for("read_file"), "allow")
        self.assertEqual(service.mode_for("write_file"), "ask")
        self.assertEqual(service.mode_for("bash"), "ask")
        # unknown tools default to allow
        self.assertEqual(service.mode_for("unknown_tool"), "allow")

    def test_set_permission_persists(self):
        tmp = tempfile.mkdtemp(prefix="tools-")
        service = ToolApprovalService(_settings(tool_permissions_file=tmp + "/perms.json"))
        service.set_permission("bash", "deny")
        service2 = ToolApprovalService(_settings(tool_permissions_file=tmp + "/perms.json"))
        self.assertEqual(service2.mode_for("bash"), "deny")

    def test_set_permission_rejects_bad_mode(self):
        service = ToolApprovalService(_settings())
        with self.assertRaises(ValueError):
            service.set_permission("bash", "maybe")

    def test_allow_returns_none(self):
        service = ToolApprovalService(_settings())
        result = asyncio.run(service.request("read_file", {"path": "a.py"}, "."))
        self.assertIsNone(result)

    def test_deny_returns_message(self):
        service = ToolApprovalService(_settings())
        service.set_permission("bash", "deny")
        result = asyncio.run(service.request("bash", {"command": "rm -rf /"}, "."))
        self.assertIsNotNone(result)
        self.assertIn("denied", result)

    def test_ask_approve_runs(self):
        async def scenario():
            service = ToolApprovalService(_settings())
            service.set_permission("bash", "ask")
            task = asyncio.create_task(
                service.request("bash", {"command": "ls"}, "/tmp")
            )
            await asyncio.sleep(0.05)
            pending = service.list_pending()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["tool"], "bash")
            self.assertEqual(pending[0]["summary"], "ls")
            ok = await service.decide(pending[0]["id"], "approved")
            self.assertTrue(ok)
            result = await task
            self.assertIsNone(result)
            self.assertEqual(service.list_pending(), [])

        asyncio.run(scenario())

    def test_ask_deny_blocks(self):
        async def scenario():
            service = ToolApprovalService(_settings())
            task = asyncio.create_task(
                service.request("write_file", {"path": "x.txt"}, "/tmp")
            )
            await asyncio.sleep(0.05)
            pending = service.list_pending()
            self.assertEqual(len(pending), 1)
            await service.decide(pending[0]["id"], "denied")
            result = await task
            self.assertIsNotNone(result)
            self.assertIn("denied", result)

        asyncio.run(scenario())

    def test_ask_timeout(self):
        service = ToolApprovalService(_settings(tool_approval_timeout=0.1))
        result = asyncio.run(
            service.request("write_file", {"path": "x.txt"}, "/tmp")
        )
        self.assertIsNotNone(result)
        self.assertIn("timed out", result)

    def test_decide_unknown_id_false(self):
        service = ToolApprovalService(_settings())
        resolved = asyncio.run(service.decide("nope", "approved"))
        self.assertFalse(resolved)

    def test_defaults_include_expected_keys(self):
        self.assertEqual(DEFAULT_PERMISSIONS["read_file"], "allow")
        self.assertEqual(DEFAULT_PERMISSIONS["write_file"], "ask")
        self.assertEqual(DEFAULT_PERMISSIONS["bash"], "ask")

    def test_write_file_summary_previews_content(self):
        """Approve write_file: summary hiển thị path + preview nội dung dòng đầu."""
        from gateway.services.tool_approval import _summarize_args
        self.assertEqual(
            _summarize_args("figma_get_node", {"file_key": "abc123", "node_id": "0:42"}),
            "file abc123 · node 0:42",
        )
        self.assertEqual(
            _summarize_args("browser_open", {"url": "https://example.com"}),
            "https://example.com",
        )
        self.assertEqual(
            _summarize_args("bash", {"command": "npm test"}),
            "npm test",
        )

        summary = _summarize_args(
            "write_file",
            {"path": "src/app.py", "content": "import os\nprint('hi')"},
        )
        self.assertIn("src/app.py", summary)
        self.assertIn("import os", summary)
        self.assertNotIn("print('hi')", summary)


if __name__ == "__main__":
    unittest.main()
