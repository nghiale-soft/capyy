import json
import os
import tempfile
import unittest
from pathlib import Path

from gateway.core.config import Settings
from gateway.services.chat_history import ChatHistoryService
from gateway.services.history_scan import scan_local_history


def _settings(**overrides) -> Settings:
    base = dict(
        codebuff_token="token",
        local_api_key=None,
        history_dir=tempfile.mkdtemp(prefix="scanhist-"),
    )
    base.update(overrides)
    return Settings(**base)


class HistoryScanTests(unittest.TestCase):
    def setUp(self):
        self._old_claude = os.environ.get("SCAN_CLAUDE_DIR")
        self._old_codex = os.environ.get("SCAN_CODEX_DIR")
        self.claude_root = Path(tempfile.mkdtemp(prefix="fake-claude-"))
        self.codex_root = Path(tempfile.mkdtemp(prefix="fake-codex-"))
        os.environ["SCAN_CLAUDE_DIR"] = str(self.claude_root)
        os.environ["SCAN_CODEX_DIR"] = str(self.codex_root)

    def tearDown(self):
        if self._old_claude is None:
            os.environ.pop("SCAN_CLAUDE_DIR", None)
        else:
            os.environ["SCAN_CLAUDE_DIR"] = self._old_claude
        if self._old_codex is None:
            os.environ.pop("SCAN_CODEX_DIR", None)
        else:
            os.environ["SCAN_CODEX_DIR"] = self._old_codex

    def _write_claude_session(self, slug, session_id, cwd, lines):
        folder = self.claude_root / slug
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{session_id}.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_scan_claude_and_codex(self):
        service = ChatHistoryService(_settings())
        now = "2026-08-01T00:00:00Z"
        self._write_claude_session(
            "-tmp-mydemo",
            "sess-1",
            "/tmp/mydemo",
            [
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": now,
                        "cwd": "/tmp/mydemo",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "claude hỏi"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": now,
                        "cwd": "/tmp/mydemo",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "claude đáp"}],
                        },
                    }
                ),
                json.dumps({"type": "file-history-snapshot", "timestamp": now}),
            ],
        )

        codex_dir = self.codex_root / "2026" / "08" / "01"
        codex_dir.mkdir(parents=True, exist_ok=True)
        (codex_dir / "rollout-x.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {"timestamp": now, "type": "session_meta", "payload": {"cwd": "/tmp/otherproj"}}
                    ),
                    json.dumps(
                        {
                            "timestamp": now,
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "text", "text": "codex hỏi"}],
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": now,
                            "type": "response_item",
                            "payload": {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "text", "text": "codex đáp"}],
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = scan_local_history(service)
        self.assertTrue(result["ok"])
        self.assertEqual(result["records_imported"], 4)
        self.assertEqual(len(service.recent("mydemo")), 2)
        self.assertEqual(len(service.recent("otherproj")), 2)

        # Idempotent: scan lần 2 không import thêm (dedupe theo session_id)
        result2 = scan_local_history(service)
        self.assertEqual(result2["records_imported"], 0)
        self.assertEqual(len(service.recent("mydemo")), 2)

        # Active clients append to this same session file. The next scan must
        # import only the new row rather than skipping the session forever.
        session_path = self.claude_root / "-tmp-mydemo" / "sess-1.jsonl"
        with session_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "type": "user", "timestamp": "2026-08-01T00:01:00Z",
                "cwd": "/tmp/mydemo",
                "message": {"role": "user", "content": [{"type": "text", "text": "claude hỏi tiếp"}]},
            }) + "\n")
        result3 = scan_local_history(service)
        self.assertEqual(result3["records_imported"], 1)
        self.assertEqual(len(service.recent("mydemo")), 3)

    def test_scan_records_have_meta_source(self):
        service = ChatHistoryService(_settings())
        now = "2026-08-01T00:00:00Z"
        self._write_claude_session(
            "-tmp-proj",
            "abc-123",
            "/tmp/proj",
            [
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": now,
                        "cwd": "/tmp/proj",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "xin chào"}],
                        },
                    }
                )
            ],
        )
        scan_local_history(service)
        rows = service.recent("proj")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["meta"]["source"], "claude")
        self.assertEqual(rows[0]["meta"]["session_id"], "claude:abc-123")
        # session_ids nhận diện đã import
        self.assertIn("claude:abc-123", service.session_ids("proj"))

    def test_unchanged_files_are_not_parsed_again(self):
        service = ChatHistoryService(_settings())
        now = "2026-08-01T00:00:00Z"
        self._write_claude_session(
            "-tmp-fast", "fast-1", "/tmp/fast",
            [json.dumps({"type": "user", "timestamp": now, "cwd": "/tmp/fast",
                        "message": {"role": "user", "content": [{"type": "text", "text": "one"}]}})],
        )
        scan_local_history(service)
        result = scan_local_history(service)
        source = next(s for s in result["sources"] if s["source"] == "claude_code")
        self.assertEqual(source["files_skipped"], 1)
        self.assertEqual(result["records_imported"], 0)

    def test_scan_skips_old_sessions(self):
        """Session ngoài cửa sổ 365 ngày không được import."""
        service = ChatHistoryService(_settings())
        old = "2020-01-01T00:00:00Z"
        self._write_claude_session(
            "-tmp-old",
            "old-1",
            "/tmp/old",
            [
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": old,
                        "cwd": "/tmp/old",
                        "message": {"role": "user", "content": [{"type": "text", "text": "cũ"}]},
                    }
                )
            ],
        )
        result = scan_local_history(service)
        self.assertEqual(result["records_imported"], 0)
        self.assertEqual(service.recent("old"), [])


if __name__ == "__main__":
    unittest.main()
