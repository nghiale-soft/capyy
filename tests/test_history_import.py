import json
import unittest

from gateway.services.history_import import claude_code_session, codex_session


class ClaudeSessionParserTests(unittest.TestCase):
    def test_claude_code_jsonl(self):
        sample = "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": "2025-01-02T03:04:05Z",
                        "cwd": "/home/me/proj",
                        "message": {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "hỏi gì đó"},
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t1",
                                    "content": "kết quả",
                                },
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2025-01-02T03:04:06Z",
                        "cwd": "/home/me/proj",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "thinking", "thinking": "nghĩ"},
                                {"type": "text", "text": "trả lời"},
                                {
                                    "type": "tool_use",
                                    "id": "t1",
                                    "name": "read_file",
                                    "input": {"path": "a.py"},
                                },
                            ],
                        },
                    }
                ),
                json.dumps({"type": "summary", "summary": "tóm tắt"}),
            ]
        )
        records, cwd = claude_code_session(sample)
        self.assertEqual(cwd, "/home/me/proj")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["role"], "user")
        self.assertIn("[tool_result] kết quả", records[0]["content"])
        self.assertEqual(records[1]["thinking"], "nghĩ")
        self.assertEqual(records[1]["tool_calls"][0]["function"]["name"], "read_file")
        arguments = records[1]["tool_calls"][0]["function"]["arguments"]
        self.assertIn('"path"', arguments)
        self.assertIn("a.py", arguments)
        # timestamp ISO -> epoch ms
        self.assertTrue(records[0]["ts"] and records[0]["ts"] > 1700000000000)

    def test_claude_thinking_only_assistant_kept(self):
        sample = json.dumps(
            {
                "type": "assistant",
                "timestamp": "2025-01-02T03:04:06Z",
                "cwd": "/tmp/p",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "chỉ suy nghĩ"}],
                },
            }
        )
        records, _ = claude_code_session(sample)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["thinking"], "chỉ suy nghĩ")
        self.assertEqual(records[0]["content"], "")


class CodexSessionParserTests(unittest.TestCase):
    def test_codex_jsonl_session(self):
        now = "2026-08-01T00:00:00Z"
        lines = [
            json.dumps({"timestamp": now, "type": "session_meta", "payload": {"cwd": "/tmp/otherproj"}}),
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
        records, cwd = codex_session("\n".join(lines))
        self.assertEqual(cwd, "/tmp/otherproj")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["content"], "codex hỏi")
        self.assertEqual(records[1]["content"], "codex đáp")

    def test_codex_message_level_tool_calls(self):
        """Codex để tool_calls ở message-level — vẫn parse được."""
        now = "2026-08-01T00:00:00Z"
        lines = [
            json.dumps(
                {
                    "timestamp": now,
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "tôi sẽ kiểm tra"}],
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                            }
                        ],
                    },
                }
            )
        ]
        records, _ = codex_session("\n".join(lines))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["tool_calls"][0]["function"]["name"], "read_file")


if __name__ == "__main__":
    unittest.main()
