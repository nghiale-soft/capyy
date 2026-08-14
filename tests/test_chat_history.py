import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from gateway.services.chat_history import ChatHistoryService
from gateway.core.config import Settings


def _settings(**overrides) -> Settings:
    base = dict(
        codebuff_token="token",
        local_api_key=None,
        history_dir=tempfile.mkdtemp(prefix="chathist-"),
    )
    base.update(overrides)
    return Settings(**base)


class FakeRequest:
    def __init__(self, headers=None):
        self._headers = headers or {}

    @property
    def headers(self):
        return self._headers


class ChatHistoryTests(unittest.TestCase):
    def test_record_and_recent_roundtrip(self):
        service = ChatHistoryService(_settings())
        service.record("proj-a", role="user", content="xin chào")
        service.record("proj-a", role="assistant", content="chào bạn")

        rows = service.recent("proj-a")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["role"], "user")
        self.assertEqual(rows[0]["content"], "xin chào")
        self.assertEqual(rows[1]["role"], "assistant")
        self.assertEqual(service._chat_file("proj-a").name, "gateway.jsonl")
        self.assertIn("projects/proj-a/sessions", str(service._chat_file("proj-a")))

    def test_record_messages_only_records_last_user(self):
        """Client gửi cả lịch sử mỗi request -> chỉ ghi message user cuối."""
        service = ChatHistoryService(_settings())
        service.record_messages(
            "p",
            [
                {"role": "system", "content": "bỏ qua"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "tool", "content": "bỏ qua"},
            ],
        )
        rows = service.recent("p")
        self.assertEqual([r["role"] for r in rows], ["user"])
        self.assertEqual(rows[0]["content"], "u2")

    def test_record_messages_skips_duplicate_retry(self):
        """Request retry cùng câu hỏi -> không ghi trùng lặp."""
        service = ChatHistoryService(_settings())
        payload = [{"role": "user", "content": "bạn có nhớ lỗi 429 không?"}]
        service.record_messages("p", payload)
        service.record_messages("p", payload)
        rows = service.recent("p")
        self.assertEqual(len(rows), 1)

    def test_current_session_context_is_isolated(self):
        service = ChatHistoryService(_settings())
        service.record("p", role="user", content="alpha private", meta={"session_id": "alpha"})
        service.record("p", role="assistant", content="alpha answer", meta={"session_id": "alpha"})
        service.record("p", role="user", content="beta private", meta={"session_id": "beta"})

        ctx = service.build_context("p", "tiếp tục", session_id="beta")
        self.assertIn("beta private", ctx)
        self.assertNotIn("alpha private", ctx)
        self.assertTrue(service._chat_file("p", "alpha").exists())
        self.assertTrue(service._chat_file("p", "beta").exists())

    def test_build_context_with_messages_list_triggers(self):
        """Routes truyền body.messages (list dict) -> vẫn nhận diện memory."""
        service = ChatHistoryService(_settings())
        service.record("p", role="user", content="tôi hỏi về lỗi 429 hôm qua")
        service.record("p", role="assistant", content="429 là rate limit")

        messages = [
            {"role": "system", "content": "buffy"},
            {"role": "user", "content": "bạn có nhớ lỗi 429 lần trước không?"},
        ]
        ctx = service.build_context("p", messages)
        self.assertIsNotNone(ctx)
        self.assertIn("429", ctx)

    def test_build_context_with_anthropic_blocks_list_triggers(self):
        """Anthropic content dạng list block vẫn nhận diện memory."""
        service = ChatHistoryService(_settings())
        service.record("p", role="user", content="trao đổi về 428 hôm trước")
        service.record("p", role="assistant", content="428 là waiting room")

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "bạn có nhớ vụ 428 lần trước không?"}
            ]},
        ]
        ctx = service.build_context("p", messages)
        self.assertIsNotNone(ctx)
        self.assertIn("428", ctx)

    def test_prune_drops_old_rows(self):
        service = ChatHistoryService(_settings(history_max_age_days=1))
        old = int(time.time() * 1000) - 2 * 86_400_000
        path = service._chat_file("p")
        path.write_text(
            "\n".join(
                [
                    json.dumps({"ts": old, "role": "user", "content": "cũ"}),
                    json.dumps(
                        {"ts": int(time.time() * 1000), "role": "user", "content": "mới"}
                    ),
                ]
            ),
            encoding="utf-8",
        )
        service.prune("p")
        rows = service.recent("p")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "mới")

    def test_resolve_project_prefers_header(self):
        service = ChatHistoryService(_settings())
        key = service.resolve_project(
            FakeRequest({"x-project-path": "/home/me/code/mydemo"}),
            {},
        )
        # không phải git repo -> key = tên folder
        self.assertEqual(key, "mydemo")

    def test_resolve_project_git_remote_is_stable_across_rename(self):
        service = ChatHistoryService(_settings())
        tmp = Path(tempfile.mkdtemp(prefix="gitproj-"))
        git = tmp / ".git"
        git.mkdir()
        (git / "config").write_text(
            '[remote "origin"]\n\turl = https://github.com/me/stuff.git\n',
            encoding="utf-8",
        )

        key1 = service.resolve_project(FakeRequest({"x-project-path": str(tmp)}), {})
        self.assertEqual(key1, "https_github.com_me_stuff.git")

        # Đổi tên folder (vẫn git remote cũ) -> key giữ nguyên
        renamed = tmp.parent / f"{tmp.name}-renamed"
        tmp.rename(renamed)
        key2 = service.resolve_project(
            FakeRequest({"x-project-path": str(renamed)}),
            {},
        )
        self.assertEqual(key2, key1)

        # Index lưu cả 2 path
        projects = service._load_projects()
        self.assertEqual(len(projects[key1]["paths"]), 2)

    def test_resolve_project_folder_move_reuses_key(self):
        service = ChatHistoryService(_settings())
        key1 = service.resolve_project(
            FakeRequest({"x-project-path": "/a/b/demo"}), {}
        )
        key2 = service.resolve_project(
            FakeRequest({"x-project-path": "/x/y/demo"}), {}
        )
        self.assertEqual(key1, key2)

    def test_build_context_memory_question_triggers(self):
        service = ChatHistoryService(_settings())
        service.record("p", role="user", content="tôi hỏi về lỗi 429 hôm qua")
        service.record("p", role="assistant", content="429 là rate limit")

        ctx = service.build_context("p", "bạn có nhớ lỗi 429 lần trước không?")
        self.assertIsNotNone(ctx)
        self.assertIn("429", ctx)

    def test_build_context_keeps_current_session_in_memory_only(self):
        service = ChatHistoryService(_settings())
        service.record("p", role="user", content="giải thích code này")
        service.record("p", role="assistant", content="code này dùng để ...")

        ctx = service.build_context("p", "giải thích thêm đi")
        self.assertIsNotNone(ctx)

    def test_build_context_always_mode_injects(self):
        service = ChatHistoryService(
            _settings(history_inject_mode="always")
        )
        service.record("p", role="user", content="nói chuyện gì đó")
        service.record("p", role="assistant", content="phản hồi gì đó")

        ctx = service.build_context("p", "tiếp tục")
        self.assertIsNotNone(ctx)

    def test_build_context_off_mode_returns_none(self):
        service = ChatHistoryService(_settings(history_inject_mode="off"))
        service.record("p", role="user", content="abc")
        self.assertIsNone(
            service.build_context("p", "bạn có nhớ gì không?")
        )

    def test_record_with_thinking_and_tool_calls(self):
        service = ChatHistoryService(_settings())
        service.record(
            "p",
            role="assistant",
            content="tôi sẽ đọc file",
            thinking="người dùng muốn đọc file",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "a.py"}',
                    },
                }
            ],
        )
        rows = service.recent("p")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["thinking"], "người dùng muốn đọc file")
        self.assertEqual(rows[0]["tool_calls"][0]["function"]["name"], "read_file")

    def test_record_tool_only_assistant_and_empty_skip(self):
        service = ChatHistoryService(_settings())
        # assistant chỉ gọi tool (content rỗng) vẫn được lưu
        service.record(
            "p",
            role="assistant",
            content="",
            tool_calls=[
                {"id": "x", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            ],
        )
        rows = service.recent("p")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "")
        self.assertEqual(rows[0]["tool_calls"][0]["function"]["name"], "bash")
        # content rỗng + không tool -> bỏ qua
        service.record("p", role="user", content=None)
        self.assertEqual(len(service.recent("p")), 1)

    def test_record_thinking_only_kept(self):
        """Record assistant chỉ có thinking (content rỗng) vẫn được giữ."""
        service = ChatHistoryService(_settings())
        service.record("p", role="assistant", content="", thinking="phân tích trước")
        rows = service.recent("p")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["thinking"], "phân tích trước")
        self.assertEqual(rows[0]["content"], "")

    def test_append_records_thinking_only_kept(self):
        """Import record thinking-only không bị bỏ qua."""
        service = ChatHistoryService(_settings())
        now = int(time.time() * 1000)
        written = service.append_records(
            "p",
            [
                {"ts": now, "role": "assistant", "content": "", "thinking": "nghĩ gì đó"},
                {"ts": now, "role": "assistant", "content": "trả lời"},
            ],
        )
        self.assertEqual(written, 2)
        rows = service.recent("p")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["thinking"], "nghĩ gì đó")

    def test_record_tool_chain_not_deduped(self):
        """Hai assistant turn chỉ gọi tool (content rỗng) không bị dedupe."""
        service = ChatHistoryService(_settings())
        service.record(
            "p",
            role="assistant",
            content="",
            tool_calls=[{"type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        )
        service.record(
            "p",
            role="assistant",
            content="",
            tool_calls=[{"type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        )
        rows = service.recent("p")
        self.assertEqual(len(rows), 2)
        names = [r["tool_calls"][0]["function"]["name"] for r in rows]
        self.assertEqual(names, ["read_file", "bash"])

    def test_content_to_text_tool_result(self):
        from gateway.services.chat_history import _content_to_text

        text = _content_to_text(
            [
                {"type": "tool_result", "tool_use_id": "t1", "content": "OK"},
                {"type": "text", "text": "tiếp tục"},
            ]
        )
        self.assertIn("[tool_result] OK", text)
        self.assertIn("tiếp tục", text)

        err = _content_to_text(
            [{"type": "tool_result", "tool_use_id": "t2", "content": "boom", "is_error": True}]
        )
        self.assertIn("[tool error]", err)

    def test_list_projects_and_delete(self):
        service = ChatHistoryService(_settings())
        service.record("proj-a", role="user", content="hello")
        service.record("proj-a", role="assistant", content="hi")
        service.record("proj-b", role="user", content="x")

        projects = service.list_projects()
        by_key = {p["key"]: p for p in projects}
        self.assertIn("proj-a", by_key)
        self.assertIn("proj-b", by_key)
        self.assertEqual(by_key["proj-a"]["count"], 2)
        self.assertEqual(by_key["proj-b"]["count"], 1)

        service.delete_project("proj-a")
        keys = [p["key"] for p in service.list_projects()]
        self.assertNotIn("proj-a", keys)
        self.assertFalse(service._chat_file("proj-a").exists())

    def test_messages_pagination(self):
        service = ChatHistoryService(_settings())
        for i in range(5):
            service.record("p", role="user", content=f"msg-{i}")
        rows = service.messages("p", limit=2, offset=2)
        self.assertEqual([r["content"] for r in rows], ["msg-2", "msg-3"])
        self.assertEqual(len(service.messages("p")), 5)
        latest = service.messages("p", limit=2, newest_first_page=True)
        self.assertEqual([r["content"] for r in latest], ["msg-3", "msg-4"])

    def test_newest_page_uses_timestamps_not_import_append_order(self):
        service = ChatHistoryService(_settings())
        now = int(time.time() * 1000)
        service.append_records("p", [
            {"ts": now + 20, "role": "assistant", "content": "new"},
            {"ts": now + 10, "role": "user", "content": "middle"},
        ])
        # A later import may append an older session after the newest row.
        service.append_records("p", [{"ts": now, "role": "user", "content": "old"}])
        rows = service.messages("p", limit=2, newest_first_page=True)
        self.assertEqual([r["content"] for r in rows], ["middle", "new"])

    def test_append_records_import(self):
        service = ChatHistoryService(_settings())
        now = int(time.time() * 1000)
        written = service.append_records(
            "p",
            [
                {"ts": now - 1000, "role": "user", "content": "u1"},
                {
                    "ts": now,
                    "role": "assistant",
                    "content": "a1",
                    "thinking": "think",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "bash", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "content": "skip me"},
            ],
        )
        self.assertEqual(written, 2)
        rows = service.recent("p")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["ts"], now - 1000)
        self.assertEqual(rows[1]["thinking"], "think")
        self.assertEqual(rows[1]["tool_calls"][0]["function"]["name"], "bash")

    def test_content_to_text_lists(self):
        from gateway.services.chat_history import _content_to_text

        self.assertEqual(
            _content_to_text(
                [{"type": "text", "text": "a"}, {"type": "image", "text": "b"}]
            ),
            "a\nb",
        )
        self.assertEqual(
            _content_to_text([{"type": "image", "source": {"type": "base64"}}]),
            "",
        )
        self.assertEqual(_content_to_text(None), "")
        self.assertEqual(_content_to_text("plain"), "plain")
        # Message dict OpenAI/Anthropic
        self.assertEqual(_content_to_text({"role": "user", "content": "x"}), "x")
        self.assertEqual(
            _content_to_text({"role": "user", "content": [{"type": "text", "text": "y"}]}),
            "y",
        )

    def test_inject_context_system_not_first_keeps_messages(self):
        """System message ở vị trí khác 0 -> không drop message đầu."""
        from gateway.services.chat_history import inject_context

        messages = [
            {"role": "user", "content": "giữ tôi"},
            {"role": "system", "content": "identity"},
            {"role": "user", "content": "hỏi"},
        ]
        result = inject_context(messages, "LỊCH SỬ")
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["content"], "giữ tôi")
        self.assertIn("LỊCH SỬ", result[1]["content"])

    def test_record_meta_source_gateway(self):
        """Record qua gateway gắn meta.source = gateway."""
        service = ChatHistoryService(_settings())
        service.record(
            "p",
            role="assistant",
            content="trả lời",
            meta={"source": "gateway"},
        )
        rows = service.recent("p")
        self.assertEqual(rows[0]["meta"]["source"], "gateway")

    def test_list_projects_includes_sources(self):
        """list_projects gom nguồn AI của từng project."""
        service = ChatHistoryService(_settings())
        service.record("p", role="user", content="u", meta={"source": "gateway"})
        service.record("p", role="assistant", content="a", meta={"source": "claude"})
        projects = {p["key"]: p for p in service.list_projects()}
        self.assertEqual(sorted(projects["p"]["sources"]), ["claude", "gateway"])

    def test_list_projects_last_user_preview(self):
        """list_projects trả preview câu user cuối (cho WhatsApp-style list)."""
        service = ChatHistoryService(_settings())
        service.record("p", role="user", content="câu hỏi đầu")
        service.record("p", role="assistant", content="trả lời 1")
        service.record("p", role="user", content="câu hỏi mới nhất")
        service.record("p", role="assistant", content="trả lời cuối")
        projects = {p["key"]: p for p in service.list_projects()}
        self.assertEqual(projects["p"]["last_user_content"], "câu hỏi mới nhất")
        self.assertGreater(projects["p"]["last_user_ts"], 0)
        # last_content vẫn là message mới nhất (bất kỳ role)
        self.assertEqual(projects["p"]["last_content"], "trả lời cuối")

    def test_list_projects_auto_title_from_first_user_message(self):
        """Tên đoạn chat được sinh từ câu user đầu tiên (không phải tên folder)."""
        service = ChatHistoryService(_settings())
        service.record("p", role="user", content="câu hỏi đầu tiên về lỗi 429")
        service.record("p", role="assistant", content="trả lời 1")
        service.record("p", role="user", content="câu hỏi mới nhất")
        projects = {p["key"]: p for p in service.list_projects()}
        self.assertEqual(projects["p"]["title"], "câu hỏi đầu tiên về lỗi 429")

    def test_list_projects_title_strips_ai_wrappers(self):
        """Title/preview loại bỏ <system-reminder>, <ide_opened_file>... để hiện câu hỏi thật."""
        service = ChatHistoryService(_settings())
        service.record(
            "p",
            role="user",
            content=(
                "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# currentDate\n</system-reminder>\n\n"
                "xin chào bạn\n\n"
                "<ide_opened_file>The user opened the file /a/b/c.py in the IDE.</ide_opened_file>"
            ),
        )
        service.record("p", role="assistant", content="chào bạn")
        projects = {p["key"]: p for p in service.list_projects()}
        self.assertEqual(projects["p"]["title"], "xin chào bạn")
        self.assertEqual(projects["p"]["last_user_content"], "xin chào bạn")

    def test_list_projects_title_strips_tool_result_prefix(self):
        """Tin nhắn user bắt đầu bằng [tool_result] -> title bỏ phần machine-generated."""
        service = ChatHistoryService(_settings())
        service.record(
            "p",
            role="user",
            content="[tool_result] The file /x/main.py contains: import logging",
        )
        projects = {p["key"]: p for p in service.list_projects()}
        self.assertEqual(projects["p"]["title"], "p")
        self.assertEqual(projects["p"]["last_user_content"], "")

    def test_list_projects_title_falls_back_to_folder(self):
        """Không có user message -> title rơi về tên folder."""
        service = ChatHistoryService(_settings())
        service.record("p", role="assistant", content="chỉ có assistant")
        projects = {p["key"]: p for p in service.list_projects()}
        self.assertEqual(projects["p"]["title"], "p")

    def test_list_projects_first_ts_tracks_first_user(self):
        """first_ts = mốc tin user đầu tiên (dùng cho filter thời gian)."""
        service = ChatHistoryService(_settings())
        service.record("p", role="user", content="u1")
        service.record("p", role="assistant", content="a1")
        service.record("p", role="user", content="u2")
        projects = {p["key"]: p for p in service.list_projects()}
        self.assertGreater(projects["p"]["first_ts"], 0)
        self.assertLessEqual(projects["p"]["first_ts"], projects["p"]["last_user_ts"])

    def test_list_projects_includes_provider_labels(self):
        """Gateway records gắn meta.provider (tên provider) để UI hiện đủ nguồn."""
        service = ChatHistoryService(_settings())
        service.record("p", role="user", content="u", meta={"source": "gateway", "provider": "FreeBuff"})
        service.record("p", role="assistant", content="a", meta={"source": "gateway", "provider": "My FreeBuff"})
        service.record("p", role="assistant", content="b", meta={"source": "claude"})
        projects = {p["key"]: p for p in service.list_projects()}
        self.assertEqual(sorted(projects["p"]["sources"]), ["claude", "gateway"])
        self.assertEqual(sorted(projects["p"]["providers"]), ["FreeBuff", "My FreeBuff"])

    def test_resolve_project_reuses_known_path(self):
        """Path đã từng lưu của project khác -> gộp về project đó."""
        service = ChatHistoryService(_settings())
        key1 = service.resolve_project(
            FakeRequest({"x-project-path": "/a/b/proj"}), {}
        )
        # Một project khác (tên khác) gửi lại đúng path cũ -> gộp về key1
        key2 = service.resolve_project(
            FakeRequest({"x-project-path": "/a/b/proj"}), {}
        )
        self.assertEqual(key1, key2)


if __name__ == "__main__":
    unittest.main()
