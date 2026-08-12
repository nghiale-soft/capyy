import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from gateway.core.config import Settings
from gateway.services.chat_service import run_tool_agent_loop
from gateway.services.toolkit import (
    client_tool_call,
    detect_tool_markers,
    execute_tool,
    has_unfulfilled_tool_intent,
    parse_tool_call,
    tool_system_prompt,
)
from gateway.compat.openai import build_upstream_payload
from providers.freebuff import FreebuffSession


class ToolCallParsingTests(unittest.TestCase):
    def test_client_xml_edit_normalizes_gateway_path_to_file_path(self) -> None:
        call, _ = client_tool_call(
            '<invoke name="Edit"><parameter name="path">a.py</parameter>'
            '<parameter name="old_string">old</parameter>'
            '<parameter name="new_string">new</parameter></invoke>'
        )
        self.assertEqual(call["name"], "Edit")
        self.assertEqual(call["arguments"]["file_path"], "a.py")
        self.assertNotIn("path", call["arguments"])

    def test_client_dsml_gateway_read_file_uses_claude_read_schema(self) -> None:
        """A model may emit gateway names in DSML; Claude only exposes Read."""
        bar = "\uff5c"
        text = (
            f"<{bar}DSML{bar}tool_calls><{bar}DSML{bar}invoke name=\"read_file\">"
            f"<{bar}DSML{bar}parameter name=\"path\" string=\"true\">"
            f"/tmp/example.txt</{bar}DSML{bar}parameter>"
            f"</{bar}DSML{bar}invoke></{bar}DSML{bar}tool_calls>"
        )

        call, clean = client_tool_call(text)

        self.assertEqual(call, {"name": "Read", "arguments": {"file_path": "/tmp/example.txt"}})
        self.assertEqual(clean, "")

    def test_parse_tool_call_extracts_name_and_arguments(self) -> None:
        text = (
            "Let me check the file first.\n"
            '<<<TOOL_CALL>>>{"name": "read_file", '
            '"arguments": {"path": "a.py"}}<<<END_TOOL_CALL>>>\n'
            "Now I can answer."
        )
        call, clean = parse_tool_call(text)
        self.assertEqual(call["name"], "read_file")
        self.assertEqual(call["arguments"], {"path": "a.py"})
        self.assertNotIn("<<<TOOL_CALL>>>", clean)
        self.assertIn("Let me check the file first.", clean)

    def test_parse_tool_call_returns_none_when_absent(self) -> None:
        call, clean = parse_tool_call("plain answer, no tools")
        self.assertIsNone(call)
        self.assertEqual(clean, "plain answer, no tools")

    def test_detects_narrated_tool_call_without_protocol(self) -> None:
        self.assertTrue(
            has_unfulfilled_tool_intent(
                "Tôi gọi Edit ngay bây giờ với đầy đủ tham số."
            )
        )
        self.assertFalse(
            has_unfulfilled_tool_intent(
                '<<<TOOL_CALL>>>{"name":"edit_file","arguments":{}}<<<END_TOOL_CALL>>>'
            )
        )
        self.assertFalse(has_unfulfilled_tool_intent("Đã sửa file xong."))

    def test_client_xml_edit_can_be_parsed_from_reasoning_text(self) -> None:
        # The same parser is intentionally usable for the upstream reasoning
        # stream: DeepSeek occasionally places a complete invocation there.
        call, clean = client_tool_call(
            'Tôi thực hiện ngay. <invoke name="Edit">'
            '<parameter name="file_path">a.py</parameter>'
            '<parameter name="old_string">10.14</parameter>'
            '<parameter name="new_string">10.15</parameter></invoke>'
        )
        self.assertEqual(call["name"], "Edit")
        self.assertEqual(call["arguments"]["file_path"], "a.py")
        self.assertNotIn("<invoke", clean)

    def test_parse_tool_call_handles_bad_json(self) -> None:
        call, _ = parse_tool_call(
            '<<<TOOL_CALL>>>{"name": <<<END_TOOL_CALL>>>'
        )
        self.assertIsNone(call)

    # --- Claude Code XML <invoke> protocol (the VSCode extension format) ---

    def test_parse_claude_xml_read(self) -> None:
        text = (
            "Let me check the file.\n"
            '<invoke name="Read">\n'
            '<parameter name="file_path">src/app.py</parameter>\n'
            "</invoke>\n"
            "Now I can answer."
        )
        call, clean = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "read_file")
        self.assertEqual(call["arguments"]["path"], "src/app.py")
        self.assertNotIn("<invoke", clean)
        self.assertIn("Let me check the file.", clean)

    def test_parse_claude_xml_edit(self) -> None:
        text = (
            '<invoke name="Edit">\n'
            '<parameter name="file_path">/Users/x/project/lib/a.dart</parameter>\n'
            '<parameter name="old_string">var a = 1;</parameter>\n'
            '<parameter name="new_string">var a = 2;</parameter>\n'
            "</invoke>"
        )
        call, clean = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "edit_file")
        self.assertEqual(call["arguments"]["path"], "/Users/x/project/lib/a.dart")
        self.assertEqual(call["arguments"]["old_string"], "var a = 1;")
        self.assertEqual(call["arguments"]["new_string"], "var a = 2;")
        self.assertEqual(clean, "")

    def test_parse_claude_xml_write(self) -> None:
        text = (
            '<invoke name="Write">\n'
            '<parameter name="file_path">config.yaml</parameter>\n'
            '<parameter name="content">key: value</parameter>\n'
            "</invoke>"
        )
        call, clean = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "write_file")
        self.assertEqual(call["arguments"]["path"], "config.yaml")
        self.assertEqual(call["arguments"]["content"], "key: value")

    def test_parse_claude_xml_bash(self) -> None:
        text = (
            '<invoke name="Bash">\n'
            '<parameter name="command">grep -r foo .</parameter>\n'
            "</invoke>"
        )
        call, clean = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "bash")
        self.assertEqual(call["arguments"]["command"], "grep -r foo .")

    def test_parse_claude_xml_unknown_tool_is_stripped_not_leaked(self) -> None:
        # Unknown tools (TodoWrite/MultiEdit/KillShell...) must NOT leak as text:
        # they are stripped and turned into a synthetic call the loop can answer.
        text = '<invoke name="TodoWrite"><parameter name="todos">x</parameter></invoke>'
        call, clean = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "claude:TodoWrite")
        self.assertNotIn("<invoke", clean)

    def test_parse_claude_xml_missing_required_param(self) -> None:
        # Read without file_path -> stripped, synthetic call, no leak.
        text = 'text\n<invoke name="Read"><parameter name="other">x</parameter></invoke>\nend'
        call, clean = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "claude:Read")
        self.assertNotIn("<invoke", clean)
        self.assertIn("text", clean)
        self.assertIn("end", clean)

    def test_parse_claude_xml_edit_requires_path(self) -> None:
        # edit_file without file_path must not resolve to the workdir itself.
        text = (
            '<invoke name="Edit">\n'
            '<parameter name="old_string">a</parameter>\n'
            '<parameter name="new_string">b</parameter>\n'
            "</invoke>"
        )
        call, _ = parse_tool_call(text)
        self.assertEqual(call["name"], "claude:Edit")

    def test_parse_claude_xml_replace_all_coerced_to_bool(self) -> None:
        text = (
            '<invoke name="Edit">\n'
            '<parameter name="file_path">a.py</parameter>\n'
            '<parameter name="old_string">x</parameter>\n'
            '<parameter name="new_string">y</parameter>\n'
            '<parameter name="replace_all">false</parameter>\n'
            "</invoke>"
        )
        call, _ = parse_tool_call(text)
        self.assertEqual(call["name"], "edit_file")
        self.assertIs(call["arguments"]["replace_all"], False)

    def test_parse_claude_xml_inline_after_text(self) -> None:
        # The real Claude-extension failure: the block sits on the SAME line as
        # trailing prose ("...then:\n<invoke ...>") or even inline after a colon.
        text = (
            "Tiếp tục fix nốt lint. Thêm guard mounted sau await: "
            "<invoke name=\"Edit\">"
            '<parameter name="replace_all">false</parameter>'
            '<parameter name="file_path">/Users/x/lib/screen.dart</parameter>'
            '<parameter name="old_string">var deleteOriginal = false;</parameter>'
            '<parameter name="new_string">if (!mounted) return;</parameter>'
            "</invoke>"
        )
        call, clean = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "edit_file")
        self.assertEqual(call["arguments"]["path"], "/Users/x/lib/screen.dart")
        self.assertIs(call["arguments"]["replace_all"], False)
        self.assertNotIn("<invoke", clean)
        self.assertIn("Tiếp tục fix nốt lint", clean)

    def test_parse_claude_xml_in_prose_does_not_misfire(self) -> None:
        # A prose mention with no <parameter> children must not be treated as a call.
        text = (
            "In docs they show <invoke name=\"Read\"> inside code examples. "
            "It has no parameters here."
        )
        call, clean = parse_tool_call(text)
        self.assertIsNone(call)
        self.assertEqual(clean, text)

    def test_parse_claude_xml_unescapes_entities(self) -> None:
        text = (
            '<invoke name="Edit">\n'
            '<parameter name="file_path">a.py</parameter>\n'
            '<parameter name="old_string">a &lt; b &amp;&amp; c &gt; d</parameter>\n'
            '<parameter name="new_string">ok</parameter>\n'
            "</invoke>"
        )
        call, _ = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["arguments"]["old_string"], "a < b && c > d")

    def test_system_prompt_describes_tools(self) -> None:
        prompt = tool_system_prompt("/tmp/work", bash_enabled=True)
        self.assertIn("read_file", prompt)
        self.assertIn("bash", prompt)
        self.assertIn("/tmp/work", prompt)

    # --- manicode DSML protocol (the Claude Code fork format) ---

    _BAR = "\uff5c"

    def _dsml(
        self,
        name: str,
        params: list[tuple[str, str]],
        *,
        bar: str | None = None,
    ) -> str:
        bar = bar or self._BAR
        inner = "".join(
            f"<{bar}DSML{bar}parameter name=\"{key}\" string=\"true\">{value}</{bar}DSML{bar}parameter>"
            for key, value in params
        )
        return (
            f"<{bar}DSML{bar}tool_calls>\r"
            f"<{bar}DSML{bar}invoke name=\"{name}\">\r{inner}"
            f"</{bar}DSML{bar}invoke>\r</{bar}DSML{bar}tool_calls>"
        )

    def test_parse_dsml_edit(self) -> None:
        text = (
            "Tiếp tục fix nốt lint. "
            + self._dsml(
                "Edit",
                [
                    ("replace_all", "false"),
                    ("file_path", "/Users/x/lib/screen.dart"),
                    ("old_string", "var deleteOriginal = false;"),
                    ("new_string", "if (!mounted) return;"),
                ],
            )
            + " done"
        )
        call, clean = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "edit_file")
        self.assertEqual(call["arguments"]["path"], "/Users/x/lib/screen.dart")
        self.assertEqual(call["arguments"]["old_string"], "var deleteOriginal = false;")
        self.assertEqual(call["arguments"]["new_string"], "if (!mounted) return;")
        self.assertIs(call["arguments"]["replace_all"], False)
        self.assertNotIn("DSML", clean)
        self.assertNotIn("<", clean)
        self.assertIn("Tiếp tục fix nốt lint", clean)
        self.assertIn("done", clean)

    def test_parse_dsml_read_with_doubled_bars(self) -> None:
        # Some manicode builds double the bar: <｜｜DSML｜｜tool_calls>.
        text = self._dsml(
            "Read",
            [("file_path", "src/app.py")],
            bar=self._BAR * 2,
        )
        call, clean = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "read_file")
        self.assertEqual(call["arguments"]["path"], "src/app.py")
        self.assertEqual(clean, "")

    def test_parse_dsml_bash(self) -> None:
        text = self._dsml("Bash", [("command", "ls -la")])
        call, _ = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "bash")
        self.assertEqual(call["arguments"]["command"], "ls -la")

    def test_parse_dsml_unknown_tool_stripped_not_leaked(self) -> None:
        text = self._dsml("TodoWrite", [("todos", "x")])
        call, clean = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "claude:TodoWrite")
        self.assertNotIn("DSML", clean)
        self.assertNotIn("<", clean)

    def test_parse_dsml_open_tag_without_invoke_is_stripped(self) -> None:
        # A stray opening tag (block got cut off in the stream) must not leak.
        text = f"text\n<{self._BAR}DSML{self._BAR}tool_calls>\nmore"
        call, clean = parse_tool_call(text)
        self.assertIsNone(call)
        self.assertNotIn("DSML", clean)
        self.assertNotIn("<", clean)
        self.assertIn("text", clean)
        self.assertIn("more", clean)

    def test_detect_tool_markers_finds_dsml(self) -> None:
        text = self._dsml("Read", [("file_path", "a.py")])
        markers = detect_tool_markers(text)
        self.assertIn("dsml", markers)
        self.assertNotIn("dsml", detect_tool_markers("plain text"))

    def test_detect_tool_markers_finds_incomplete_internal_marker(self) -> None:
        markers = detect_tool_markers("<tool_invoke_edit>\n</tool_invoke>")
        self.assertIn("incomplete-tool-invoke", markers)

    def test_parse_dsml_multi_invoke_keeps_second_for_next_iteration(self) -> None:
        # Two invokes in one tool_calls block: only the first is consumed, the
        # second must remain in clean text so the loop parses it next iteration.
        bar = self._BAR
        text = (
            f"<{bar}DSML{bar}tool_calls>"
            f"<{bar}DSML{bar}invoke name=\"Read\">"
            f"<{bar}DSML{bar}parameter name=\"file_path\" string=\"true\">a.py</{bar}DSML{bar}parameter>"
            f"</{bar}DSML{bar}invoke>"
            f"<{bar}DSML{bar}invoke name=\"Read\">"
            f"<{bar}DSML{bar}parameter name=\"file_path\" string=\"true\">b.py</{bar}DSML{bar}parameter>"
            f"</{bar}DSML{bar}invoke>"
            f"</{bar}DSML{bar}tool_calls>"
        )
        call, clean = parse_tool_call(text)
        self.assertEqual(call["name"], "read_file")
        self.assertEqual(call["arguments"]["path"], "a.py")
        # Second invoke survives so the loop can parse it next round.
        self.assertIn('name="Read"', clean)
        self.assertIn("b.py", clean)
        self.assertNotIn("tool_calls>", clean)

    def test_parse_dsml_unescapes_entities(self) -> None:
        text = self._dsml(
            "Edit",
            [
                ("file_path", "a.py"),
                ("old_string", "a &lt; b &amp;&amp; c &gt; d"),
                ("new_string", "ok"),
            ],
        )
        call, _ = parse_tool_call(text)
        self.assertEqual(call["name"], "edit_file")
        self.assertEqual(call["arguments"]["old_string"], "a < b && c > d")


class ToolExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workdir = self._tmp.name

    def _write(self, name: str, content: str) -> Path:
        path = Path(self.workdir) / name
        path.write_text(content, encoding="utf-8")
        return path

    def _run(self, call, **kw) -> str:
        return asyncio.run(execute_tool(call, self.workdir, **kw))

    def test_read_file(self) -> None:
        self._write("a.txt", "hello world")
        result = self._run({"name": "read_file", "arguments": {"path": "a.txt"}})
        self.assertEqual(result, "hello world")

    def test_read_file_missing(self) -> None:
        result = self._run({"name": "read_file", "arguments": {"path": "nope.txt"}})
        self.assertIn("not found", result.lower())

    def test_write_file_then_read(self) -> None:
        result = self._run(
            {
                "name": "write_file",
                "arguments": {"path": "b.txt", "content": "written"},
            }
        )
        self.assertIn("b.txt", result)
        self.assertEqual((Path(self.workdir) / "b.txt").read_text(), "written")

    def test_list_dir(self) -> None:
        self._write("x.txt", "x")
        result = self._run({"name": "list_dir", "arguments": {"path": "."}})
        self.assertIn("x.txt", result)

    def test_glob(self) -> None:
        self._write("m1.py", "x")
        self._write("m2.py", "x")
        result = self._run({"name": "glob", "arguments": {"pattern": "*.py"}})
        self.assertIn("m1.py", result)
        self.assertIn("m2.py", result)

    def test_grep(self) -> None:
        self._write("app.py", "def main():\n    pass\n")
        result = self._run(
            {"name": "grep", "arguments": {"pattern": "def main", "path": "."}}
        )
        self.assertIn("app.py:1:def main", result)

    def test_bash_runs_command(self) -> None:
        result = self._run(
            {"name": "bash", "arguments": {"command": "echo hello-tool"}}
        )
        self.assertIn("hello-tool", result)

    def test_bash_disabled(self) -> None:
        result = self._run(
            {"name": "bash", "arguments": {"command": "echo x"}},
            bash_enabled=False,
        )
        self.assertIn("disabled", result.lower())

    def test_unknown_tool(self) -> None:
        result = self._run({"name": "nope", "arguments": {}})
        self.assertIn("unknown tool", result.lower())

    def test_tool_error_never_raises(self) -> None:
        result = self._run(
            {"name": "read_file", "arguments": {"path": "/definitely/not/here"}}
        )
        self.assertIn("not found", result.lower())

    def test_edit_file_targeted(self) -> None:
        self._write("e.txt", "aaa\nbbb\naaa")
        result = self._run(
            {
                "name": "edit_file",
                "arguments": {"path": "e.txt", "old_string": "bbb", "new_string": "CCC"},
            }
        )
        self.assertIn("Replaced 1", result)
        self.assertEqual((Path(self.workdir) / "e.txt").read_text(), "aaa\nCCC\naaa")

    def test_edit_file_replace_all(self) -> None:
        self._write("e2.txt", "aaa bbb aaa")
        result = self._run(
            {
                "name": "edit_file",
                "arguments": {
                    "path": "e2.txt",
                    "old_string": "aaa",
                    "new_string": "X",
                    "replace_all": True,
                },
            }
        )
        self.assertIn("Replaced 2", result)
        self.assertEqual((Path(self.workdir) / "e2.txt").read_text(), "X bbb X")

    def test_edit_file_missing_pattern(self) -> None:
        self._write("e3.txt", "hello")
        result = self._run(
            {
                "name": "edit_file",
                "arguments": {"path": "e3.txt", "old_string": "zzz", "new_string": "x"},
            }
        )
        self.assertIn("not found", result)

    def test_read_file_lines(self) -> None:
        self._write("l.txt", "\n".join(f"line{i}" for i in range(1, 6)))
        result = self._run(
            {"name": "read_file_lines", "arguments": {"path": "l.txt", "start": 2, "end": 4}}
        )
        self.assertIn("2:line2", result)
        self.assertIn("4:line4", result)
        self.assertNotIn("1:line1", result)

    def test_base64_roundtrip(self) -> None:
        encoded = self._run({"name": "base64_encode", "arguments": {"text": "hi"}})
        self.assertEqual(encoded, "aGk=")
        decoded = self._run({"name": "base64_decode", "arguments": {"text": encoded}})
        self.assertEqual(decoded, "hi")

    def test_url_roundtrip(self) -> None:
        encoded = self._run({"name": "url_encode", "arguments": {"text": "a b&c"}})
        self.assertNotIn(" ", encoded)
        decoded = self._run({"name": "url_decode", "arguments": {"text": encoded}})
        self.assertEqual(decoded, "a b&c")

    def test_uuid_and_timestamp(self) -> None:
        result = self._run({"name": "uuid", "arguments": {}})
        self.assertEqual(len(result), 36)
        ts = self._run({"name": "timestamp", "arguments": {}})
        self.assertIn("T", ts)

    def test_json_parse(self) -> None:
        result = self._run({"name": "json_parse", "arguments": {"text": '{"a":1}'}})
        self.assertIn('"a"', result)
        bad = self._run({"name": "json_parse", "arguments": {"text": "{oops"}})
        self.assertIn("Invalid JSON", bad)

    def test_git_status_in_workdir(self) -> None:
        result = self._run({"name": "git_status", "arguments": {}})
        # git may not be initialized in the temp dir — both paths are fine.
        self.assertIsInstance(result, str)

    def test_http_get_rejects_non_http(self) -> None:
        result = self._run({"name": "http_get", "arguments": {"url": "ftp://x"}})
        self.assertIn("http(s)", result)


class UpstreamToolStrippingTests(unittest.TestCase):
    def test_build_upstream_payload_strips_tools(self) -> None:
        payload = build_upstream_payload(
            {
                "model": "deepseek/deepseek-v4-flash",
                "messages": [],
                "tools": [{"type": "function", "function": {"name": "read_file"}}],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
            session=FreebuffSession(
                instance_id="instance-1",
                model="deepseek/deepseek-v4-flash",
            ),
            run_id="run-1",
            client_id="client-1",
        )
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertNotIn("parallel_tool_calls", payload)


class _ToolLoopClient:
    """Fake upstream that first asks for a tool call, then answers."""

    def __init__(self, tool_name: str = "read_file", workdir: str = ".") -> None:
        self.tool_name = tool_name
        self.workdir = workdir
        self.chat_events_calls = 0

    async def chat_events(self, payload):
        self.chat_events_calls += 1
        if self.chat_events_calls == 1:
            yield _chunk(
                "Let me read it. "
                '<<<TOOL_CALL>>>{"name": "read_file", '
                '"arguments": {"path": "notes.txt"}}<<<END_TOOL_CALL>>>'
            )
            yield "data: [DONE]"
            return
        yield _chunk("The file says: IMPORTANT CONTENT.")
        yield "data: [DONE]"


def _chunk(content: str) -> str:
    data = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek/deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": "stop",
            }
        ],
    }
    return f"data: {json.dumps(data)}"


class ToolAgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_loop_executes_tool_and_returns_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "notes.txt").write_text(
                "IMPORTANT CONTENT", encoding="utf-8"
            )
            settings = Settings(
                codebuff_token="token",
                local_api_key=None,
                tool_workdir=tmp,
            )
            client = _ToolLoopClient(workdir=tmp)
            payload = {
                "model": "deepseek/deepseek-v4-flash",
                "messages": [{"role": "user", "content": "read notes.txt"}],
            }
            response = await run_tool_agent_loop(
                client,
                payload,
                body=payload,
                settings=settings,
                model="deepseek/deepseek-v4-flash",
            )

            self.assertEqual(client.chat_events_calls, 2)
            message = response["choices"][0]["message"]
            self.assertEqual(message["content"], "The file says: IMPORTANT CONTENT.")
            # The upstream payload must never carry tools after the loop run
            self.assertNotIn("tools", payload)

    async def test_loop_executes_claude_xml_tool_call(self) -> None:
        """Claude-Code XML (<invoke name=...>) must execute locally, not leak."""

        class XmlClient(_ToolLoopClient):
            async def chat_events(self, payload):
                self.chat_events_calls += 1
                if self.chat_events_calls == 1:
                    yield _chunk(
                        'Let me read it.\n<invoke name="Read">'
                        '<parameter name="file_path">notes.txt</parameter>'
                        "</invoke>"
                    )
                    yield "data: [DONE]"
                    return
                yield _chunk("The file says: IMPORTANT CONTENT.")
                yield "data: [DONE]"

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "notes.txt").write_text(
                "IMPORTANT CONTENT", encoding="utf-8"
            )
            settings = Settings(
                codebuff_token="token",
                local_api_key=None,
                tool_workdir=tmp,
            )
            client = XmlClient(workdir=tmp)
            payload = {
                "model": "deepseek/deepseek-v4-flash",
                "messages": [{"role": "user", "content": "read notes.txt"}],
            }
            response = await run_tool_agent_loop(
                client,
                payload,
                body=payload,
                settings=settings,
                model="deepseek/deepseek-v4-flash",
            )
            self.assertEqual(client.chat_events_calls, 2)
            message = response["choices"][0]["message"]
            self.assertEqual(message["content"], "The file says: IMPORTANT CONTENT.")
            self.assertNotIn("<invoke", message["content"])

    async def test_loop_executes_dsml_tool_call(self) -> None:
        """manicode DSML (<｜DSML｜tool_calls>) must execute locally, not leak."""
        bar = "\uff5c"

        class DsmlClient(_ToolLoopClient):
            async def chat_events(self, payload):
                self.chat_events_calls += 1
                if self.chat_events_calls == 1:
                    yield _chunk(
                        "Let me read it. "
                        f"<{bar}DSML{bar}tool_calls>"
                        f"<{bar}DSML{bar}invoke name=\"Read\">"
                        f"<{bar}DSML{bar}parameter name=\"file_path\" string=\"true\">notes.txt</{bar}DSML{bar}parameter>"
                        f"</{bar}DSML{bar}invoke>"
                        f"</{bar}DSML{bar}tool_calls>"
                    )
                    yield "data: [DONE]"
                    return
                yield _chunk("The file says: IMPORTANT CONTENT.")
                yield "data: [DONE]"

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "notes.txt").write_text(
                "IMPORTANT CONTENT", encoding="utf-8"
            )
            settings = Settings(
                codebuff_token="token",
                local_api_key=None,
                tool_workdir=tmp,
            )
            client = DsmlClient(workdir=tmp)
            payload = {
                "model": "deepseek/deepseek-v4-flash",
                "messages": [{"role": "user", "content": "read notes.txt"}],
            }
            response = await run_tool_agent_loop(
                client,
                payload,
                body=payload,
                settings=settings,
                model="deepseek/deepseek-v4-flash",
            )
            self.assertEqual(client.chat_events_calls, 2)
            message = response["choices"][0]["message"]
            self.assertEqual(message["content"], "The file says: IMPORTANT CONTENT.")
            self.assertNotIn("DSML", message["content"])

    async def test_loop_returns_direct_answer_when_no_tool_call(self) -> None:
        class PlainClient(_ToolLoopClient):
            async def chat_events(self, payload):
                self.chat_events_calls += 1
                yield _chunk("No tools needed.")
                yield "data: [DONE]"

        settings = Settings(codebuff_token="token", local_api_key=None)
        client = PlainClient()
        payload = {
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
        }
        response = await run_tool_agent_loop(
            client,
            payload,
            body=payload,
            settings=settings,
            model="deepseek/deepseek-v4-flash",
        )
        self.assertEqual(client.chat_events_calls, 1)
        message = response["choices"][0]["message"]
        self.assertEqual(message["content"], "No tools needed.")

    async def test_loop_respects_iteration_budget(self) -> None:
        class LoopingClient(_ToolLoopClient):
            async def chat_events(self, payload):
                self.chat_events_calls += 1
                yield _chunk(
                    '<<<TOOL_CALL>>>{"name": "list_dir", '
                    '"arguments": {"path": "."}}<<<END_TOOL_CALL>>>'
                )
                yield "data: [DONE]"

        settings = Settings(
            codebuff_token="token",
            local_api_key=None,
            tool_max_iterations=3,
        )
        client = LoopingClient()
        payload = {
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "keep looping"}],
        }
        response = await run_tool_agent_loop(
            client,
            payload,
            body=payload,
            settings=settings,
            model="deepseek/deepseek-v4-flash",
        )
        self.assertEqual(client.chat_events_calls, 3)
        self.assertIsNotNone(response)


if __name__ == "__main__":
    unittest.main()
