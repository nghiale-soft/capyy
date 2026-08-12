import json
import tempfile
import unittest
from pathlib import Path

from gateway.deps import detect_client
from gateway.services.chat_service import (
    _compiler_execution_context,
    _compiler_task_context,
    tool_history_to_text_protocol,
)
from gateway.services.toolkit import (
    client_tool_call,
    coerce_client_tool_call_arguments,
    has_unfulfilled_tool_intent,
    parse_compiler_protocol,
    validate_client_tool_call,
    adapt_client_tool_call,
    load_tool_aliases,
)


class _FakeRequest:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class DetectClientTests(unittest.TestCase):
    def test_claude_code_cli_user_agent(self) -> None:
        request = _FakeRequest(
            {"user-agent": "claude-cli/2.1.85 (external, cli)", "x-app": "cli"}
        )
        self.assertEqual(detect_client(request), "claude-code")

    def test_claude_code_extension_user_agent(self) -> None:
        request = _FakeRequest(
            {"user-agent": "claude-code/2.1.181 (sdk-cli)"}
        )
        self.assertEqual(detect_client(request), "claude-code")

    def test_cline_via_x_title(self) -> None:
        request = _FakeRequest({"user-agent": "node", "x-title": "Cline"})
        self.assertEqual(detect_client(request), "cline")

    def test_codex_cli_user_agent(self) -> None:
        request = _FakeRequest(
            {"user-agent": "codex_cli_rs/0.105.0 (Darwin arm64) terminal"}
        )
        self.assertEqual(detect_client(request), "codex")

    def test_cursor_user_agent(self) -> None:
        request = _FakeRequest({"user-agent": "Cursor/2.4.28 (darwin arm64)"})
        self.assertEqual(detect_client(request), "cursor")

    def test_generic_api_falls_back_to_api(self) -> None:
        request = _FakeRequest({"user-agent": "curl/8.0"})
        self.assertEqual(detect_client(request), "api")


class ClientToolCallTests(unittest.TestCase):
    def test_json_aliases_override_runtime_claude_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tool-aliases.json"
            path.write_text(
                json.dumps({"version": 1, "mappings": [{
                    "canonical": "read_file",
                    "claude-code": {"tool": "OpenText", "arguments": {"path": "filename"}},
                }]}),
                encoding="utf-8",
            )
            parsed, reverse = load_tool_aliases(path)

        self.assertEqual(parsed["OpenText"], ("read_file", {"filename": "path"}))
        self.assertEqual(reverse["read_file"], ("OpenText", {"path": "filename"}))

    _BAR = "\uff5c"

    def test_json_protocol_maps_to_client_tool(self) -> None:
        text = (
            "Let me check. "
            '<<<TOOL_CALL>>>{"name": "read_file", '
            '"arguments": {"path": "a.py"}}<<<END_TOOL_CALL>>>'
        )
        call, clean = client_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "Read")
        self.assertEqual(call["arguments"], {"file_path": "a.py"})
        self.assertNotIn("<<<TOOL_CALL>>>", clean)
        self.assertIn("Let me check.", clean)

    def test_xml_bash_passthrough(self) -> None:
        text = (
            'Let me run it.\n<invoke name="Bash">\n'
            '<parameter name="command">ls -la</parameter>\n'
            "</invoke>"
        )
        call, clean = client_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "Bash")
        self.assertEqual(call["arguments"]["command"], "ls -la")
        self.assertNotIn("<invoke", clean)

    def test_xml_edit_passthrough(self) -> None:
        text = (
            '<invoke name="Edit">\n'
            '<parameter name="file_path">a.dart</parameter>\n'
            '<parameter name="old_string">x</parameter>\n'
            '<parameter name="new_string">y</parameter>\n'
            "</invoke>"
        )
        call, _ = client_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "Edit")
        self.assertEqual(call["arguments"]["file_path"], "a.dart")
        self.assertEqual(call["arguments"]["old_string"], "x")
        self.assertEqual(call["arguments"]["new_string"], "y")

    def test_dsml_passthrough(self) -> None:
        bar = self._BAR
        text = (
            f"<{bar}DSML{bar}tool_calls>"
            f"<{bar}DSML{bar}invoke name=\"Bash\">"
            f"<{bar}DSML{bar}parameter name=\"command\" string=\"true\">fvm flutter analyze</{bar}DSML{bar}parameter>"
            f"</{bar}DSML{bar}invoke>"
            f"</{bar}DSML{bar}tool_calls>"
        )
        call, clean = client_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "Bash")
        self.assertEqual(call["arguments"]["command"], "fvm flutter analyze")
        self.assertNotIn("DSML", clean)

    def test_no_tool_returns_none(self) -> None:
        call, clean = client_tool_call("plain answer, no tools")
        self.assertIsNone(call)
        self.assertEqual(clean, "plain answer, no tools")

    def test_plan_without_tool_is_detected_as_unfulfilled(self) -> None:
        self.assertTrue(has_unfulfilled_tool_intent(
            "Bắt đầu bằng việc tìm tài nguyên trong workspace và kiểm tra Figma."
        ))
        self.assertTrue(has_unfulfilled_tool_intent(
            "Tôi đã kết nối Figma MCP. Giờ lấy metadata file và context thiết kế để phân tích."
        ))

    def test_client_tool_validation_requires_declared_schema(self) -> None:
        tools = [
            {
                "name": "Edit",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                    },
                    "required": ["file_path", "old_string"],
                    "additionalProperties": False,
                },
            }
        ]
        self.assertEqual(
            validate_client_tool_call(
                {"name": "Edit", "arguments": {"file_path": "a.py", "old_string": "x"}},
                tools,
            ),
            (True, ""),
        )
        valid, reason = validate_client_tool_call(
            {"name": "Bash", "arguments": {"command": "pwd"}}, tools
        )
        self.assertFalse(valid)
        self.assertIn("not declared", reason)
        valid, reason = validate_client_tool_call(
            {"name": "Edit", "arguments": {"file_path": "a.py"}}, tools
        )
        self.assertFalse(valid)
        self.assertIn("missing required", reason)

    def test_listdir_is_safely_adapted_to_declared_bash(self) -> None:
        call = adapt_client_tool_call(
            {"name": "ListDir", "arguments": {"path": "my folder; no-op"}},
            [{"name": "Bash", "input_schema": {"type": "object"}}],
        )
        self.assertEqual(call["name"], "Bash")
        self.assertEqual(call["arguments"]["command"], "ls -la -- 'my folder; no-op'")

    def test_glob_is_safely_adapted_to_declared_bash(self) -> None:
        call = adapt_client_tool_call(
            {"name": "Glob", "arguments": {"pattern": "**/*.md; no-op"}},
            [{"name": "Bash", "input_schema": {"type": "object"}}],
        )
        self.assertEqual(call["name"], "Bash")
        self.assertEqual(
            call["arguments"]["command"], "rg --files -g '**/*.md; no-op'"
        )

    def test_read_file_lines_preserves_range_for_native_read(self) -> None:
        call, _ = client_tool_call(
            '<<<TOOL_CALL>>>{"name":"read_file_lines","arguments":{"path":"a.py","start":10,"end":14}}<<<END_TOOL_CALL>>>'
        )
        self.assertEqual(
            call,
            {"name": "Read", "arguments": {"file_path": "a.py", "offset": 10, "limit": 5}},
        )

    def test_compiler_protocol_json_and_empty_required_string(self) -> None:
        call, final = parse_compiler_protocol(
            '{"action":"tool_call","name":"read_file","arguments":{"path":"a.py"}}'
        )
        self.assertFalse(final)
        self.assertEqual(call, {"name": "Read", "arguments": {"file_path": "a.py"}})
        self.assertEqual(parse_compiler_protocol('{"action":"final"}'), (None, True))
        call, final = parse_compiler_protocol(
            'Tool selected:\n```json\n{"action":"tool_call","name":"read_file","arguments":{"path":"b.py"}}\n```'
        )
        self.assertFalse(final)
        self.assertEqual(call, {"name": "Read", "arguments": {"file_path": "b.py"}})
        call, final = parse_compiler_protocol(
            'Use {"action":"final"} only when done. Actual: '
            '{"action":"tool_call","name":"read_file","arguments":{"path":"c.py"}}'
        )
        self.assertFalse(final)
        self.assertEqual(call, {"name": "Read", "arguments": {"file_path": "c.py"}})
        call, final = parse_compiler_protocol(
            '{"name":"Bash","command":"pwd","description":"Show current directory"}'
        )
        self.assertFalse(final)
        self.assertEqual(
            call,
            {
                "name": "Bash",
                "arguments": {
                    "command": "pwd",
                    "description": "Show current directory",
                },
            },
        )
        valid, reason = validate_client_tool_call(
            {"name": "Edit", "arguments": {"file_path": "", "old_string": "x"}},
            [{"name": "Edit", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}}, "required": ["file_path", "old_string"]}}],
        )
        self.assertFalse(valid)
        self.assertIn("empty required", reason)

    def test_coerces_schema_number_and_boolean_strings(self) -> None:
        call = coerce_client_tool_call_arguments(
            {"name": "Bash", "arguments": {"timeout": "60", "background": "false"}},
            [{"name": "Bash", "input_schema": {"type": "object", "properties": {
                "timeout": {"type": "number"}, "background": {"type": "boolean"},
            }}}],
        )
        self.assertEqual(call["arguments"], {"timeout": 60, "background": False})


class ToolHistoryTextProtocolTests(unittest.TestCase):
    def test_compiler_context_keeps_prior_tool_arguments_and_results(self) -> None:
        context = _compiler_execution_context([
            {"role": "assistant", "tool_calls": [{"id": "f1", "function": {"name": "mcp__plugin_figma_figma__get_metadata", "arguments": '{"fileKey":"abc","nodeId":"0:1"}'}}]},
            {"role": "tool", "tool_call_id": "f1", "content": "page metadata"},
        ])
        self.assertIn('"fileKey\\\":\\\"abc', context)
        self.assertIn("page metadata", context)

    def test_compiler_task_context_keeps_pending_client_instruction(self) -> None:
        context = _compiler_task_context([
            {"role": "user", "content": "Update docs/project and write status-map.md."},
            {"role": "assistant", "content": "I will update it."},
            {"role": "system", "content": "Do not mark complete before the files are written."},
            {"role": "tool", "content": "ignored"},
        ])
        self.assertIn("Update docs/project", context)
        self.assertIn("Do not mark complete", context)
        self.assertNotIn("ignored", context)

    def test_converts_tool_result_to_text_protocol(self) -> None:
        messages = [
            {"role": "assistant", "content": "Running the check.",
             "tool_calls": [{"id": "call-1", "type": "function",
                             "function": {"name": "Bash", "arguments": "{\"command\": \"ls\"}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "file1\nfile2"},
        ]
        converted = tool_history_to_text_protocol(messages)

        self.assertEqual(len(converted), 2)
        assistant = converted[0]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"], "Running the check.")
        self.assertNotIn("tool_calls", assistant)
        self.assertEqual(converted[1]["role"], "user")
        self.assertIn("[tool result for Bash", converted[1]["content"])
        self.assertIn("file1\nfile2", converted[1]["content"])

    def test_passthrough_plain_messages(self) -> None:
        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
        ]
        converted = tool_history_to_text_protocol(messages)
        self.assertEqual(converted, messages)

    def test_tool_without_matching_assistant_call(self) -> None:
        messages = [
            {"role": "tool", "tool_call_id": "ghost", "content": "out"},
        ]
        converted = tool_history_to_text_protocol(messages)
        self.assertEqual(converted[0]["role"], "user")
        self.assertIn("[tool result for tool", converted[0]["content"])


if __name__ == "__main__":
    unittest.main()
