"""Local tool execution for the gateway agent loop.

Freebuff free models reject requests that carry ``tools`` (429
``free-models-per-day-high-balance``), so the gateway cannot ask upstream to
call tools. Instead the gateway:

1. strips ``tools`` / ``tool_choice`` from the upstream payload,
2. tells the model (via a system prompt) to emit tool calls as a single JSON
   line: ``<<<TOOL_CALL>>>{"name": "...", "arguments": {...}}<<<END_TOOL_CALL>>>``,
3. parses that line, executes the tool locally, appends the result as a new
   user message and lets the model continue — repeating until it answers
   without a tool call (or the iteration budget is exhausted).

Tools: filesystem (read/write/edit/list/glob/grep), shell (bash), git,
HTTP fetch, pure helpers (base64/url/uuid/timestamp/json), headless Chrome
(browser_*).
"""

from __future__ import annotations

import base64
import glob as _glob
import json
import logging
import re
import shlex
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

logger = logging.getLogger("gateway.services.toolkit")

TOOL_CALL_RE = re.compile(
    r"<<<TOOL_CALL>>>(.*?)<<<END_TOOL_CALL>>>",
    re.DOTALL,
)

# DSML = the tool-call protocol used by manicode (a Claude Code fork). The model
# wraps its tool calls in fullwidth-vertical-bar tags (\uff5c = U+FF5C):
#
#   <｜DSML｜tool_calls>
#   <｜DSML｜invoke name="Edit">
#   <｜DSML｜parameter name="file_path" string="true">src/a.py</｜DSML｜parameter>
#   <｜DSML｜parameter name="old_string" string="true">x</｜DSML｜parameter>
#   <｜DSML｜parameter name="new_string" string="true">y</｜DSML｜parameter>
#   </｜DSML｜invoke>
#   </｜DSML｜tool_calls>
#
# The bar count varies between single (｜DSML｜) and doubled (｜｜DSML｜｜) in the
# wild, so the regexes accept one-or-more bars.
_DSML_BAR = "\uff5c"  # ｜ FULLWIDTH VERTICAL LINE
_DSML_OPEN_RE = re.compile(
    rf"<{_DSML_BAR}+DSML{_DSML_BAR}+tool_calls>", re.IGNORECASE
)
_DSML_CLOSE_RE = re.compile(
    rf"</{_DSML_BAR}+DSML{_DSML_BAR}+tool_calls>", re.IGNORECASE
)
_DSML_INVOKE_RE = re.compile(
    rf"<{_DSML_BAR}+DSML{_DSML_BAR}+invoke\s+name=\"([^\"]+)\">(.*?)</{_DSML_BAR}+DSML{_DSML_BAR}+invoke>",
    re.DOTALL | re.IGNORECASE,
)
_DSML_PARAM_RE = re.compile(
    rf"<{_DSML_BAR}+DSML{_DSML_BAR}+parameter\s+name=\"([^\"]+)\"[^>]*>(.*?)</{_DSML_BAR}+DSML{_DSML_BAR}+parameter>",
    re.DOTALL | re.IGNORECASE,
)

# Any leftover marker fragment (opening/closing tags that arrived without a
# complete block, or the outer <...tool_calls> wrappers after an invoke block
# is removed) — used to detect leaks and to strip stray markers. Restricted to
# tags WITHOUT attributes so an untouched <invoke name=...> block in a
# multi-invoke text survives for the next loop iteration.
_DSML_FRAGMENT_RE = re.compile(
    rf"</?{_DSML_BAR}+DSML{_DSML_BAR}+[a-z_]+\s*>", re.IGNORECASE
)


def detect_tool_markers(text: str) -> list[str]:
    """Return which tool-call protocols appear in a text (for leak diagnostics)."""
    text = text or ""
    found: list[str] = []
    if TOOL_CALL_RE.search(text):
        found.append("gateway-json")
    if _INVOKE_RE.search(text):
        found.append("claude-xml")
    if (
        _DSML_OPEN_RE.search(text)
        or _DSML_FRAGMENT_RE.search(text)
        or _DSML_INVOKE_RE.search(text)
    ):
        found.append("dsml")
    return found


# Some small/free upstream models occasionally narrate a tool call ("I will
# call Edit now") but omit the protocol block altogether.  In a native Claude
# tool round-trip that is terminal: the extension can only continue after it
# receives an actual ``tool_use`` event.  Keep this deliberately narrow so a
# normal final answer that merely mentions a tool is not retried.
_UNFULFILLED_TOOL_INTENT_RE = re.compile(
    r"(?:\b(?:i(?:'ll| will| am going to)?|let me|please|tôi|tao|mình)\b"
    r".{0,72}?\b(?:call|run|execute|use|invoke|gọi|chạy|thực hiện|dùng)\b"
    r".{0,72}?\b(?:tool|edit|read|write|bash|command|lệnh)\b)",
    re.IGNORECASE | re.DOTALL,
)

# A FreeBuff model may stop after only describing the first intended action.
# In a native client-tool turn this is neither a final answer nor a usable
# tool_use event, so the gateway must ask once for an actual protocol action.
_UNFULFILLED_PLAN_RE = re.compile(
    r"\b(?:bắt đầu|trước tiên|đầu tiên|sẽ|đang|giờ|sau đó|tiếp theo|kế tiếp|"
    r"first|start|begin|next|now|then)\b"
    r".{0,160}?\b(?:tìm|kiểm tra|phân tích|đọc|truy cập|thực hiện|find|check|analy[sz]e|read|access|inspect)\b",
    re.IGNORECASE | re.DOTALL,
)


def has_unfulfilled_tool_intent(text: str) -> bool:
    """Whether a response promises a tool action but contains no tool call."""
    return not detect_tool_markers(text or "") and bool(
        _UNFULFILLED_TOOL_INTENT_RE.search(text or "")
        or _UNFULFILLED_PLAN_RE.search(text or "")
    )

# Tools that read only — always safe.
READ_TOOLS = ("read_file", "read_file_lines", "list_dir", "glob", "grep")


def tool_system_prompt(workdir: str, *, bash_enabled: bool) -> str:
    """System prompt that teaches the model the local tool-call protocol."""
    bash_line = (
        "- bash(command) -> run a shell command (timeout enforced, output capped)"
        if bash_enabled
        else "- bash(command) -> DISABLED on this gateway"
    )
    return f"""\
You are running inside an AI coding gateway on the user's machine.
You have access to LOCAL tools. When you need to inspect or modify the codebase,
run a command, open a web page or read a design file, output a tool call as a
single line with EXACTLY this JSON format (nothing else on that line):
<<<TOOL_CALL>>>{{"name": "read_file", "arguments": {{"path": "README.md"}}}}<<<END_TOOL_CALL>>>
If your client instructs you to call tools with Claude-style XML
(<invoke name="Edit">...</invoke>) or manicode DSML
(<｜DSML｜tool_calls><｜DSML｜invoke name="Edit">...</｜DSML｜tool_calls>), those are also
accepted and executed — use the parameter names from those instructions
(e.g. file_path/old_string/new_string).
The tool runs immediately and its output is appended to the conversation so you
can continue. Use tools whenever you need facts about the filesystem, a command
result, a web page or a Figma design — never guess.

Available tools:
- read_file(path) -> read a text file (output capped)
- read_file_lines(path, start, end?) -> read a line range with line numbers
- write_file(path, content) -> create/overwrite a text file (asks approval)
- edit_file(path, old_string, new_string, replace_all?) -> targeted edit (asks approval)
- list_dir(path='.') -> list a directory's entries
- glob(pattern) -> match files by glob pattern (relative to the workdir)
- grep(pattern, path='.') -> search file contents under a path
{bash_line}
- git_status() / git_diff(path?) -> git state / diff
- http_get(url) -> fetch a URL (asks approval; network access)
- base64_encode(text) / base64_decode(text) / url_encode(text) / url_decode(text)
- uuid() / timestamp() -> random id / current UTC time
- json_parse(text) -> validate + pretty-print JSON
- browser_open(url) / browser_snapshot() / browser_click(selector) / browser_type(selector, text) / browser_eval(js) / browser_screenshot(path?) / browser_close() -> headless Chrome session (navigation/click/type ask approval)

Work directory: {workdir}
After the tool output arrives, continue reasoning and, when the task is done, reply
to the user WITHOUT any tool-call line."""


def parse_tool_call(text: str) -> tuple[dict[str, Any] | None, str]:
    """Extract the first tool call from model text.

    Recognizes two protocols:

    1. The gateway's own ``<<<TOOL_CALL>>>{"name": ..., "arguments": ...}<<<END_TOOL_CALL>>>``
    2. Claude Code XML: ``<invoke name="Read">...<parameter name="path">...</parameter>...</invoke>``
       (Claude Code / the VSCode extension teach the model this format; if we
       don't parse it, the XML leaks into the final answer as garbage and the
       client hangs waiting for a tool result that never arrives).

    Returns ``(call, clean_text)`` where ``call`` is ``{"name", "arguments"}``
    (or ``None``) and ``clean_text`` is the text with the tool-call removed.
    """
    text = text or ""
    match = TOOL_CALL_RE.search(text)
    if match:
        raw = match.group(1).strip()
        try:
            call = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("tool call JSON parse failed: %s", raw[:200])
            call = {}
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            name = call["name"]
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            clean = (text[: match.start()] + text[match.end():]).strip()
            return {"name": name, "arguments": arguments}, clean

    xml_call, xml_clean = _parse_claude_xml_invoke(text)
    if xml_call is not None:
        return xml_call, xml_clean

    dsml_call, dsml_clean = _parse_dsml(text)
    if dsml_call is not None:
        return dsml_call, dsml_clean
    if dsml_clean != text:
        # Stray DSML markers were stripped (open tag with no usable block).
        return None, dsml_clean
    return None, text


# Claude Code tool names -> (gateway tool name, {claude param name: gateway arg}).
# Maps Claude Code's XML tool protocol to the gateway's local toolkit so the
# VSCode extension's tool calls execute locally instead of leaking as text.
_CLAUDE_TOOL_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "Read": ("read_file", {"file_path": "path"}),
    "Write": ("write_file", {"file_path": "path", "content": "content"}),
    "Edit": (
        "edit_file",
        {
            "file_path": "path",
            "old_string": "old_string",
            "new_string": "new_string",
            "replace_all": "replace_all",
        },
    ),
    "Bash": ("bash", {"command": "command", "description": "description"}),
    "Glob": ("glob", {"pattern": "pattern"}),
    "Grep": ("grep", {"pattern": "pattern"}),
    "ListDir": ("list_dir", {"path": "path"}),
    "WebFetch": ("http_get", {"url": "url"}),
}

# Require a path for file tools; bash needs a command; read tools need their key.
_CLAUDE_REQUIRED: dict[str, tuple[str, ...]] = {
    "read_file": ("path",),
    "write_file": ("path", "content"),
    "edit_file": ("path", "old_string"),
    "bash": ("command",),
    "glob": ("pattern",),
    "grep": ("pattern",),
    "list_dir": ("path",),
    "http_get": ("url",),
}

# Claude Code may emit the block inline after text ("... then:
# <invoke name=...>"); match it anywhere. To avoid misfiring on prose that
# merely *mentions* an invoke tag, a match only counts when it has at least one
# <parameter> child (real tool calls always carry parameters).
_INVOKE_RE = re.compile(r"<invoke\s+name=\"([^\"]+)\">(.*?)</invoke>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter\s+name=\"([^\"]+)\">(.*?)</parameter>", re.DOTALL)


def _parse_claude_xml_invoke(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse a Claude Code ``<invoke name="...">`` block into a gateway tool call.

    Handles the plain-text protocol Claude Code uses (``<parameter name="x">``
    tags). For a matched-but-unusable block (unknown tool / missing required
    param) it still strips the XML and returns a synthetic unknown-tool call so
    the agent loop continues instead of leaking the XML as garbage to the client.
    """
    match = _INVOKE_RE.search(text)
    if not match:
        return None, text
    name = match.group(1).strip()
    params = list(_PARAM_RE.finditer(match.group(2)))
    block_clean = (text[: match.start()] + text[match.end():]).strip()
    mapped = _CLAUDE_TOOL_MAP.get(name)
    # No <parameter> children -> prose mention, not a real call: leave the text
    # untouched so quoting a docs snippet doesn't get mangled.
    if not params:
        return None, text
    if mapped is None:
        logger.info("claude tool not mapped: %s", name)
        return {"name": f"claude:{name}", "arguments": {}}, block_clean
    gateway_name, param_map = mapped
    arguments: dict[str, Any] = {}
    for param_match in params:
        param_name = param_match.group(1).strip()
        if param_name not in param_map:
            continue
        value: Any = _unescape_xml(param_match.group(2))
        # XML "true"/"false" strings would be truthy for replace_all — coerce.
        if param_name == "replace_all":
            value = value.strip().lower() in ("1", "true", "yes")
        arguments[param_map[param_name]] = value
    required = _CLAUDE_REQUIRED.get(gateway_name, ())
    if not all(key in arguments for key in required):
        return {"name": f"claude:{name}", "arguments": {}}, block_clean
    return {"name": gateway_name, "arguments": arguments}, block_clean


def _parse_dsml(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse a manicode DSML tool block into a gateway tool call.

    DSML wraps the Claude-style invoke block in ``<｜DSML｜tool_calls>`` markers
    (｜ = U+FF5C fullwidth vertical bar). The inner structure is the same as
    Claude Code's plain XML (``<｜DSML｜invoke name="Edit">`` with
    ``<｜DSML｜parameter name="x">value</｜DSML｜parameter>`` children), so the
    same tool map / required-param rules apply.
    """
    if not _DSML_OPEN_RE.search(text):
        return None, text
    invoke_match = _DSML_INVOKE_RE.search(text)
    if not invoke_match:
        # Opening tag present but no usable invoke block: strip the stray
        # markers so they don't leak as garbage into the final answer.
        clean = _DSML_FRAGMENT_RE.sub("", text).strip()
        return None, clean
    name = invoke_match.group(1).strip()
    params = list(_DSML_PARAM_RE.finditer(invoke_match.group(2)))
    # Cut only THIS invoke block (not the outer </tool_calls>), so that a
    # multi-invoke block keeps its remaining invokes in the text for the next
    # loop iteration — matching the Claude-XML parser's behavior. The leftover
    # <…tool_calls> wrappers are then stripped by the fragment regex.
    clean = (text[: invoke_match.start()] + text[invoke_match.end():]).strip()
    clean = _DSML_FRAGMENT_RE.sub("", clean).strip()
    if not params:
        return None, clean
    mapped = _CLAUDE_TOOL_MAP.get(name)
    if mapped is None:
        logger.info("dsml tool not mapped: %s", name)
        return {"name": f"claude:{name}", "arguments": {}}, clean
    gateway_name, param_map = mapped
    arguments: dict[str, Any] = {}
    for param_match in params:
        param_name = param_match.group(1).strip()
        if param_name not in param_map:
            continue
        value: Any = _unescape_xml(param_match.group(2))
        if param_name == "replace_all":
            value = value.strip().lower() in ("1", "true", "yes")
        arguments[param_map[param_name]] = value
    required = _CLAUDE_REQUIRED.get(gateway_name, ())
    if not all(key in arguments for key in required):
        return {"name": f"claude:{name}", "arguments": {}}, clean
    return {"name": gateway_name, "arguments": arguments}, clean


def _unescape_xml(value: str) -> str:
    return (
        value.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )


# Reverse of _CLAUDE_TOOL_MAP: gateway tool name -> (client tool name, arg map).
# Used to convert a parsed gateway tool call back into the client's native tool
# so the client (Claude Code / Cline) can execute it itself (native tool_use).
_CLIENT_TOOL_REVERSE: dict[str, tuple[str, dict[str, str]]] = {
    "read_file": ("Read", {"path": "file_path"}),
    "read_file_lines": ("Read", {"path": "file_path"}),
    "write_file": ("Write", {"path": "file_path", "content": "content"}),
    "edit_file": (
        "Edit",
        {
            "path": "file_path",
            "old_string": "old_string",
            "new_string": "new_string",
            "replace_all": "replace_all",
        },
    ),
    "bash": ("Bash", {"command": "command", "description": "description"}),
    "glob": ("Glob", {"pattern": "pattern"}),
    "grep": ("Grep", {"pattern": "pattern", "path": "path"}),
    "list_dir": ("ListDir", {"path": "path"}),
    "http_get": ("WebFetch", {"url": "url"}),
}


def _as_client_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return a call using the tool schema advertised by the IDE client.

    Models sometimes emit the gateway's canonical names inside XML/DSML
    (``read_file`` + ``path``), even though the client only exposes Claude
    Code's native names (``Read`` + ``file_path``).  Preserve genuinely native
    or unknown tools, but translate known gateway names before sending a native
    ``tool_use`` block back to the client.
    """
    if name == "read_file_lines":
        # Claude's native Read accepts offset/limit rather than an inclusive
        # start/end pair. Preserve the requested range instead of silently
        # degrading to a full-file Read.
        client_args: dict[str, Any] = {}
        if "path" in arguments:
            client_args["file_path"] = arguments["path"]
        if isinstance(arguments.get("start"), int):
            client_args["offset"] = arguments["start"]
            if isinstance(arguments.get("end"), int):
                client_args["limit"] = max(1, arguments["end"] - arguments["start"] + 1)
        return {"name": "Read", "arguments": client_args}

    mapped = _CLIENT_TOOL_REVERSE.get(name)
    if mapped is None:
        # Some upstream models use Claude's native tool *name* but gateway's
        # internal `path` argument. Claude Code validates the native schema
        # strictly and requires `file_path`, otherwise Edit fails before it
        # reaches the host tool.
        if name in {"Read", "Write", "Edit"} and "path" in arguments and "file_path" not in arguments:
            arguments = {**arguments, "file_path": arguments["path"]}
            arguments.pop("path", None)
        return {"name": name, "arguments": arguments}
    client_name, arg_map = mapped
    client_args = {
        arg_map[key]: value for key, value in arguments.items() if key in arg_map
    }
    return {"name": client_name, "arguments": client_args}


def client_tool_call(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse the first tool call and express it with CLIENT tool names/params.

    Unlike ``parse_tool_call`` (which maps everything to gateway tools for local
    execution), this returns the tool call exactly as the client expects it so
    the gateway can stream a native ``tool_use`` block and let the client run
    the tool itself (Claude Code extension / Cline show their own approval UI
    and execute on the host). Recognizes:

    1. The gateway's own ``<<<TOOL_CALL>>>{"name": ..., "arguments": ...}``
       (mapped to the client tool name via ``_CLIENT_TOOL_REVERSE``)
    2. Claude Code XML ``<invoke name="Bash">`` — name + ALL params are kept
       verbatim (client-native names like ``command``/``file_path``), including
       for tools the gateway does not map locally.
    3. manicode DSML ``<｜DSML｜invoke name="Edit">`` — same.

    Returns ``(call, clean_text)`` where ``call`` is ``{"name", "arguments"}``
    or ``None``, and ``clean_text`` is the model text with the tool call removed.
    """
    text = text or ""
    match = TOOL_CALL_RE.search(text)
    if match:
        raw = match.group(1).strip()
        try:
            call = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("client tool call JSON parse failed: %s", raw[:200])
            call = {}
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            name = call["name"]
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            clean = (text[: match.start()] + text[match.end():]).strip()
            return _as_client_tool_call(name, arguments), clean

    xml_call, xml_clean = _parse_client_xml(text)
    if xml_call is not None:
        return xml_call, xml_clean

    dsml_call, dsml_clean = _parse_client_dsml(text)
    if dsml_call is not None:
        return dsml_call, dsml_clean
    if dsml_clean != text:
        return None, dsml_clean
    return None, text


def validate_client_tool_call(
    call: dict[str, Any], client_tools: Any,
) -> tuple[bool, str]:
    """Validate a normalized call against tools declared by the API client.

    FreeBuff never receives these schemas.  They are retained at the gateway
    boundary so an upstream text protocol cannot invoke an undeclared local or
    MCP tool, nor omit a required input before it reaches the IDE client.
    Supports Anthropic ``name/input_schema`` and OpenAI
    ``function.name/function.parameters`` shapes.
    """
    if not isinstance(client_tools, list):
        return False, "request did not declare a usable tools list"

    name = call.get("name")
    arguments = call.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return False, "tool call has invalid name or arguments"

    schema: dict[str, Any] | None = None
    for tool in client_tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            candidate_name = function.get("name")
            candidate_schema = function.get("parameters")
        else:
            candidate_name = tool.get("name")
            candidate_schema = tool.get("input_schema") or tool.get("parameters")
        if candidate_name == name:
            schema = candidate_schema if isinstance(candidate_schema, dict) else {}
            break
    if schema is None:
        return False, f"tool {name!r} was not declared by the client"

    required = schema.get("required") or []
    if isinstance(required, list):
        for key in required:
            if isinstance(key, str) and (key not in arguments or arguments[key] is None):
                return False, f"tool {name!r} is missing required argument {key!r}"

    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return True, ""
    for key in required if isinstance(required, list) else []:
        if not isinstance(key, str):
            continue
        spec = properties.get(key)
        value = arguments.get(key)
        if (
            isinstance(spec, dict)
            and spec.get("type") == "string"
            and isinstance(value, str)
            and not value.strip()
        ):
            return False, f"tool {name!r} has empty required argument {key!r}"
    if schema.get("additionalProperties") is False:
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            return False, f"tool {name!r} has undeclared arguments: {', '.join(unexpected)}"
    for key, value in arguments.items():
        spec = properties.get(key)
        if not isinstance(spec, dict) or value is None:
            continue
        expected = spec.get("type")
        valid = (
            expected is None
            or (expected == "string" and isinstance(value, str))
            or (expected == "boolean" and isinstance(value, bool))
            or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
            or (expected == "array" and isinstance(value, list))
            or (expected == "object" and isinstance(value, dict))
        )
        if not valid:
            return False, f"tool {name!r} argument {key!r} must be {expected}"
        enum = spec.get("enum")
        if isinstance(enum, list) and value not in enum:
            return False, f"tool {name!r} argument {key!r} is outside its enum"
    return True, ""


def coerce_client_tool_call_arguments(
    call: dict[str, Any], client_tools: Any,
) -> dict[str, Any]:
    """Coerce unambiguous JSON scalar strings to the declared schema type."""
    if not isinstance(client_tools, list) or not isinstance(call.get("arguments"), dict):
        return call
    name = call.get("name")
    schema: dict[str, Any] | None = None
    for tool in client_tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        candidate_name = function.get("name") if isinstance(function, dict) else tool.get("name")
        candidate_schema = (
            function.get("parameters") if isinstance(function, dict)
            else tool.get("input_schema") or tool.get("parameters")
        )
        if candidate_name == name and isinstance(candidate_schema, dict):
            schema = candidate_schema
            break
    if schema is None:
        return call
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return call
    arguments = dict(call["arguments"])
    for key, value in arguments.items():
        spec = properties.get(key)
        if not isinstance(spec, dict) or not isinstance(value, str):
            continue
        raw = value.strip()
        expected = spec.get("type")
        if expected == "boolean" and raw.lower() in {"true", "false"}:
            arguments[key] = raw.lower() == "true"
        elif expected == "integer" and re.fullmatch(r"[+-]?\d+", raw):
            arguments[key] = int(raw)
        elif expected == "number" and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw):
            arguments[key] = float(raw) if "." in raw else int(raw)
    return {**call, "arguments": arguments}


def parse_compiler_protocol(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse the strict, private compiler contract without language heuristics.

    The compiler should return ``{"action":"tool_call",...}`` or
    ``{"action":"final"}``. A fenced JSON object is accepted for resilience
    with models that wrap JSON in Markdown. Some upstream models emit a flat
    call object (``{"name":"Bash","command":"pwd"}``); normalize that
    structural variant here, then let the normal client-schema validator decide
    whether it is actually executable. Arbitrary prose is still rejected.
    """
    raw = (text or "").strip()
    if raw.startswith("```json") and raw.endswith("```"):
        raw = raw[7:-3].strip()
    elif raw.startswith("```") and raw.endswith("```"):
        raw = raw[3:-3].strip()
    values: list[Any] = []
    strict_value: Any | None = None
    try:
        strict_value = json.loads(raw)
        values.append(strict_value)
    except (TypeError, ValueError):
        # A model may add one sentence before/after an otherwise valid object.
        # Extract only objects carrying the exact compiler `action` contract;
        # prose itself never becomes a tool call.
        decoder = json.JSONDecoder()
        for start, char in enumerate(raw):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw[start:])
            except ValueError:
                continue
            values.append(value)
    # A final declaration is valid only as the entire compiler response. Models
    # often mention the literal `{\"action\":\"final\"}` while explaining a
    # later tool call; treating that embedded example as final caused a false
    # stop. Embedded JSON may recover tool_call only.
    if isinstance(strict_value, dict) and strict_value.get("action") == "final":
        return None, True
    for value in values:
        if not isinstance(value, dict):
            continue
        if (
            value.get("action") == "tool_call"
            and isinstance(value.get("name"), str)
            and isinstance(value.get("arguments"), dict)
        ):
            return _as_client_tool_call(value["name"], value["arguments"]), False
        # FreeBuff occasionally follows the *shape* of a tool call but omits
        # the compiler envelope and flattens arguments at the root.  This is
        # deterministic JSON, not language inference. Keep every non-metadata
        # field (including `description`, which is a valid optional Bash arg)
        # and validate it against the client's declared schema downstream.
        name = value.get("name")
        if isinstance(name, str) and name and value.get("action") is None:
            if isinstance(value.get("arguments"), dict):
                return _as_client_tool_call(name, value["arguments"]), False
            arguments = {
                key: item
                for key, item in value.items()
                if key not in {"name", "action", "type", "id"}
            }
            if arguments:
                return _as_client_tool_call(name, arguments), False
    # Compatibility with the original text protocol during rollout.
    call, _ = client_tool_call(text)
    if call is not None:
        return call, False
    if raw == "<<<FINAL>>>":
        return None, True
    return None, False


def declared_client_tool_names(client_tools: Any) -> list[str]:
    """Extract names from either Anthropic or OpenAI client tool schemas."""
    if not isinstance(client_tools, list):
        return []
    names: list[str] = []
    for tool in client_tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else tool.get("name")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def adapt_client_tool_call(call: dict[str, Any], client_tools: Any) -> dict[str, Any]:
    """Make a safe compatibility conversion only to a declared client tool.

    Claude Code does not expose a native ``ListDir`` tool, although some models
    emit that common gateway alias. When Bash is declared, ``ls`` is the exact
    host-side equivalent. Shell-quote the path and leave every other call
    untouched for normal schema validation.
    """
    names = declared_client_tool_names(client_tools)
    arguments = call.get("arguments")
    if (
        call.get("name") == "ListDir"
        and "ListDir" not in names
        and "Bash" in names
        and isinstance(arguments, dict)
        and isinstance(arguments.get("path"), str)
    ):
        return {
            "name": "Bash",
            "arguments": {"command": f"ls -la -- {shlex.quote(arguments['path'])}"},
        }
    if (
        call.get("name") == "Glob"
        and "Glob" not in names
        and "Bash" in names
        and isinstance(arguments, dict)
        and isinstance(arguments.get("pattern"), str)
    ):
        return {
            "name": "Bash",
            "arguments": {
                "command": f"rg --files -g {shlex.quote(arguments['pattern'])}",
                "description": "List files matching the requested glob pattern",
            },
        }
    return call


def _parse_client_xml(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse one Claude Code XML ``<invoke name=...>`` block for the client.

    Native names/parameters (``command``, ``file_path``, ``old_string``, ...)
    are preserved.  A known gateway name such as ``read_file`` is normalized to
    the equivalent client tool (``Read``) so it is executable by the extension.
    """
    match = _INVOKE_RE.search(text)
    if not match:
        return None, text
    name = match.group(1).strip()
    params = list(_PARAM_RE.finditer(match.group(2)))
    clean = (text[: match.start()] + text[match.end():]).strip()
    # No <parameter> children -> prose mention, not a real call.
    if not params:
        return None, text
    arguments: dict[str, Any] = {}
    for param_match in params:
        param_name = param_match.group(1).strip()
        value: Any = _unescape_xml(param_match.group(2))
        if param_name == "replace_all":
            value = value.strip().lower() in ("1", "true", "yes")
        arguments[param_name] = value
    return _as_client_tool_call(name, arguments), clean


def _parse_client_dsml(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse one manicode DSML block and normalize known gateway tool names."""
    if not _DSML_OPEN_RE.search(text):
        return None, text
    invoke_match = _DSML_INVOKE_RE.search(text)
    if not invoke_match:
        clean = _DSML_FRAGMENT_RE.sub("", text).strip()
        return None, clean
    name = invoke_match.group(1).strip()
    params = list(_DSML_PARAM_RE.finditer(invoke_match.group(2)))
    clean = (text[: invoke_match.start()] + text[invoke_match.end():]).strip()
    clean = _DSML_FRAGMENT_RE.sub("", clean).strip()
    if not params:
        return None, clean
    arguments: dict[str, Any] = {}
    for param_match in params:
        param_name = param_match.group(1).strip()
        value: Any = _unescape_xml(param_match.group(2))
        if param_name == "replace_all":
            value = value.strip().lower() in ("1", "true", "yes")
        arguments[param_name] = value
    return _as_client_tool_call(name, arguments), clean


async def execute_tool(
    call: dict[str, Any],
    workdir: str,
    *,
    bash_enabled: bool = True,
    command_timeout: float = 30.0,
    output_cap: int = 50000,
    file_cap: int = 100000,
    project_key: str = "",
) -> str:
    """Execute one tool call and return its text output (never raises)."""
    name = call.get("name") or ""
    arguments = call.get("arguments") or {}
    try:
        if name == "read_file":
            return _read_file(arguments, workdir, file_cap=file_cap)
        if name == "read_file_lines":
            return _read_file_lines(arguments, workdir)
        if name == "write_file":
            return _write_file(arguments, workdir)
        if name == "edit_file":
            return _edit_file(arguments, workdir)
        if name == "list_dir":
            return _list_dir(arguments, workdir)
        if name == "glob":
            return _glob_files(arguments, workdir)
        if name == "grep":
            return _grep(arguments, workdir, output_cap=output_cap)
        if name == "bash":
            return _bash(
                arguments,
                workdir,
                enabled=bash_enabled,
                timeout=command_timeout,
                output_cap=output_cap,
            )
        if name == "git_status":
            return _git(arguments, workdir, args=["status", "--short"])
        if name == "git_diff":
            return _git(arguments, workdir, args=["diff"])
        if name == "http_get":
            return await _http_get(arguments)
        if name == "base64_encode":
            return _base64(arguments, encode=True)
        if name == "base64_decode":
            return _base64(arguments, encode=False)
        if name == "url_encode":
            return quote(str(arguments.get("text") or ""), safe="")
        if name == "url_decode":
            return unquote(str(arguments.get("text") or ""))
        if name == "uuid":
            return str(uuid.uuid4())
        if name == "timestamp":
            return datetime.now(timezone.utc).isoformat()
        if name == "json_parse":
            return _json_parse(arguments)
        if name.startswith("browser_"):
            return await _browser_dispatch(name, arguments)
        return f"Unknown tool: {name}"
    except Exception as error:  # noqa: BLE001 - tools must never crash the loop
        logger.warning("tool %s failed: %s", name, error)
        return f"Tool {name} failed: {error}"


def _resolve(path_value: Any, workdir: str) -> Path:
    path = str(path_value or "").strip() or "."
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(workdir) / candidate


def _read_file(arguments: dict[str, Any], workdir: str, *, file_cap: int) -> str:
    path = _resolve(arguments.get("path"), workdir)
    if not path.exists():
        return f"File not found: {path}"
    if path.is_dir():
        return f"Is a directory, not a file: {path}"
    data = path.read_text(encoding="utf-8", errors="replace")
    if len(data) > file_cap:
        data = data[:file_cap] + f"\n...[truncated, {len(data) - file_cap} chars dropped]"
    return data


def _read_file_lines(arguments: dict[str, Any], workdir: str) -> str:
    path = _resolve(arguments.get("path"), workdir)
    if not path.exists():
        return f"File not found: {path}"
    if path.is_dir():
        return f"Is a directory, not a file: {path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, int(arguments.get("start") or 1))
    end_value = arguments.get("end")
    end = len(lines) if end_value is None else min(len(lines), max(start, int(end_value)))
    selected = lines[start - 1:end]
    if not selected:
        return "(empty range)"
    return "\n".join(f"{i + start}:{line}" for i, line in enumerate(selected))


def _write_file(arguments: dict[str, Any], workdir: str) -> str:
    path = _resolve(arguments.get("path"), workdir)
    content = arguments.get("content")
    if not isinstance(content, str):
        return "write_file requires a string 'content' argument"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


def _edit_file(arguments: dict[str, Any], workdir: str) -> str:
    path = _resolve(arguments.get("path"), workdir)
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string") or ""
    if not isinstance(old_string, str) or not old_string:
        return "edit_file requires a non-empty 'old_string'"
    if not path.exists():
        return f"File not found: {path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    if arguments.get("replace_all"):
        count = text.count(old_string)
        if not count:
            return f"Pattern not found in {path}"
        text = text.replace(old_string, new_string)
    else:
        index = text.find(old_string)
        if index == -1:
            return f"Pattern not found in {path}"
        count = 1
        text = text[:index] + new_string + text[index + len(old_string):]
    path.write_text(text, encoding="utf-8")
    return f"Replaced {count} occurrence(s) in {path}"


def _list_dir(arguments: dict[str, Any], workdir: str) -> str:
    path = _resolve(arguments.get("path"), workdir)
    if not path.exists():
        return f"Path not found: {path}"
    if not path.is_dir():
        return f"Not a directory: {path}"
    entries = sorted(path.iterdir())
    lines = []
    for entry in entries:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{entry.name}{suffix}")
    return "\n".join(lines) if lines else "(empty directory)"


def _glob_files(arguments: dict[str, Any], workdir: str) -> str:
    pattern = str(arguments.get("pattern") or "").strip()
    if not pattern:
        return "glob requires a 'pattern' argument"
    search = pattern if pattern.startswith("/") else f"{workdir}/{pattern}"
    matches = sorted(_glob.glob(search, recursive=True))
    return "\n".join(matches) if matches else "(no matches)"


def _grep(arguments: dict[str, Any], workdir: str, *, output_cap: int) -> str:
    pattern = str(arguments.get("pattern") or "")
    if not pattern:
        return "grep requires a 'pattern' argument"
    base = _resolve(arguments.get("path"), workdir)
    if not base.exists():
        return f"Path not found: {base}"
    lines: list[str] = []
    try:
        regex = re.compile(pattern)
    except re.error as error:
        return f"Invalid regex: {error}"
    paths = [base] if base.is_file() else sorted(base.rglob("*"))
    for path in paths:
        if path.is_dir():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                lines.append(f"{path}:{line_number}:{line}")
                if len(lines) >= 200:
                    lines.append("...[too many matches, truncated]")
                    return "\n".join(lines)
    return "\n".join(lines) if lines else "(no matches)"


def _bash(
    arguments: dict[str, Any],
    workdir: str,
    *,
    enabled: bool,
    timeout: float,
    output_cap: int,
) -> str:
    if not enabled:
        return "bash is disabled on this gateway"
    command = str(arguments.get("command") or "").strip()
    if not command:
        return "bash requires a 'command' argument"
    logger.info("tool bash workdir=%s command=%s", workdir, command)
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout:g}s"
    except OSError as error:
        return f"Failed to run command: {error}"
    output = completed.stdout or ""
    if completed.stderr:
        output += "\n[stderr]\n" + completed.stderr
    if completed.returncode != 0:
        output += f"\n[exit code {completed.returncode}]"
    if len(output) > output_cap:
        output = output[:output_cap] + f"\n...[truncated, {len(output) - output_cap} chars dropped]"
    return output


def _git(arguments: dict[str, Any], workdir: str, *, args: list[str]) -> str:
    path = str(arguments.get("path") or "").strip()
    cmd = ["git", *args] + ([path] if path and args == ["diff"] else [])
    try:
        completed = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        return f"git failed: {error}"
    output = completed.stdout.strip() or "(no output)"
    if completed.returncode != 0:
        output += f"\n[git exit {completed.returncode}] {completed.stderr.strip()}"
    return output[:50000]


def _base64(arguments: dict[str, Any], *, encode: bool) -> str:
    text = str(arguments.get("text") or "")
    if encode:
        return base64.b64encode(text.encode("utf-8")).decode("ascii")
    try:
        return base64.b64decode(text).decode("utf-8", errors="replace")
    except Exception as error:  # noqa: BLE001
        return f"base64 decode failed: {error}"


def _json_parse(arguments: dict[str, Any]) -> str:
    text = str(arguments.get("text") or "")
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError) as error:
        return f"Invalid JSON: {error}"
    return json.dumps(parsed, ensure_ascii=False, indent=2)[:50000]


async def _http_get(arguments: dict[str, Any]) -> str:
    url = str(arguments.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return "http_get requires an http(s) url"
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "ai-gateway-tool/1.0"},
            )
    except Exception as error:  # noqa: BLE001
        return f"http_get failed: {error}"
    text = resp.text or ""
    if len(text) > 50000:
        text = text[:50000] + "\n...[truncated]"
    return f"HTTP {resp.status_code} ({len(resp.text)} bytes)\n\n{text}"


async def _browser_dispatch(name: str, arguments: dict[str, Any]) -> str:
    from .browser_tools import (
        browser_click,
        browser_close,
        browser_eval,
        browser_navigate,
        browser_open,
        browser_screenshot,
        browser_snapshot,
        browser_type,
    )

    if name == "browser_open":
        return await browser_open(str(arguments.get("url") or ""))
    if name == "browser_navigate":
        return await browser_navigate(str(arguments.get("url") or ""))
    if name == "browser_snapshot":
        return await browser_snapshot(max_chars=50000)
    if name == "browser_click":
        return await browser_click(str(arguments.get("selector") or ""))
    if name == "browser_type":
        return await browser_type(str(arguments.get("selector") or ""), str(arguments.get("text") or ""))
    if name == "browser_eval":
        return await browser_eval(str(arguments.get("js") or ""))
    if name == "browser_screenshot":
        return await browser_screenshot(str(arguments.get("path") or ""))
    if name == "browser_close":
        return await browser_close()
    return f"Unknown browser tool: {name}"


async def _figma_dispatch(
    name: str,
    arguments: dict[str, Any],
    *,
    project_key: str = "",
) -> str:
    from .figma import figma_export_image, figma_get_file, figma_get_node

    file_key = str(arguments.get("file_key") or "")
    if name == "figma_get_file":
        return await figma_get_file(file_key, project_key=project_key)
    if name == "figma_get_node":
        return await figma_get_node(
            file_key,
            str(arguments.get("node_id") or ""),
            project_key=project_key,
        )
    if name == "figma_export_image":
        return await figma_export_image(
            file_key,
            str(arguments.get("node_id") or ""),
            str(arguments.get("format") or "png"),
            project_key=project_key,
        )
    return f"Unknown figma tool: {name}"
