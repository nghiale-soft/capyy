from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from gateway.core.config import Settings

logger = logging.getLogger("gateway.services.chat_history")

# Optional headers clients can use to declare the project path when it cannot be inferred.
PROJECT_PATH_HEADER = "x-project-path"
PROJECT_ID_HEADER = "x-project-id"
SESSION_ID_HEADERS = ("x-session-id", "x-conversation-id")

# Keywords suggesting the user is asking about a past conversation.
_MEMORY_PATTERNS = (
    r"bạn có nhớ",
    r"có nhớ",
    r"nhớ (không|ko|k)?",
    r"lần trước",
    r"lần trước (đó|đấy|kia)?",
    r"trước (đây|kia|đó)?",
    r"hôm trước",
    r"hôm qua",
    r"đợt trước",
    r"earlier",
    r"before",
    r"previous",
    r"last time",
    r"remember",
    r"trước đó",
    r"lúc nãy",
    r"ban nãy",
)

_MEMORY_RE = re.compile("|".join(_MEMORY_PATTERNS), re.IGNORECASE)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "default"


# AI-tool wrapper blocks that pollute user messages (Claude Code, Codex, IDE
# context) — stripped when generating titles / previews so the UI shows the
# real question instead of <system-reminder> / <ide_opened_file> noise.
_TAG_BLOCK_RE = re.compile(
    r"<(system-reminder|ide_opened_file|local-command-caveat|retrieval_status|recommended_plugins|environment_context)\b[^>]*>.*?</\1>",
    re.S | re.IGNORECASE,
)
# Unclosed tag that eats everything after it (e.g. <codex_internal_context ...>).
_OPEN_TAG_TAIL_RE = re.compile(
    r"<codex_internal_context\b[^>]*>.*$", re.S | re.IGNORECASE
)
# Labels after which the message is machine-generated, not the user's words.
_TOOL_PREFIXES = ("[tool_result]", "[tool error]", "[Request interrupted")


def _clean_message_text(text: str) -> str:
    """Strip AI-tool wrappers so titles/previews show the real user question.

    Removes <system-reminder>...</system-reminder>, <ide_opened_file>...,
    <codex_internal_context> tails, [tool_result]/[tool error] labels, then
    collapses whitespace to a single line.
    """
    if not text:
        return ""
    cleaned = _TAG_BLOCK_RE.sub(" ", text)
    cleaned = _OPEN_TAG_TAIL_RE.sub("", cleaned)
    for prefix in _TOOL_PREFIXES:
        index = cleaned.find(prefix)
        if index != -1:
            cleaned = cleaned[:index]
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _import_record_fingerprint(record: dict[str, Any]) -> str:
    """Fingerprint the meaningful immutable fields of one imported row."""
    return json.dumps(
        {
            "ts": int(record.get("ts") or 0),
            "role": record.get("role"),
            "content": _content_to_text(record.get("content")),
            "thinking": _content_to_text(record.get("thinking")),
            "tool_calls": _normalize_tool_calls(record.get("tool_calls")),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class ChatHistoryService:
    """Stores per-project chat history for recall when the user asks about the past.

    Design:
    - One JSONL file per conversation: ``<history_dir>/projects/<project>/sessions/<session>.jsonl``.
    - Stable key based on the **git remote URL** when present (survives folder
      rename/move), otherwise the folder name. ``projects.json`` keeps the
      mapping from key to every path/name seen, so renamed folders are recognized.
    - Each JSONL line: ``{ts, role, content, model}``.
    - Rows older than ``max_age_days`` (default 365 days) are pruned.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = Path(settings.history_dir)
        # ``chats`` is kept only as a read-only legacy location. New writes are
        # split by project and conversation so a request never scans an entire
        # project's transcript just to recover the current chat context.
        self.chats_dir = self.root / "chats"
        self.projects_dir = self.root / "projects"
        self.projects_file = self.root / "projects.json"
        self.conversation_index_file = self.root / "conversations.json"
        self.max_age_ms = settings.history_max_age_days * 86_400_000
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self._conversation_cache: list[dict[str, Any]] = self._load_conversation_index()

    def _load_conversation_index(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.conversation_index_file.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    # ------------------------------------------------------------------
    # Project identity
    # ------------------------------------------------------------------

    def _load_projects(self) -> dict[str, Any]:
        if not self.projects_file.exists():
            return {}
        try:
            data = json.loads(self.projects_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("failed to read projects index: %s", error)
            return {}
        return data if isinstance(data, dict) else {}

    def _save_projects(self, projects: dict[str, Any]) -> None:
        try:
            self.projects_file.write_text(
                json.dumps(projects, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            logger.warning("failed to write projects index: %s", error)

    @staticmethod
    def _git_remote(path: Path) -> str:
        git_dir = path / ".git"
        config = git_dir / "config"
        if git_dir.is_dir() and config.is_file():
            try:
                text = config.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
            for match in re.finditer(
                r'\[remote "([^"]+)"\][^\[]*?url\s*=\s*(\S+)',
                text,
                re.IGNORECASE,
            ):
                if match.group(1).lower() == "origin":
                    return match.group(2).strip().rstrip("/")
        return ""

    def _identity_for_path(self, raw_path: str) -> tuple[str, dict[str, Any]]:
        """Return (key, metadata) for a project path.

        Prefers the git remote URL; otherwise uses the folder name. Always
        records the path so renames/moves of the storage location are detected.
        """
        path = Path(raw_path).expanduser()
        remote = self._git_remote(path)
        folder = path.name or "default"
        key = _sanitize_filename(remote) if remote else _sanitize_filename(folder)
        metadata = {
            "folder": folder,
            "path": str(path),
        }
        if remote:
            metadata["remote"] = remote
        return key, metadata

    def resolve_path(self, raw_path: str) -> str:
        """Public API: resolve and register a project key from a path."""
        key, metadata = self._identity_for_path(str(raw_path))
        return self._register_project(key, metadata)

    def resolve_project(self, request: Any, body: dict[str, Any]) -> str:
        """Resolve the project key from a request.

        Order: header ``x-project-path`` -> body ``metadata`` -> fallback
        ``default``. If the path was renamed/moved but is the same repo, the old
        key is reused via the ``projects.json`` index.
        """
        header_path = (
            request.headers.get(PROJECT_PATH_HEADER)
            if request is not None
            else None
        )
        header_id = (
            request.headers.get(PROJECT_ID_HEADER)
            if request is not None
            else None
        )
        if header_id:
            return _sanitize_filename(header_id)

        raw_path = header_path
        if not raw_path and isinstance(body, dict):
            metadata = body.get("metadata")
            if isinstance(metadata, dict):
                raw_path = (
                    metadata.get("cwd")
                    or metadata.get("project_path")
                    or metadata.get("projectPath")
                    or metadata.get("workspace")
                )

        if not raw_path:
            return "default"

        key, metadata = self._identity_for_path(str(raw_path))
        return self._register_project(key, metadata)

    def resolve_session(self, request: Any, body: dict[str, Any]) -> str:
        """Return the client conversation id; this is never an API token.

        Claude/Codex integrations can send either ``x-session-id`` or
        ``metadata.session_id``.  Bare API callers do not have a durable chat
        id, so they intentionally share the small ``gateway`` session.
        """
        if request is not None:
            for header in SESSION_ID_HEADERS:
                value = request.headers.get(header)
                if value:
                    return _sanitize_filename(str(value))
        metadata = body.get("metadata") if isinstance(body, dict) else None
        if isinstance(metadata, dict):
            for key in ("session_id", "sessionId", "conversation_id", "conversationId"):
                value = metadata.get(key)
                if value:
                    return _sanitize_filename(str(value))
        return "gateway"

    def _register_project(self, key: str, metadata: dict[str, Any]) -> str:
        """Update the projects.json index and return the stable key.

        If the folder was renamed but the git remote matches, merge into the old
        key. If the generated key differs but the folder name matches an existing
        project (parent path changed), reuse the old key.
        """
        projects = self._load_projects()
        folder = metadata.get("folder", "")
        remote = metadata.get("remote", "")
        path = metadata.get("path", "")

        # Git remote is the strongest identity: if the new key has a remote and
        # another project already has the same remote -> merge into the old key.
        if remote:
            for existing_key, existing in projects.items():
                if (
                    existing.get("remote") == remote
                    and existing_key != key
                ):
                    existing.setdefault("paths", [])
                    if path and path not in existing["paths"]:
                        existing["paths"].append(path)
                    self._save_projects(projects)
                    return existing_key

        # No remote: folder moved to a different parent (same name) -> reuse.
        if not remote and folder:
            for existing_key, existing in projects.items():
                if (
                    not existing.get("remote")
                    and existing.get("folder") == folder
                    and existing_key != key
                ):
                    existing.setdefault("paths", [])
                    if path and path not in existing["paths"]:
                        existing["paths"].append(path)
                    self._save_projects(projects)
                    return existing_key

        # Path seen before (project moved back, or restored from backup) -> reuse
        # the key of the project that owns that path.
        if path:
            for existing_key, existing in projects.items():
                if existing_key == key:
                    continue
                if path in existing.get("paths") or []:
                    existing.setdefault("paths", [])
                    if path not in existing["paths"]:
                        existing["paths"].append(path)
                    self._save_projects(projects)
                    return existing_key

        entry = projects.setdefault(
            key,
            {"folder": folder, "paths": [], "first_seen": _now_ms()},
        )
        if remote:
            entry["remote"] = remote
        if path and path not in entry.setdefault("paths", []):
            entry["paths"].append(path)
        entry["last_seen"] = _now_ms()
        self._save_projects(projects)
        return key

    # ------------------------------------------------------------------
    # Store / read history
    # ------------------------------------------------------------------

    def _project_dir(self, project_key: str) -> Path:
        return self.projects_dir / _sanitize_filename(project_key)

    def _chat_file(self, project_key: str, session_id: str = "gateway") -> Path:
        path = self._project_dir(project_key) / "sessions" / f"{_sanitize_filename(session_id)}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _legacy_chat_file(self, project_key: str) -> Path:
        return self.chats_dir / f"{_sanitize_filename(project_key)}.jsonl"

    @staticmethod
    def _session_id(meta: dict[str, Any] | None) -> str:
        return _sanitize_filename(str((meta or {}).get("session_id") or "gateway"))

    def _session_files(self, project_key: str) -> list[Path]:
        directory = self._project_dir(project_key) / "sessions"
        try:
            return list(directory.glob("*.jsonl"))
        except OSError:
            return []

    def record(
        self,
        project_key: str,
        *,
        role: str,
        content: Any,
        model: str = "",
        thinking: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Append one history row; supports thinking and tool_calls."""
        text = _content_to_text(content)
        calls = _normalize_tool_calls(tool_calls)
        thinking_text = _content_to_text(thinking)
        # Keep records that carry thinking/tool_calls even with empty content
        # (e.g. Claude Code sends assistant messages with only a thinking block).
        if not text and not calls and not thinking_text:
            return
        # Dedupe (text only): skip if the last row in the file is identical
        # (client retries the same question -> avoid duplicates). Tool/thinking-only
        # records are always written.
        session_id = self._session_id(meta)
        if text and self._last_recorded(project_key, session_id) == (role, text):
            return
        line = {
            "ts": _now_ms(),
            "role": role,
            "content": text,
            "model": model,
        }
        if thinking_text:
            line["thinking"] = thinking_text
        if calls:
            line["tool_calls"] = calls
        if meta:
            line["meta"] = meta
        try:
            path = self._chat_file(project_key, session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError as error:
            logger.warning("failed to record chat history: %s", error)
            return
        # Pruning is performed by the history sync task, not on every turn.
        # Rewriting a 100 MB transcript here was the main request-path stall.

    def _last_recorded(self, project_key: str, session_id: str = "gateway") -> tuple[str, str] | None:
        """Return (role, content) of the last row, None if the file is empty."""
        path = self._chat_file(project_key, session_id)
        if not path.exists():
            return None
        try:
            rows = _tail_jsonl(path, 1)
            if rows:
                return (rows[-1].get("role"), str(rows[-1].get("content") or ""))
        except OSError:
            pass
        return None

    # ------------------------------------------------------------------
    # API/UI: list projects + read messages + delete project
    # ------------------------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        """List projects with light stats (row count, most recent activity)."""
        projects = self._load_projects()
        keys = set(projects.keys())
        try:
            keys.update(path.stem for path in self.chats_dir.glob("*.jsonl"))
            keys.update(path.name for path in self.projects_dir.iterdir() if path.is_dir())
        except OSError:
            pass
        result: list[dict[str, Any]] = []
        for key in sorted(keys):
            entry = projects.get(key, {})
            count = 0
            last_ts = 0
            last_content = ""
            last_user_ts = 0
            last_user_content = ""
            first_ts = 0
            title = ""
            sources: set[str] = set()
            providers: set[str] = set()
            paths = self._session_files(key)
            # Legacy files remain visible in the dashboard until their next
            # background import; they are deliberately never used for request
            # context, where their size previously caused multi-second stalls.
            if not paths and self._legacy_chat_file(key).exists():
                paths = [self._legacy_chat_file(key)]
            for path in paths:
                for row in _read_jsonl(path):
                    count += 1
                    ts = int(row.get("ts") or 0)
                    content = str(row.get("content") or "")
                    if ts >= last_ts:
                        last_ts = ts
                        if content:
                            last_content = content[:200]
                    role = row.get("role")
                    if role == "user" and content:
                        clean = _clean_message_text(content)
                        if ts >= last_user_ts:
                            last_user_ts = ts
                            last_user_content = _truncate(clean, 160)
                        if not title and clean:
                            title = _truncate(clean, 72)
                        if not first_ts:
                            first_ts = ts
                    meta = row.get("meta")
                    if isinstance(meta, dict) and meta.get("source"):
                        sources.add(str(meta["source"]))
                    if isinstance(meta, dict) and meta.get("provider"):
                        providers.add(str(meta["provider"]))
            result.append(
                {
                    "key": key,
                    "folder": entry.get("folder") or key,
                    "paths": entry.get("paths") or [],
                    "remote": entry.get("remote"),
                    "count": count,
                    "title": title or entry.get("folder") or key,
                    "first_ts": first_ts,
                    "last_ts": last_ts,
                    "last_content": last_content,
                    "last_user_ts": last_user_ts,
                    "last_user_content": last_user_content,
                    "sources": sorted(sources),
                    "providers": sorted(providers),
                }
            )
        return result

    def messages(
        self,
        project_key: str,
        *,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        newest_first_page: bool = False,
    ) -> list[dict[str, Any]]:
        """Read project messages, optionally paging from the newest records."""
        cutoff = _now_ms() - self.max_age_ms
        rows: list[dict[str, Any]] = []
        paths = [self._chat_file(project_key, session_id)] if session_id else self._session_files(project_key)
        if not paths and self._legacy_chat_file(project_key).exists():
            paths = [self._legacy_chat_file(project_key)]
        for path in paths:
            rows.extend(row for row in _read_jsonl(path) if int(row.get("ts") or 0) >= cutoff)
        # External sessions are imported file-by-file, so append order is not
        # necessarily chronological across sessions. The dashboard must always
        # render the actual newest conversation at its bottom.
        rows.sort(key=lambda row: int(row.get("ts") or 0))
        if newest_first_page:
            end = max(len(rows) - offset, 0)
            start = max(end - limit, 0)
            return rows[start:end]
        return rows[offset:offset + limit]

    def sessions(self, project_key: str) -> list[dict[str, Any]]:
        """Summarize independent Claude/Codex sessions within one project."""
        grouped: dict[str, dict[str, Any]] = {}
        for path in self._session_files(project_key):
            session_id = path.stem
            for row in self.messages(project_key, session_id=session_id, limit=1_000_000):
                meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
                item = grouped.setdefault(session_id, {"id": session_id, "count": 0, "first_ts": 0, "last_ts": 0, "title": "", "source": meta.get("source") or "gateway"})
                item["count"] += 1
                ts = int(row.get("ts") or 0)
                if not item["first_ts"] or ts < item["first_ts"]:
                    item["first_ts"] = ts
                item["last_ts"] = max(item["last_ts"], ts)
                if not item["title"] and row.get("role") == "user":
                    title = _clean_message_text(str(row.get("content") or ""))
                    if title:
                        item["title"] = _truncate(title, 88)
        if not grouped and self._legacy_chat_file(project_key).exists():
            for row in self.messages(project_key, limit=1_000_000):
                meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
                session_id = self._session_id(meta)
                item = grouped.setdefault(session_id, {"id": session_id, "count": 0, "first_ts": 0, "last_ts": 0, "title": "", "source": meta.get("source") or "gateway"})
                item["count"] += 1
                ts = int(row.get("ts") or 0)
                item["first_ts"] = ts if not item["first_ts"] else min(item["first_ts"], ts)
                item["last_ts"] = max(item["last_ts"], ts)
                if not item["title"] and row.get("role") == "user":
                    item["title"] = _truncate(_clean_message_text(str(row.get("content") or "")), 88)
        return sorted(grouped.values(), key=lambda item: item["last_ts"], reverse=True)

    def conversations(self) -> list[dict[str, Any]]:
        """Return the cached lightweight conversation index immediately."""
        return list(self._conversation_cache)

    def refresh_conversation_index(self) -> None:
        """Rebuild the expensive preview index off the request path."""
        self._conversation_cache = self._build_conversations()
        try:
            self.conversation_index_file.write_text(
                json.dumps(self._conversation_cache, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as error:
            logger.warning("failed to persist conversation index: %s", error)

    def _build_conversations(self) -> list[dict[str, Any]]:
        """Flat conversation list; project is metadata, not a UI container."""
        projects = {item["key"]: item for item in self.list_projects()}
        result: list[dict[str, Any]] = []
        for project_key, project in projects.items():
            for session in self.sessions(project_key):
                result.append({
                    **session,
                    "project_key": project_key,
                    "project": project.get("folder") or project_key,
                    "paths": project.get("paths") or [],
                })
        return sorted(result, key=lambda item: item["last_ts"], reverse=True)

    def delete_project(self, project_key: str) -> None:
        """Delete the history file and its entry in projects.json."""
        import shutil
        shutil.rmtree(self._project_dir(project_key), ignore_errors=True)
        self._legacy_chat_file(project_key).unlink(missing_ok=True)
        projects = self._load_projects()
        if project_key in projects:
            del projects[project_key]
            self._save_projects(projects)

    def append_records(
        self,
        project_key: str,
        records: list[dict[str, Any]],
    ) -> int:
        """Write converted rows directly (used by the local-history scan).

        Keeps original conversation timestamps, dedupes against the last row,
        then prunes by max_age_days.
        """
        if not records:
            return 0
        cutoff = _now_ms() - self.max_age_ms
        last_by_session: dict[str, tuple[str, str] | None] = {}
        lines_by_session: dict[str, list[str]] = {}
        written = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            role = record.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = _content_to_text(record.get("content"))
            calls = _normalize_tool_calls(record.get("tool_calls"))
            thinking = _content_to_text(record.get("thinking"))
            # Keep records with thinking/tool_calls even when content is empty.
            if not text and not calls and not thinking:
                continue
            # Only import records inside the retention window (365 days) so the
            # returned count matches the rows actually written.
            ts = int(record.get("ts") or _now_ms())
            if ts < cutoff:
                continue
            # Dedupe retries (same role+text); tool/thinking-only rows are kept.
            meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
            session_id = self._session_id(meta)
            last = last_by_session.setdefault(session_id, self._last_recorded(project_key, session_id))
            if text and last == (role, text):
                continue
            line: dict[str, Any] = {
                "ts": ts,
                "role": role,
                "content": text,
                "model": record.get("model") or "imported",
            }
            if thinking:
                line["thinking"] = thinking
            if calls:
                line["tool_calls"] = calls
            if isinstance(meta, dict) and meta:
                line["meta"] = meta
            lines_by_session.setdefault(session_id, []).append(json.dumps(line, ensure_ascii=False))
            last_by_session[session_id] = (role, text)
            written += 1
        if not lines_by_session:
            return 0
        try:
            for session_id, lines in lines_by_session.items():
                with self._chat_file(project_key, session_id).open("a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
        except OSError as error:
            logger.warning("failed to import chat history: %s", error)
            return 0
        return written

    def session_ids(self, project_key: str) -> set[str]:
        """Imported session_ids (from meta.session_id) — used for idempotent scans."""
        ids: set[str] = set()
        for path in self._session_files(project_key):
            for row in _read_jsonl(path):
                meta = row.get("meta")
                if isinstance(meta, dict) and meta.get("session_id"):
                    ids.add(str(meta["session_id"]))
        return ids

    def imported_record_fingerprints(
        self, project_key: str, session_id: str
    ) -> set[str]:
        """Return stable fingerprints already imported from one local session.

        Claude Code and Codex append to an existing JSONL session throughout a
        chat.  A session-level ``seen`` marker therefore cannot be used to skip
        the file forever: a later scan must import just its newly appended rows.
        """
        return self.imported_session_fingerprints(project_key).get(session_id, set())

    def imported_session_fingerprints(self, project_key: str) -> dict[str, set[str]]:
        """Return imported fingerprints grouped by session in one file pass."""
        sessions: dict[str, set[str]] = {}
        for path in self._session_files(project_key):
            for row in _read_jsonl(path):
                meta = row.get("meta")
                session_id = str(meta.get("session_id")) if isinstance(meta, dict) and meta.get("session_id") else path.stem
                sessions.setdefault(session_id, set()).add(_import_record_fingerprint(row))
        return sessions

    def record_messages(
        self,
        project_key: str,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Record only the newest user message from the payload.

        Clients (Claude Code, Cursor...) send the FULL history with every
        request; recording all of it would balloon the file exponentially and
        `recent()` would return mostly duplicate rows. Only the last user
        message (the new question) is recorded; the assistant reply is recorded
        through the ``on_assistant`` callback after the stream/completion ends.
        """
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                self.record(
                    project_key,
                    role="user",
                    content=message.get("content"),
                    model=model,
                    meta=meta,
                )
                return

    def recent(
        self,
        project_key: str,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return a conversation tail, or aggregate only for dashboard/import callers."""
        cutoff = _now_ms() - self.max_age_ms
        if session_id is not None:
            path = self._chat_file(project_key, session_id)
            return [row for row in _tail_jsonl(path, limit) if int(row.get("ts") or 0) >= cutoff]
        rows = self.messages(project_key, limit=1_000_000)
        return [row for row in rows if int(row.get("ts") or 0) >= cutoff][-limit:]

    def prune(self, project_key: str, session_id: str = "gateway") -> None:
        """Remove rows older than max_age_days; delete the file if empty."""
        path = self._chat_file(project_key, session_id)
        if not path.exists():
            return
        cutoff = _now_ms() - self.max_age_ms
        try:
            kept: list[str] = []
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if int(row.get("ts") or 0) >= cutoff:
                    kept.append(raw)
            if kept:
                path.write_text("\n".join(kept) + "\n", encoding="utf-8")
            else:
                path.unlink(missing_ok=True)
        except OSError as error:
            logger.warning("failed to prune chat history: %s", error)

    # ------------------------------------------------------------------
    # Context injection (answers like "do you remember ... last time")
    # ------------------------------------------------------------------

    def build_context(
        self,
        project_key: str,
        question: str,
        *,
        session_id: str = "gateway",
        max_chars: int | None = None,
    ) -> str | None:
        """Build the history context snippet to inject into the payload.

        The current chat's bounded tail is always available. Other projects are
        never read on this request path.
        """
        if self.settings.history_inject_mode == "off":
            return None

        # `question` can be a str or a list of messages (routes pass body.messages
        # straight through) -> use the last user message as the question.
        question_text = _question_text(question)
        is_memory = bool(_MEMORY_RE.search(question_text))
        rows = self.recent(project_key, session_id=session_id, limit=50)
        if not rows:
            return None

        # Filter by specific keywords appearing in the question (e.g. "429", "428").
        keywords = _extract_keywords(question_text)
        relevant = rows
        if is_memory and keywords:
            filtered = [
                row
                for row in rows
                if any(
                    k in str(row.get("content", ""))
                    or any(
                        k in str((tc.get("function") or {}).get("name") or "")
                        for tc in row.get("tool_calls") or []
                    )
                    for k in keywords
                )
            ]
            if filtered:
                relevant = filtered[-20:]

        limit = max_chars or self.settings.history_context_max_chars
        rendered = _render_history(relevant, limit)
        if not rendered:
            return None
        return (
            "Below is the past conversation history with the user in this project "
            f"(use as context when recalling):\n{rendered}"
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        projects = self._load_projects()
        return {
            "projects": len(projects),
            "dir": str(self.root),
            "max_age_days": self.settings.history_max_age_days,
            "inject_mode": self.settings.history_inject_mode,
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a transcript for dashboard/import work, never the chat hot path."""
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError as error:
        logger.warning("failed to read chat history %s: %s", path, error)
    return rows


def _tail_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read the final JSONL records using bounded reverse file reads.

    A conversation can itself be large. Seeking from EOF avoids reparsing old
    turns when only the last 50 are required for the next model request.
    """
    if limit < 1 or not path.exists():
        return []
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            position = fh.tell()
            chunks: list[bytes] = []
            newlines = 0
            while position > 0 and newlines <= limit + 1:
                size = min(16_384, position)
                position -= size
                fh.seek(position)
                chunk = fh.read(size)
                chunks.append(chunk)
                newlines += chunk.count(b"\n")
    except OSError as error:
        logger.warning("failed to tail chat history %s: %s", path, error)
        return []
    lines = b"".join(reversed(chunks)).splitlines()
    rows: list[dict[str, Any]] = []
    for raw in lines[-limit:]:
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def inject_context(messages: list[dict[str, Any]], context: str) -> list[dict[str, Any]]:
    """Inject the history context into the first system message if present.

    Avoids creating a second system message (every system message gets a BUFFY
    identity prepended by ``normalize_chat_messages`` -> costs free quota).
    If there is no system message, prepend a new one.
    """
    if not context or not isinstance(messages, list):
        return messages
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            merged = dict(msg)
            merged["content"] = f"{content}\n\n{context}"
            return messages[:index] + [merged] + messages[index + 1:]
        if isinstance(content, list):
            merged = dict(msg)
            blocks = list(content)
            blocks.append({"type": "text", "text": context})
            merged["content"] = blocks
            return messages[:index] + [merged] + messages[index + 1:]
        break
    return [{"role": "system", "content": context}] + messages


def _content_to_text(content: Any) -> str:
    """Convert content (str / list of blocks / message dict) to plain text.

    Recognizes blocks: text, thinking, tool_result (Anthropic), and also
    OpenAI/Anthropic message dicts ({"role": ..., "content": ...}).
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = _block_to_text(item)
                if text:
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return _block_to_text(content)
    return str(content)


def _block_to_text(block: dict[str, Any]) -> str:
    """One content block -> text. Returns "" when unrecognized."""
    text = block.get("text")
    if isinstance(text, str):
        return text
    block_type = block.get("type")
    if block_type == "thinking":
        thinking = block.get("thinking")
        if isinstance(thinking, str) and thinking:
            return f"[thinking] {thinking}"
    if block_type == "tool_result":
        return _tool_result_to_text(block)
    # Message dict OpenAI/Anthropic: {"role": ..., "content": ...}
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _content_to_text(content)
    return ""


def _tool_result_to_text(block: dict[str, Any]) -> str:
    """Anthropic tool_result block -> labeled text."""
    content = block.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = _content_to_text(content)
    else:
        text = str(content)
    if block.get("is_error") is True:
        text = f"[tool error] {text}"
    return f"[tool_result] {text}" if text.strip() else ""


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Filter tool_calls into a clean dict list (drop junk entries)."""
    if not isinstance(tool_calls, list):
        return []
    calls = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = (call.get("function") or {}).get("name") or call.get("name")
        if not name:
            continue
        calls.append(call)
    return calls


def _question_text(question: Any) -> str:
    """Extract the question from a str or a messages list (routes pass body.messages).

    Prefers the last user message — the user's newest question.
    """
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        for message in reversed(question):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "user":
                return _content_to_text(message.get("content"))
        return ""
    return _content_to_text(question)


def _extract_keywords(question: str) -> list[str]:
    """Extract short numeric/special tokens (e.g. 429, session, freebuff...)."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]{1,31}", question or "")
    stopwords = {
        "bạn",
        "có",
        "nhớ",
        "không",
        "lần",
        "trước",
        "về",
        "của",
        "với",
        "mình",
        "tôi",
        "đây",
        "khi",
        "mà",
        "thì",
        "the",
        "and",
        "you",
        "that",
        "with",
        "this",
        "from",
    }
    return [t for t in tokens if t.lower() not in stopwords][:8]


def _render_history(rows: list[dict[str, Any]], max_chars: int) -> str:
    lines: list[str] = []
    for row in rows:
        role = "User" if row.get("role") == "user" else "Assistant"
        content = str(row.get("content") or "").strip()
        tool_names = [
            str((tc.get("function") or {}).get("name") or tc.get("name") or "tool")
            for tc in row.get("tool_calls") or []
            if isinstance(tc, dict)
        ]
        if not content and not tool_names:
            continue
        line = f"{role}: {content}" if content else f"{role}:"
        for name in tool_names:
            line += f" [tool_use:{name}]"
        lines.append(line)
    if not lines:
        return ""
    text = "\n".join(lines)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return "…" + text[-(max_chars - 1):]
