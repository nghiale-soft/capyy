from __future__ import annotations

"""Scan local chat history from AI tools and import it into the history store.

Supported sources (defaults to the gateway user's home):
- Claude Code: ``~/.claude/projects/**/*.jsonl`` (carries ``cwd``)
- Codex CLI: ``~/.codex/sessions/**/*.jsonl`` (``session_meta`` carries ``cwd``)

Paths can be overridden with the ``SCAN_CLAUDE_DIR`` / ``SCAN_CODEX_DIR`` env
vars (handy for tests and for different volume mounts in Docker).

Scanning is incremental: each imported row is tagged ``meta.session_id`` and a
fingerprint. Later scans import only rows appended to an active session.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any

from .chat_history import _import_record_fingerprint
from .history_import import claude_code_session, codex_session

logger = logging.getLogger("gateway.services.history_scan")

# Per-process source-file state. Most session files are immutable after their
# chat ends; avoiding their JSON parse on every dashboard refresh keeps active
# sync cheap even when a user has years of local history.
_SCANNED_FILE_STATE: dict[str, tuple[int, int]] = {}


def _changed_since_last_scan(path: Path) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    state = (stat.st_mtime_ns, stat.st_size)
    key = str(path)
    if _SCANNED_FILE_STATE.get(key) == state:
        return False
    _SCANNED_FILE_STATE[key] = state
    return True


def _claude_root() -> Path:
    return Path(os.getenv("SCAN_CLAUDE_DIR") or Path.home() / ".claude" / "projects")


def _codex_root() -> Path:
    return Path(os.getenv("SCAN_CODEX_DIR") or Path.home() / ".codex" / "sessions")


def _session_id_from_file(source: str, path: Path) -> str:
    """Stable session_id = source + file name (without extension)."""
    return f"{source}:{path.stem}"


def _import_session(
    service: Any,
    source: str,
    path: Path,
    records: list[dict[str, Any]],
    cwd: str | None,
    existing: dict[str, dict[str, set[str]]],
    summary: dict[str, Any],
) -> None:
    if not records:
        summary["files_empty"] += 1
        return
    key = service.resolve_path(cwd) if cwd else "default"
    session_id = _session_id_from_file(source, path)
    sessions = existing.setdefault(key, service.imported_session_fingerprints(key))
    seen = sessions.setdefault(session_id, set())
    for record in records:
        record["meta"] = {
            "source": source,
            "session_id": session_id,
            "file": str(path),
        }
    # Compare after metadata has been attached, matching the persisted shape.
    # `seen` is updated in-memory, so each scan reads the history file once per
    # session rather than once per record.
    pending = [
        record for record in records
        if _import_record_fingerprint(record) not in seen
    ]
    if not pending:
        summary["files_skipped"] += 1
        return
    written = service.append_records(key, pending)
    if written:
        seen.update(_import_record_fingerprint(record) for record in pending)
        summary["files_imported"] += 1
        summary["records_imported"] += written
        summary["projects"].add(key)
    else:
        summary["files_empty"] += 1


def _scan_claude_code(
    service: Any, summary: dict[str, Any], existing: dict[str, dict[str, set[str]]]
) -> None:
    root = _claude_root()
    files = sorted(root.glob("*/*.jsonl")) if root.is_dir() else []
    summary["files_found"] = len(files)
    for path in files:
        if not _changed_since_last_scan(path):
            summary["files_skipped"] += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            logger.warning("cannot read claude session %s: %s", path, error)
            summary["files_skipped"] += 1
            continue
        records, cwd = claude_code_session(text)
        _import_session(service, "claude", path, records, cwd, existing, summary)


def _scan_codex(
    service: Any, summary: dict[str, Any], existing: dict[str, dict[str, set[str]]]
) -> None:
    root = _codex_root()
    files = sorted(root.rglob("*.jsonl")) if root.is_dir() else []
    summary["files_found"] = len(files)
    for path in files:
        if not _changed_since_last_scan(path):
            summary["files_skipped"] += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            logger.warning("cannot read codex session %s: %s", path, error)
            summary["files_skipped"] += 1
            continue
        records, cwd = codex_session(text)
        _import_session(service, "codex", path, records, cwd, existing, summary)


def scan_local_history(
    service: Any,
    *,
    sources: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Quét toàn bộ nguồn lịch sử local, trả về tóm tắt kết quả."""
    sources = sources or ("claude_code", "codex")
    existing: dict[str, dict[str, set[str]]] = {}
    results: list[dict[str, Any]] = []
    total_records = 0
    for source in sources:
        summary = {
            "source": source,
            "files_found": 0,
            "files_imported": 0,
            "files_skipped": 0,
            "files_empty": 0,
            "records_imported": 0,
            "projects": set(),
        }
        try:
            if source == "claude_code":
                _scan_claude_code(service, summary, existing)
            elif source == "codex":
                _scan_codex(service, summary, existing)
            else:
                logger.warning("unknown scan source: %s", source)
                continue
        except Exception:
            logger.exception("scan source %s failed", source)
            summary["error"] = "failed"
        summary["projects"] = sorted(summary["projects"])
        total_records += summary["records_imported"]
        results.append(summary)
    return {
        "ok": True,
        "scanned_at": int(time.time() * 1000),
        "records_imported": total_records,
        "sources": results,
    }
