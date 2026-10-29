from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("gateway.services.image_store")

# IDs are deliberately unguessable and validated before touching the filesystem so a
# caller cannot use path traversal to read another session.
IMAGE_ID_RE = re.compile(r"^img_[0-9a-f]{32}$")
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_MAX_BYTES = 25 * 1024 * 1024


class ImageSessionError(Exception):
    """Base error for image session storage."""


class ImageSessionNotFound(ImageSessionError):
    """The requested image session does not exist (or the id is malformed)."""


class ImageSessionExpired(ImageSessionError):
    """The requested image session exists but its retention window elapsed."""


class ImageSessionTooLarge(ImageSessionError):
    """The image exceeds the configured per-file storage limit."""


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(
        path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    )


class ImageSessionStore:
    """File-backed store for translated images and their editable region state.

    One directory per image id contains the original image, every rendered version,
    and a ``state.json`` describing the current regions and version history.

    The interface is intentionally small so a production deployment can replace the
    implementation with object storage or a database without changing the image
    service or routes.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_bytes = max(1, int(max_bytes))
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------ helpers
    def _session_dir(self, image_id: str) -> Path:
        if not IMAGE_ID_RE.fullmatch(str(image_id)):
            raise ImageSessionNotFound(f"Invalid image id: {image_id!r}")
        return self.root / image_id

    def _state_path(self, image_id: str) -> Path:
        return self._session_dir(image_id) / "state.json"

    def _check_size(self, data: bytes) -> None:
        if len(data) > self.max_bytes:
            raise ImageSessionTooLarge(
                f"Image exceeds the {self.max_bytes} byte storage limit"
            )

    def lock(self, image_id: str) -> asyncio.Lock:
        """Return a per-session in-process lock used to serialize repairs."""
        if not IMAGE_ID_RE.fullmatch(str(image_id)):
            raise ImageSessionNotFound(f"Invalid image id: {image_id!r}")
        lock = self._locks.get(image_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[image_id] = lock
        return lock

    # ------------------------------------------------------------------ reads
    def get(self, image_id: str) -> dict[str, Any]:
        state_path = self._state_path(image_id)
        try:
            state = _read_json(state_path)
        except FileNotFoundError as exc:
            raise ImageSessionNotFound(f"Image session not found: {image_id}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ImageSessionNotFound(
                f"Image session is unreadable: {image_id}"
            ) from exc

        expires_at = float(state.get("expires_at") or 0)
        if expires_at and time.time() > expires_at:
            self.delete(image_id)
            raise ImageSessionExpired(f"Image session expired: {image_id}")
        return state

    def get_original(self, image_id: str) -> bytes:
        self.get(image_id)
        path = self._session_dir(image_id) / "original.jpg"
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise ImageSessionNotFound(f"Original image missing: {image_id}") from exc

    def get_version(self, image_id: str, version: int | None = None) -> dict[str, Any]:
        state = self.get(image_id)
        versions = state.get("versions") or []
        if version is None:
            version = int(state.get("version") or 0)
        for item in versions:
            if int(item.get("version") or 0) == int(version):
                return item
        raise ImageSessionNotFound(f"Version {version} not found for {image_id}")

    def get_version_bytes(self, image_id: str, version: int | None = None) -> bytes:
        item = self.get_version(image_id, version)
        path = self._session_dir(image_id) / str(item.get("file") or "")
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise ImageSessionNotFound(
                f"Rendered image missing for {image_id} version {item.get('version')}"
            ) from exc

    def find_version_by_idempotency_key(
        self, image_id: str, idempotency_key: str | None
    ) -> dict[str, Any] | None:
        if not idempotency_key:
            return None
        state = self.get(image_id)
        for item in state.get("versions") or []:
            if item.get("idempotency_key") == idempotency_key:
                return item
        return None

    # ------------------------------------------------------------------ writes
    def create(
        self,
        original_bytes: bytes,
        *,
        output_bytes: bytes,
        regions: list[dict[str, Any]],
        target_language: str,
        instruction: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        idempotency_key: str | None = None,
        qc_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._check_size(original_bytes)
        self._check_size(output_bytes)
        image_id = f"img_{uuid.uuid4().hex}"
        session_dir = self._session_dir(image_id)
        session_dir.mkdir(parents=True, exist_ok=False)

        now = time.time()
        _atomic_write(session_dir / "original.jpg", original_bytes)
        _atomic_write(session_dir / "output_v1.jpg", output_bytes)

        version_item = {
            "version": 1,
            "created_at": now,
            "instruction": instruction,
            "model": model,
            "provider": provider,
            "regions": regions,
            "file": "output_v1.jpg",
            "idempotency_key": idempotency_key,
            "qc_report": qc_report,
        }
        state = {
            "id": image_id,
            "version": 1,
            "created_at": now,
            "expires_at": now + self.ttl_seconds,
            "target_language": target_language,
            "instruction": instruction,
            "model": model,
            "provider": provider,
            "regions": regions,
            "versions": [version_item],
        }
        _write_json(session_dir / "state.json", state)
        logger.info("Created image session %s regions=%d", image_id, len(regions))
        return state

    def save_version(
        self,
        image_id: str,
        output_bytes: bytes,
        *,
        regions: list[dict[str, Any]],
        instruction: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        idempotency_key: str | None = None,
        qc_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._check_size(output_bytes)
        state = self.get(image_id)
        version = int(state.get("version") or 0) + 1
        session_dir = self._session_dir(image_id)
        file_name = f"output_v{version}.jpg"
        _atomic_write(session_dir / file_name, output_bytes)

        now = time.time()
        version_item = {
            "version": version,
            "created_at": now,
            "instruction": instruction,
            "model": model,
            "provider": provider,
            "regions": regions,
            "file": file_name,
            "idempotency_key": idempotency_key,
            "qc_report": qc_report,
        }
        state["version"] = version
        state["regions"] = regions
        # Sliding retention: a repaired session stays alive for a full TTL from
        # its last write, so a user can continue iterating.
        state["expires_at"] = now + self.ttl_seconds
        # Keep the original translation metadata at the session level. The
        # per-version row carries the repair instruction that produced it.
        state["versions"] = list(state.get("versions") or []) + [version_item]
        _write_json(session_dir / "state.json", state)
        logger.info("Stored image session %s version %d", image_id, version)
        return state

    def delete(self, image_id: str) -> bool:
        session_dir = self._session_dir(image_id)
        if not session_dir.exists():
            return False
        shutil.rmtree(session_dir, ignore_errors=True)
        self._locks.pop(image_id, None)
        logger.info("Deleted image session %s", image_id)
        return True

    def cleanup_expired(self) -> int:
        """Delete sessions whose retention window has elapsed. Returns removed count."""
        now = time.time()
        removed = 0
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return 0
        for entry in entries:
            if not entry.is_dir() or not IMAGE_ID_RE.fullmatch(entry.name):
                continue
            state_path = entry / "state.json"
            try:
                state = _read_json(state_path)
                expires_at = float(state.get("expires_at") or 0)
            except (OSError, json.JSONDecodeError):
                expires_at = 0
            if not expires_at or now > expires_at:
                shutil.rmtree(entry, ignore_errors=True)
                self._locks.pop(entry.name, None)
                removed += 1
        if removed:
            logger.info("Cleaned up %d expired image sessions", removed)
        return removed
