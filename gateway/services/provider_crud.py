from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


logger = logging.getLogger("gateway.services.provider_crud")

CONFIG_PATH_ENV = "AI_GATEWAY_PROVIDERS_FILE"
DEFAULT_CONFIG_PATH = "config/providers.json"

# Fields never returned through the API to avoid leakage
_SECRET_FIELDS = {"api_key"}


@dataclass
class ProviderConfig:
    id: str
    name: str
    # How this provider is reached: "url" (HTTP endpoint) or "command" (local CLI).
    source: str = "url"
    # For source="command": which CLI (freebuff / claude / codex / commandcode ...).
    command: str = ""
    # For source="url": API flavor label (openai-compatible / anthropic / ...).
    type: str = "openai-compatible"
    base_url: str = ""
    api_key: str | None = None
    models: list[str] = field(default_factory=list)
    enabled: bool = True
    default: bool = False
    # Failover priority: lower runs first. Persisted so the dashboard can reorder.
    priority: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> dict[str, Any]:
        """Return a dict with secrets masked."""
        data = asdict(self)
        for secret in _SECRET_FIELDS:
            if data.get(secret):
                data[secret] = "***"
        # Deprecated: routing is determined solely by enabled + priority.
        data.pop("default", None)
        return data


class ProviderCrudService:
    """Provider CRUD persisted in a JSON file.

    Thread-safe via a Lock. Each change rewrites the whole file for simplicity.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = Lock()
        self._path = Path(
            path or DEFAULT_CONFIG_PATH
        )
        self._providers: dict[str, ProviderConfig] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            items = raw.get("providers") if isinstance(raw, dict) else raw
            if not isinstance(items, list):
                return
            for item in items:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                cfg_type = str(item.get("type") or "openai-compatible")
                # Migrate legacy entries: type=freebuff implies a freebuff command.
                source = str(item.get("source") or ("command" if cfg_type == "freebuff" else "url"))
                command = str(item.get("command") or ("freebuff" if cfg_type == "freebuff" else ""))
                cfg = ProviderConfig(
                    id=str(item["id"]),
                    name=str(item.get("name") or item["id"]),
                    source=source,
                    command=command,
                    type=cfg_type,
                    base_url=str(item.get("base_url") or ""),
                    api_key=item.get("api_key"),
                    models=[str(m) for m in item.get("models") or []],
                    enabled=bool(item.get("enabled", True)),
                    default=bool(item.get("default", False)),
                    priority=int(item.get("priority") or 0),
                    extra=dict(item.get("extra") or {}),
                )
                # One model per provider: legacy freebuff entries used to store
                # several models — keep only the first so the dashboard shows
                # a single model (user picks a different one via a new provider).
                if (source == "command" and command == "freebuff") and len(cfg.models) > 1:
                    logger.info(
                        "migrating freebuff provider '%s': %s models -> 1",
                        cfg.id,
                        len(cfg.models),
                    )
                    cfg.models = cfg.models[:1]
                self._providers[cfg.id] = cfg
            if any(
                p.source == "command" and p.command == "freebuff" and len(p.models) > 1
                for p in self._providers.values()
            ):
                self._save()
        except (json.JSONDecodeError, OSError) as error:
            logger.warning("failed to load providers file: %s", error)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        items = [asdict(p) for p in self._providers.values()]
        payload = {"version": 1, "providers": items}
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list(self) -> list[ProviderConfig]:
        with self._lock:
            return list(self._providers.values())

    def reorder(self, order: list[str]) -> list[ProviderConfig]:
        """Set failover priority from an ordered list of provider ids.

        Priority = index in the list (0 = first / highest priority).
        """
        with self._lock:
            for index, provider_id in enumerate(order):
                cfg = self._providers.get(provider_id)
                if cfg is None:
                    raise KeyError(f"provider '{provider_id}' not found")
                cfg.priority = index
            self._save()
            return list(self._providers.values())

    def ordered(self) -> list[ProviderConfig]:
        """Providers sorted by failover priority (stable)."""
        with self._lock:
            return sorted(
                self._providers.values(),
                key=lambda p: p.priority,
            )

    def get(self, provider_id: str) -> ProviderConfig | None:
        with self._lock:
            return self._providers.get(provider_id)

    def create(self, cfg: ProviderConfig) -> ProviderConfig:
        with self._lock:
            if cfg.id in self._providers:
                raise ValueError(f"provider '{cfg.id}' already exists")
            cfg.default = False
            self._providers[cfg.id] = cfg
            self._save()
            return cfg

    def update(
        self,
        provider_id: str,
        changes: dict[str, Any],
    ) -> ProviderConfig:
        with self._lock:
            existing = self._providers.get(provider_id)
            if existing is None:
                raise KeyError(f"provider '{provider_id}' not found")
            for key, value in changes.items():
                if key == "id":
                    continue
                if hasattr(existing, key):
                    setattr(existing, key, value)
                else:
                    existing.extra[key] = value
            existing.default = False
            self._save()
            return existing

    def delete(self, provider_id: str) -> None:
        with self._lock:
            if provider_id not in self._providers:
                raise KeyError(f"provider '{provider_id}' not found")
            del self._providers[provider_id]
            self._save()

    def default_provider(self) -> ProviderConfig | None:
        with self._lock:
            for p in sorted(self._providers.values(), key=lambda p: p.priority):
                if p.enabled:
                    return p
            return None
