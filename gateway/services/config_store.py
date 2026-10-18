from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pymysql
from pymysql.connections import Connection


logger = logging.getLogger("gateway.services.config_store")


class MariaDBConfigStore:
    """Store Capyy configuration in domain-specific relational tables."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_]+", database):
            raise ValueError("CAPYY_DB_NAME may contain only letters, digits, and underscores")
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

    def _connect(self, database: str | None = None) -> Connection:
        return pymysql.connect(
            host=self.host,
            port=self.port,
            database=database or self.database,
            user=self.user,
            password=self.password,
            charset="utf8mb4",
            connect_timeout=4,
            read_timeout=10,
            write_timeout=10,
            autocommit=False,
        )

    def initialize(self) -> None:
        with self._connect(database="mysql") as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            connection.commit()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS config_migrations (
                        migration_key VARCHAR(191) NOT NULL PRIMARY KEY,
                        completed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cursor.execute(
                    "ALTER TABLE config_migrations MODIFY migration_key VARCHAR(191) NOT NULL"
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS app_settings (
                        setting_key VARCHAR(191) NOT NULL PRIMARY KEY,
                        setting_value BIGINT NOT NULL DEFAULT 0
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cursor.execute(
                    """
                    ALTER TABLE app_settings MODIFY setting_key VARCHAR(191) NOT NULL
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ai_tokens (
                        token_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        provider_id VARCHAR(128) NOT NULL,
                        token_value TEXT NOT NULL,
                        position INT NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN NOT NULL DEFAULT FALSE,
                        UNIQUE KEY uq_ai_tokens_provider_position (provider_id, position)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS providers (
                        provider_id VARCHAR(128) NOT NULL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        source VARCHAR(32) NOT NULL DEFAULT 'url',
                        command VARCHAR(128) NOT NULL DEFAULT '',
                        provider_type VARCHAR(128) NOT NULL DEFAULT 'openai-compatible',
                        base_url TEXT NOT NULL,
                        enabled BOOLEAN NOT NULL DEFAULT TRUE,
                        priority INT NOT NULL DEFAULT 0,
                        extra JSON NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS provider_models (
                        provider_id VARCHAR(128) NOT NULL,
                        model_id VARCHAR(255) NOT NULL,
                        position INT NOT NULL,
                        PRIMARY KEY (provider_id, position),
                        CONSTRAINT fk_provider_models_provider
                            FOREIGN KEY (provider_id) REFERENCES providers(provider_id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tool_permissions (
                        tool_name VARCHAR(128) NOT NULL PRIMARY KEY,
                        permission_mode VARCHAR(16) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
            connection.commit()
        self._ensure_ai_token_columns()
        self._migrate_legacy_freebuff_tokens()
        self._migrate_provider_api_keys()
        self._migrate_legacy_config()
        self._sync_ai_token_active_flags()
        logger.info("Capyy relational config tables initialized schema=%s", self.database)

    def _ensure_ai_token_columns(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = 'ai_tokens'
                    """,
                    (self.database,),
                )
                columns = {str(row[0]) for row in cursor.fetchall()}
                if "created_at" not in columns:
                    cursor.execute(
                        """
                        ALTER TABLE ai_tokens ADD COLUMN created_at
                        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        """
                    )
                if "is_active" not in columns:
                    cursor.execute(
                        """
                        ALTER TABLE ai_tokens ADD COLUMN is_active
                        BOOLEAN NOT NULL DEFAULT FALSE
                        """
                    )
            connection.commit()

    def _sync_ai_token_active_flags(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT DISTINCT provider_id FROM ai_tokens")
                provider_ids = [str(row[0]) for row in cursor.fetchall()]
                for provider_id in provider_ids:
                    cursor.execute(
                        "SELECT setting_value FROM app_settings WHERE setting_key = %s",
                        (self._active_ai_token_setting(provider_id),),
                    )
                    active_row = cursor.fetchone()
                    active_position = int(active_row[0]) if active_row else 0
                    cursor.execute(
                        "SELECT COUNT(*) FROM ai_tokens WHERE provider_id = %s",
                        (provider_id,),
                    )
                    token_count = int(cursor.fetchone()[0])
                    if token_count:
                        active_position %= token_count
                        cursor.execute(
                            """
                            INSERT INTO app_settings (setting_key, setting_value)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value)
                            """,
                            (self._active_ai_token_setting(provider_id), active_position),
                        )
                    cursor.execute(
                        """
                        UPDATE ai_tokens
                        SET is_active = (position = %s)
                        WHERE provider_id = %s
                        """,
                        (active_position, provider_id),
                    )
            connection.commit()

    def _migrate_legacy_freebuff_tokens(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = 'freebuff_tokens'
                    """,
                    (self.database,),
                )
                if not cursor.fetchone()[0]:
                    return

                cursor.execute(
                    "CREATE TABLE IF NOT EXISTS freebuff_tokens_legacy_backup "
                    "LIKE freebuff_tokens"
                )
                cursor.execute(
                    "INSERT IGNORE INTO freebuff_tokens_legacy_backup "
                    "SELECT * FROM freebuff_tokens"
                )
                cursor.execute(
                    """
                    INSERT INTO ai_tokens
                        (provider_id, token_value, position, is_active)
                    SELECT 'freebuff', token_value, position,
                           (position = COALESCE(
                               (SELECT setting_value FROM app_settings
                                WHERE setting_key = 'active_token_index'),
                               0
                           ))
                    FROM freebuff_tokens
                    ON DUPLICATE KEY UPDATE token_value = VALUES(token_value)
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO app_settings (setting_key, setting_value)
                    SELECT 'active_ai_token_index:freebuff', setting_value
                    FROM app_settings WHERE setting_key = 'active_token_index'
                    ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value)
                    """
                )
                cursor.execute("DROP TABLE freebuff_tokens")
                cursor.execute(
                    """
                    INSERT IGNORE INTO config_migrations (migration_key)
                    VALUES ('ai_tokens:freebuff')
                    """
                )
            connection.commit()
        logger.info("migrated Freebuff credentials to ai_tokens")

    def _migrate_provider_api_keys(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = 'providers'
                      AND column_name = 'api_key'
                    """,
                    (self.database,),
                )
                if not cursor.fetchone()[0]:
                    return

                cursor.execute(
                    "CREATE TABLE IF NOT EXISTS providers_legacy_backup LIKE providers"
                )
                cursor.execute(
                    "INSERT IGNORE INTO providers_legacy_backup SELECT * FROM providers"
                )
                cursor.execute(
                    "SELECT provider_id, api_key FROM providers "
                    "WHERE api_key IS NOT NULL AND api_key <> ''"
                )
                provider_credentials = cursor.fetchall()
                for provider_id, api_key in provider_credentials:
                    cursor.execute(
                        """
                        SELECT COALESCE(MAX(position), -1),
                               COALESCE(MAX(is_active), 0)
                        FROM ai_tokens WHERE provider_id = %s
                        """,
                        (provider_id,),
                    )
                    max_position, has_active = cursor.fetchone()
                    cursor.execute(
                        """
                        SELECT 1 FROM ai_tokens
                        WHERE provider_id = %s AND token_value = %s LIMIT 1
                        """,
                        (provider_id, api_key),
                    )
                    if cursor.fetchone():
                        continue
                    position = int(max_position) + 1
                    is_active = not bool(has_active)
                    cursor.execute(
                        """
                        INSERT INTO ai_tokens
                            (provider_id, token_value, position, is_active)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (provider_id, api_key, position, is_active),
                    )
                    if is_active:
                        cursor.execute(
                            """
                            INSERT INTO app_settings (setting_key, setting_value)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value)
                            """,
                            (self._active_ai_token_setting(str(provider_id)), position),
                        )
                cursor.execute("ALTER TABLE providers DROP COLUMN api_key")
            connection.commit()
        logger.info("migrated provider API keys to ai_tokens")

    def _migrate_legacy_config(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = 'app_config'
                    """,
                    (self.database,),
                )
                if not cursor.fetchone()[0]:
                    return
                cursor.execute("SELECT setting_key, setting_value FROM app_config")
                rows = cursor.fetchall()
            connection.commit()

        legacy: dict[str, Any] = {}
        for key, value in rows:
            try:
                legacy[str(key)] = json.loads(value)
            except (TypeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Invalid legacy Capyy config entry: {key}") from error

        if "freebuff_tokens" in legacy:
            data = legacy["freebuff_tokens"]
            if isinstance(data, dict):
                self.save_ai_tokens(
                    "freebuff",
                    data.get("tokens", []),
                    int(data.get("active_index", 0)),
                )
                self.mark_migrated("ai_tokens:freebuff")
            self.mark_migrated("freebuff_tokens")
        if "providers" in legacy:
            data = legacy["providers"]
            items = data.get("providers", []) if isinstance(data, dict) else data
            provider_items = items if isinstance(items, list) else []
            credential_updates = {
                str(item["id"]): str(item["api_key"])
                for item in provider_items
                if isinstance(item, dict)
                and item.get("id")
                and item.get("api_key")
            }
            self.save_providers(
                provider_items,
                credential_updates=credential_updates,
            )
            self.mark_migrated("providers")
        if "tool_permissions" in legacy:
            data = legacy["tool_permissions"]
            self.save_tool_permissions(data if isinstance(data, dict) else {})
            self.mark_migrated("tool_permissions")

        known = {"freebuff_tokens", "providers", "tool_permissions"}
        if set(legacy).issubset(known):
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("DROP TABLE app_config")
                connection.commit()
            logger.info("migrated legacy app_config data into relational tables")
        else:
            logger.warning("kept legacy app_config table because it contains unknown settings")

    def has_migrated(self, key: str) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM config_migrations WHERE migration_key = %s",
                    (key,),
                )
                migrated = cursor.fetchone() is not None
            connection.commit()
        return migrated

    def mark_migrated(self, key: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT IGNORE INTO config_migrations (migration_key) VALUES (%s)",
                    (key,),
                )
            connection.commit()

    @staticmethod
    def _active_ai_token_setting(provider_id: str) -> str:
        normalized = provider_id.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("provider_id must contain 1 to 128 characters")
        return f"active_ai_token_index:{normalized}"

    @staticmethod
    def _ai_token_migration_key(provider_id: str) -> str:
        return f"ai_tokens:{provider_id.strip()}"

    def load_ai_tokens_or_migrate(
        self,
        provider_id: str,
        path: str | Path,
    ) -> tuple[list[str], int]:
        setting_key = self._active_ai_token_setting(provider_id)
        migration_key = self._ai_token_migration_key(provider_id)
        if not self.has_migrated(migration_key):
            source = Path(path)
            data: Any = None
            if source.exists():
                try:
                    data = json.loads(source.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise RuntimeError(f"Could not migrate {provider_id} token file") from error
            if isinstance(data, dict):
                tokens = data.get("tokens", [])
                active_index = data.get("active_index", 0)
            else:
                tokens, active_index = [], 0
            self.save_ai_tokens(
                provider_id,
                tokens if isinstance(tokens, list) else [],
                int(active_index or 0),
            )
            self.mark_migrated(migration_key)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT token_value FROM ai_tokens
                    WHERE provider_id = %s ORDER BY position
                    """,
                    (provider_id.strip(),),
                )
                tokens = [str(row[0]) for row in cursor.fetchall()]
                cursor.execute(
                    "SELECT setting_value FROM app_settings WHERE setting_key = %s",
                    (setting_key,),
                )
                row = cursor.fetchone()
            connection.commit()
        return tokens, int(row[0]) if row else 0

    def save_ai_tokens(
        self,
        provider_id: str,
        tokens: list[str] | tuple[str, ...],
        active_index: int,
    ) -> None:
        normalized_provider = provider_id.strip()
        setting_key = self._active_ai_token_setting(normalized_provider)
        normalized = [str(token).strip() for token in tokens if str(token).strip()]
        active = active_index % len(normalized) if normalized else 0
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT token_value, created_at FROM ai_tokens
                    WHERE provider_id = %s
                    """,
                    (normalized_provider,),
                )
                created_at_by_token = {
                    str(token): created_at
                    for token, created_at in cursor.fetchall()
                }
                cursor.execute(
                    "DELETE FROM ai_tokens WHERE provider_id = %s",
                    (normalized_provider,),
                )
                if normalized:
                    cursor.executemany(
                        """
                        INSERT INTO ai_tokens
                            (provider_id, token_value, position, created_at, is_active)
                        VALUES (%s, %s, %s, COALESCE(%s, CURRENT_TIMESTAMP), %s)
                        """,
                        [
                            (
                                normalized_provider,
                                token,
                                index,
                                created_at_by_token.get(token),
                                index == active,
                            )
                            for index, token in enumerate(normalized)
                        ],
                    )
                cursor.execute(
                    """
                    INSERT INTO app_settings (setting_key, setting_value)
                    VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value)
                    """,
                    (setting_key, active),
                )
            connection.commit()

    def load_providers_or_migrate(self, path: str | Path) -> list[dict[str, Any]]:
        if not self.has_migrated("providers"):
            data: Any = None
            source = Path(path)
            if source.exists():
                try:
                    data = json.loads(source.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise RuntimeError("Could not migrate provider config file") from error
            items = data.get("providers", []) if isinstance(data, dict) else data
            provider_items = items if isinstance(items, list) else []
            credential_updates = {
                str(item["id"]): str(item["api_key"])
                for item in provider_items
                if isinstance(item, dict)
                and item.get("id")
                and item.get("api_key")
            }
            self.save_providers(
                provider_items,
                credential_updates=credential_updates,
            )
            self.mark_migrated("providers")
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT provider_id, name, source, command, provider_type,
                           base_url, enabled, priority, extra
                    FROM providers ORDER BY priority, provider_id
                    """
                )
                rows = cursor.fetchall()
                result = []
                for row in rows:
                    cursor.execute(
                        """
                        SELECT model_id FROM provider_models
                        WHERE provider_id = %s ORDER BY position
                        """,
                        (row[0],),
                    )
                    models = [model[0] for model in cursor.fetchall()]
                    cursor.execute(
                        """
                        SELECT token_value FROM ai_tokens
                        WHERE provider_id = %s
                        ORDER BY is_active DESC, position
                        LIMIT 1
                        """,
                        (row[0],),
                    )
                    credential = cursor.fetchone()
                    result.append(
                        {
                            "id": row[0],
                            "name": row[1],
                            "source": row[2],
                            "command": row[3],
                            "type": row[4],
                            "base_url": row[5],
                            "api_key": str(credential[0]) if credential else None,
                            "enabled": bool(row[6]),
                            "priority": int(row[7]),
                            "extra": json.loads(row[8]) if isinstance(row[8], str) else row[8],
                            "models": models,
                        }
                    )
            connection.commit()
        return result

    def save_providers(
        self,
        providers: list[dict[str, Any]],
        credential_updates: dict[str, str] | None = None,
        credentials_to_delete: list[str] | None = None,
    ) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                for provider_id in credentials_to_delete or []:
                    cursor.execute(
                        "DELETE FROM ai_tokens WHERE provider_id = %s",
                        (provider_id,),
                    )
                cursor.execute("DELETE FROM providers")
                for provider in providers:
                    if not isinstance(provider, dict) or not provider.get("id"):
                        continue
                    provider_id = str(provider["id"])
                    cursor.execute(
                        """
                        INSERT INTO providers
                            (provider_id, name, source, command, provider_type,
                             base_url, enabled, priority, extra)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            provider_id,
                            str(provider.get("name") or provider_id),
                            str(provider.get("source") or "url"),
                            str(provider.get("command") or ""),
                            str(provider.get("type") or "openai-compatible"),
                            str(provider.get("base_url") or ""),
                            bool(provider.get("enabled", True)),
                            int(provider.get("priority") or 0),
                            json.dumps(provider.get("extra") or {}, ensure_ascii=False),
                        ),
                    )
                    models = provider.get("models") or []
                    cursor.executemany(
                        "INSERT INTO provider_models (provider_id, model_id, position) VALUES (%s, %s, %s)",
                        [
                            (provider_id, str(model), index)
                            for index, model in enumerate(models)
                        ],
                    )
                for provider_id, api_key in (credential_updates or {}).items():
                    if not api_key:
                        continue
                    cursor.execute(
                        "DELETE FROM ai_tokens WHERE provider_id = %s",
                        (provider_id,),
                    )
                    cursor.execute(
                        """
                        INSERT INTO ai_tokens
                            (provider_id, token_value, position, is_active)
                        VALUES (%s, %s, 0, TRUE)
                        """,
                        (provider_id, api_key),
                    )
                    cursor.execute(
                        """
                        INSERT INTO app_settings (setting_key, setting_value)
                        VALUES (%s, 0)
                        ON DUPLICATE KEY UPDATE setting_value = 0
                        """,
                        (self._active_ai_token_setting(provider_id),),
                    )
            connection.commit()

    def load_tool_permissions_or_migrate(self, path: str | Path) -> dict[str, str]:
        if not self.has_migrated("tool_permissions"):
            source = Path(path)
            data: Any = None
            if source.exists():
                try:
                    data = json.loads(source.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise RuntimeError("Could not migrate tool permissions file") from error
            self.save_tool_permissions(data if isinstance(data, dict) else {})
            self.mark_migrated("tool_permissions")
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT tool_name, permission_mode FROM tool_permissions")
                permissions = {str(row[0]): str(row[1]) for row in cursor.fetchall()}
            connection.commit()
        return permissions

    def save_tool_permissions(self, permissions: dict[str, str]) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM tool_permissions")
                cursor.executemany(
                    "INSERT INTO tool_permissions (tool_name, permission_mode) VALUES (%s, %s)",
                    list(permissions.items()),
                )
            connection.commit()


def create_config_store(settings: Any) -> MariaDBConfigStore | None:
    if not settings.database_host:
        return None
    if not settings.database_password:
        raise RuntimeError("CAPYY_DB_PASSWORD is required when CAPYY_DB_HOST is set")
    return MariaDBConfigStore(
        host=settings.database_host,
        port=settings.database_port,
        database=settings.database_name,
        user=settings.database_user,
        password=settings.database_password,
    )
