from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


HAR_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class Settings:
    codebuff_token: str | None
    local_api_key: str | None
    codebuff_base_url: str = "https://www.codebuff.com"
    zeroclick_base_url: str = "https://zeroclick.dev"
    session_id: str = ""
    client_id: str = ""
    ad_providers: tuple[str, ...] = ("gravity", "zeroclick")
    request_timeout: float = 60.0
    debug: bool = False
    log_level: str = "INFO"
    log_body_chars: int = 2000
    log_color: bool = True
    host: str = "0.0.0.0"
    port: int = 1221
    # Dashboard UI (web) runs on its own port; no paths are served on the API port
    dashboard_enabled: bool = True
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 2222
    proxy_enabled: bool = False
    proxy_url: str | None = None
    timezone: str = "Asia/Shanghai"
    locale: str = "zh-CN"
    os_name: str = "windows"
    # Freebuff free models have a daily quota; upstream checks remaining quota
    # against max_tokens in the request and returns 429
    # free-models-per-day-high-balance when exceeded. 0 = unlimited.
    max_tokens: int = 8192
    # Auto-retry upstream 429/5xx
    retry_attempts: int = 4
    retry_base_delay: float = 1.0
    retry_max_delay: float = 10.0
    # Cooldown seconds after an account hits 429: no new requests are assigned
    # to that account during this window (multi-account auto-switching)
    account_cooldown: float = 60.0
    # File storing Freebuff tokens configured via the dashboard (gitignored).
    # If the file exists and has tokens, it wins over the FREEBUFF_TOKEN env.
    tokens_file: str = "config/freebuff-tokens.json"
    # Per-project chat history (JSONL). "always" injects history context into
    # every request; "memory_only" injects only when the user asks about the
    # past (do you remember / last time / previously...); "off" disables it.
    history_dir: str = "data/chat_history"
    history_max_age_days: int = 365
    history_inject_mode: str = "memory_only"
    history_context_max_chars: int = 4000
    router_mode: str = "fallback"
    providers_file: str = "config/providers.json"
    # Local tool execution (agent loop). Freebuff free models reject requests that
    # carry `tools` (429 free-models-per-day-high-balance), so the gateway strips
    # them upstream and instead executes tools locally (read_file, bash, ...).
    tool_workdir: str = "."
    tool_max_iterations: int = 8
    tool_bash_enabled: bool = True
    tool_command_timeout: float = 30.0
    tool_output_cap: int = 50000
    tool_file_cap: int = 100000
    # Tool approval (Hướng B): per-tool mode allow/ask/deny persisted in
    # tool_permissions_file (editable from the Dashboard → Settings). In "ask"
    # mode the agent loop pauses and waits for the user to approve/deny via the
    # dashboard; timeout seconds before the tool call is auto-denied.
    tool_approval_timeout: float = 120.0
    tool_permissions_file: str = "config/tool-permissions.json"
    tool_mapping_contributions_file: str = "data/tool-mapping-contributions.json"
    tool_mapping_issue_repository: str = "nghiale-soft/capyy"

    @property
    def codebuff_api_url(self) -> str:
        return self.codebuff_base_url.strip().rstrip("/")

    @property
    def zeroclick_api_url(self) -> str:
        return self.zeroclick_base_url.rstrip("/")

    @property
    def upstream_proxy_url(self) -> str | None:
        if not self.proxy_enabled:
            return None
        if not self.proxy_url:
            return None
        return self.proxy_url.strip() or None

    @property
    def codebuff_tokens(self) -> tuple[str, ...]:
        if not self.codebuff_token:
            return ()
        values = [item.strip() for item in self.codebuff_token.split(",")]
        return tuple(item for item in values if item)


def _csv(name: str, default: str) -> tuple[str, ...]:
    values = [item.strip() for item in os.getenv(name, default).split(",")]
    return tuple(item for item in values if item)


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _api_base_url() -> str:
    return (
        os.getenv("FREEBUFF_API_BASE_URL")
        or os.getenv("CODEBUFF_BASE_URL")
        or "https://www.codebuff.com"
    )


def load_settings() -> Settings:
    debug = _bool("FREEBUFF_DEBUG", False)
    log_level = "DEBUG" if debug else os.getenv("FREEBUFF_LOG_LEVEL", "INFO")
    color_default = os.getenv("NO_COLOR") is None
    return Settings(
        codebuff_token=os.getenv("FREEBUFF_TOKEN") or os.getenv("CODEBUFF_TOKEN"),
        local_api_key=os.getenv("FREEBUFF_API_KEY") or os.getenv("OPENAI_API_KEY"),
        codebuff_base_url=_api_base_url(),
        zeroclick_base_url=os.getenv("ZEROCLICK_BASE_URL", "https://zeroclick.dev"),
        session_id=os.getenv("FREEBUFF_SESSION_ID", str(uuid.uuid4())),
        client_id=os.getenv("FREEBUFF_CLIENT_ID", uuid.uuid4().hex[:11]),
        ad_providers=_csv("FREEBUFF_AD_PROVIDERS", "gravity,carbon"),
        request_timeout=float(os.getenv("FREEBUFF_TIMEOUT", "60")),
        debug=debug,
        log_level=log_level,
        log_body_chars=_int("FREEBUFF_LOG_BODY_CHARS", 0 if debug else 2000),
        log_color=_bool("FREEBUFF_LOG_COLOR", color_default),
        host=os.getenv("FREEBUFF_HOST", "0.0.0.0"),
        port=_int("FREEBUFF_PORT", 1221),
        dashboard_enabled=_bool("FREEBUFF_DASHBOARD_ENABLED", True),
        dashboard_host=os.getenv("FREEBUFF_DASHBOARD_HOST", "0.0.0.0"),
        dashboard_port=_int("FREEBUFF_DASHBOARD_PORT", 2222),
        proxy_enabled=_bool("FREEBUFF_PROXY_ENABLED", False),
        proxy_url=os.getenv("FREEBUFF_PROXY_URL"),
        timezone=os.getenv("FREEBUFF_TIMEZONE", "Asia/Shanghai"),
        locale=os.getenv("FREEBUFF_LOCALE", "zh-CN"),
        os_name=os.getenv("FREEBUFF_OS", "windows"),
        max_tokens=_int("FREEBUFF_MAX_TOKENS", 8192),
        retry_attempts=_int("FREEBUFF_RETRY_ATTEMPTS", 4),
        retry_base_delay=float(os.getenv("FREEBUFF_RETRY_BASE_DELAY", "1.0")),
        retry_max_delay=float(os.getenv("FREEBUFF_RETRY_MAX_DELAY", "10.0")),
        account_cooldown=float(os.getenv("FREEBUFF_ACCOUNT_COOLDOWN", "60.0")),
        tokens_file=os.getenv("FREEBUFF_TOKENS_FILE", "config/freebuff-tokens.json"),
        history_dir=os.getenv("FREEBUFF_HISTORY_DIR", "data/chat_history"),
        history_max_age_days=_int("FREEBUFF_HISTORY_MAX_AGE_DAYS", 365),
        history_inject_mode=os.getenv("FREEBUFF_HISTORY_INJECT_MODE", "memory_only"),
        history_context_max_chars=_int("FREEBUFF_HISTORY_CONTEXT_MAX_CHARS", 4000),
        router_mode=os.getenv("AI_GATEWAY_ROUTER_MODE", "fallback"),
        providers_file=os.getenv(
            "AI_GATEWAY_PROVIDERS_FILE", "config/providers.json"
        ),
        tool_workdir=os.getenv("FREEBUFF_TOOL_WORKDIR", "."),
        tool_max_iterations=_int("FREEBUFF_TOOL_MAX_ITERATIONS", 8),
        tool_bash_enabled=_bool("FREEBUFF_TOOL_BASH_ENABLED", True),
        tool_command_timeout=float(
            os.getenv("FREEBUFF_TOOL_COMMAND_TIMEOUT", "30.0")
        ),
        tool_output_cap=_int("FREEBUFF_TOOL_OUTPUT_CAP", 50000),
        tool_file_cap=_int("FREEBUFF_TOOL_FILE_CAP", 100000),
        tool_approval_timeout=float(
            os.getenv("FREEBUFF_TOOL_APPROVAL_TIMEOUT", "120.0")
        ),
        tool_permissions_file=os.getenv(
            "FREEBUFF_TOOL_PERMISSIONS_FILE", "config/tool-permissions.json"
        ),
        tool_mapping_contributions_file=os.getenv("CAPYY_TOOL_MAPPING_CONTRIBUTIONS_FILE", "data/tool-mapping-contributions.json"),
        tool_mapping_issue_repository=os.getenv("CAPYY_TOOL_MAPPING_ISSUE_REPOSITORY", "nghiale-soft/capyy"),
    )
