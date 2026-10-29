from __future__ import annotations

import logging
import asyncio
from contextlib import asynccontextmanager
from contextlib import suppress
from typing import Any, AsyncIterator

from fastapi import FastAPI

from registry import ProviderRegistry
from router import Router

from .core.config import load_settings
from .core.logging import configure_logging
from .routes import auth, chat, contributions, freebuff, health, history, images, messages, models, providers, tools, videos
from .services.chat_history import ChatHistoryService
from .services.history_scan import scan_local_history
from .services.gateway_service import GatewayService
from .services.image_service import ImageTranslationService
from .services.video_service import VideoTranslationService
from .services.billing_service import BillingService
from .services.image_store import ImageSessionStore
from .services.provider_crud import ProviderCrudService
from .services.session_service import SessionService
from .services.tool_approval import ToolApprovalService
from .services.contributions import Contributions
from .services.config_store import create_config_store
from .compat.models import resolve_model


logger = logging.getLogger("gateway.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    configure_logging(settings)
    config_store = create_config_store(settings)
    if config_store is not None:
        config_store.initialize()

    # Provider CRUD + Registry + Router
    crud = ProviderCrudService(config_store=config_store)

    # Seed a FreeBuff provider so the dashboard is never empty. If the token
    # file exists the pool takes over; the provider entry itself is just a
    # config shell the user can edit/rename on the dashboard.
    if not crud.list():
        from gateway.services.provider_crud import ProviderConfig
        crud.create(
            ProviderConfig(
                id="freebuff",
                name="FreeBuff",
                source="command",
                command="freebuff",
                type="freebuff",
                models=["deepseek/deepseek-v4-flash"],
                default=True,
            )
        )
        logger.info("seeded default freebuff provider")

    registry = ProviderRegistry()
    router = Router(registry, mode=settings.router_mode)

    # Freebuff account pool (giữ logic cũ)
    accounts = SessionService(settings, config_store=config_store)

    # Editable image translation sessions (original + rendered versions).
    image_store = ImageSessionStore(
        settings.image_store_dir,
        ttl_seconds=max(0, settings.image_store_ttl_hours) * 3600,
        max_bytes=settings.image_store_max_bytes,
    )
    try:
        await asyncio.to_thread(image_store.cleanup_expired)
    except Exception:
        logger.exception("image session cleanup failed")

    # Per-project chat history
    chat_history = ChatHistoryService(settings)

    # Tool approval (allow/ask/deny per tool; pending approvals via dashboard)
    tool_approval = ToolApprovalService(settings, config_store=config_store)
    contributions_service = Contributions(settings.contributions_file, settings.contribution_issue_repository)

    # Build registry từ config đã lưu; freebuff qua factory
    def _freebuff_factory(cfg: Any):
        from gateway.services.freebuff_provider import FreebuffProviderAdapter
        return FreebuffProviderAdapter(cfg, accounts)

    registry.build_from_config(crud.list(), freebuff_factory=_freebuff_factory)

    # registry luôn có provider "freebuff" nếu có token
    if "freebuff" not in registry and settings.codebuff_token:
        from gateway.services.freebuff_provider import FreebuffProviderAdapter
        from gateway.services.provider_crud import ProviderConfig
        cfg = ProviderConfig(
            id="freebuff",
            name="Freebuff",
            type="freebuff",
            default=True,
        )
        registry.register("freebuff", FreebuffProviderAdapter(cfg, accounts), default=True)

    gateway = GatewayService(
        registry,
        router,
    )

    def _reload_registry() -> None:
        """Rebuild the registry from the persisted provider configs.

        Called after provider create/update/delete so new providers are
        routable immediately without a container restart. Mirrors the startup
        logic: CRUD providers first, then the env-token freebuff fallback.
        """
        registry.reload(crud.list(), freebuff_factory=_freebuff_factory)
        if "freebuff" not in registry and settings.codebuff_token:
            from gateway.services.freebuff_provider import FreebuffProviderAdapter
            from gateway.services.provider_crud import ProviderConfig
            cfg = ProviderConfig(
                id="freebuff",
                name="Freebuff",
                type="freebuff",
                default=True,
            )
            registry.register("freebuff", FreebuffProviderAdapter(cfg, accounts), default=True)

    app.state.settings = settings
    app.state.accounts = accounts
    app.state.chat_history = chat_history
    app.state.tool_approval = tool_approval
    app.state.contributions = contributions_service
    app.state.model_resolver = resolve_model
    app.state.provider_crud = crud
    app.state.registry = registry
    app.state.router = router
    app.state.gateway = gateway
    app.state.image_store = image_store
    app.state.image_service = ImageTranslationService(gateway, store=image_store)
    app.state.video_service = VideoTranslationService(gateway)
    app.state.billing_service = BillingService(settings)
    app.state.reload_registry = _reload_registry
    app.state.logger = logger

    async def _history_sync_loop() -> None:
        """Keep Claude/Codex JSONL imports current independently of the UI."""
        while True:
            try:
                await asyncio.to_thread(scan_local_history, chat_history)
                await asyncio.to_thread(chat_history.refresh_conversation_index)
            except Exception:
                logger.exception("background chat-history scan failed")
            await asyncio.sleep(20)

    history_sync_task = asyncio.create_task(_history_sync_loop())

    logger.info(
        "ai-gateway started providers=%s default=%s",
        list(registry.all().keys()),
        getattr(registry.get_default(), "id", None) if len(registry) else None,
    )
    try:
        yield
    finally:
        history_sync_task.cancel()
        with suppress(asyncio.CancelledError):
            await history_sync_task
        await registry.aclose()
        await accounts.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="capyy", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(messages.router)
    app.include_router(images.router)
    app.include_router(providers.router)
    app.include_router(freebuff.router)
    app.include_router(history.router)
    app.include_router(tools.router)
    app.include_router(contributions.router)
    app.include_router(videos.router)
    return app


app = create_app()
