"""Headless Chrome tools (Playwright) for the gateway agent loop.

A single browser/page instance is shared across tool calls (module-level), so
``browser_open`` → ``browser_click`` → ``browser_snapshot`` work as a session.
Playwright is imported lazily so the gateway still starts if it isn't
installed; the tools then return a clear setup message.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger("gateway.services.browser_tools")

_pw: Any = None
_browser: Any = None
_page: Any = None
_lock = asyncio.Lock()

_SETUP_HINT = (
    "Playwright is not installed. Run: uv pip install playwright && "
    "playwright install chromium (or in Docker rebuild with the new image)."
)


async def _ensure_browser() -> str | None:
    """Lazily start headless Chromium. Returns an error message or None."""
    global _pw, _browser
    if _browser is not None:
        return None
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return _SETUP_HINT
    try:
        _pw = await async_playwright().start()
        _browser = await _pw.chromium.launch(headless=True)
    except Exception as exc:  # noqa: BLE001 - report launch failure to the model
        logger.warning("chromium launch failed: %s", exc)
        return f"Failed to start Chromium: {exc}"
    logger.info("headless chromium started")
    return None


async def _page_or_error() -> tuple[Any | None, str | None]:
    error = await _ensure_browser()
    if error:
        return None, error
    global _page
    if _page is None:
        _page = await _browser.new_page()
    return _page, None


async def browser_open(url: str) -> str:
    if not str(url).startswith(("http://", "https://")):
        return "browser_open requires an http(s) url"
    async with _lock:
        page, error = await _page_or_error()
        if error:
            return error
        try:
            await page.goto(url, timeout=30000, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            return f"Failed to open {url}: {exc}"
        return f"Opened {url}\nTitle: {await page.title()}\nURL: {page.url}"


async def browser_navigate(url: str) -> str:
    return await browser_open(url)


async def browser_snapshot(max_chars: int = 50000) -> str:
    async with _lock:
        page, error = await _page_or_error()
        if error:
            return error
        try:
            text = await page.inner_text("body")
        except Exception as exc:  # noqa: BLE001
            return f"Snapshot failed: {exc}"
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return f"URL: {page.url}\n\n{text}"


async def browser_click(selector: str) -> str:
    async with _lock:
        page, error = await _page_or_error()
        if error:
            return error
        try:
            await page.click(selector, timeout=10000)
        except Exception as exc:  # noqa: BLE001
            return f"Click failed on '{selector}': {exc}"
        return f"Clicked '{selector}' — URL: {page.url}"


async def browser_type(selector: str, text: str) -> str:
    async with _lock:
        page, error = await _page_or_error()
        if error:
            return error
        try:
            await page.fill(selector, str(text))
        except Exception as exc:  # noqa: BLE001
            return f"Type failed on '{selector}': {exc}"
        return f"Typed into '{selector}'"


async def browser_eval(js: str) -> str:
    async with _lock:
        page, error = await _page_or_error()
        if error:
            return error
        try:
            result = await page.evaluate(str(js))
        except Exception as exc:  # noqa: BLE001
            return f"eval failed: {exc}"
        return str(result)[:20000]


async def browser_screenshot(path: str = "") -> str:
    async with _lock:
        page, error = await _page_or_error()
        if error:
            return error
        target = str(path).strip() or "browser-screenshot.png"
        try:
            await page.screenshot(path=target, full_page=True)
        except Exception as exc:  # noqa: BLE001
            return f"Screenshot failed: {exc}"
        return f"Screenshot saved to {target}"


async def browser_close() -> str:
    global _browser, _page
    async with _lock:
        try:
            if _browser is not None:
                await _browser.close()
        except Exception:  # noqa: BLE001
            pass
        _browser = None
        _page = None
        return "Browser closed"
