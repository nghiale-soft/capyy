import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _run(coro):
    return asyncio.run(coro)

from gateway.routes.providers import _body_to_config
from gateway.services.provider_crud import ProviderConfig, ProviderCrudService


class ProviderConfigSourceTests(unittest.TestCase):
    def test_defaults_to_url_source(self) -> None:
        cfg = ProviderConfig(id="x", name="X")
        self.assertEqual(cfg.source, "url")
        self.assertEqual(cfg.command, "")

    def test_command_provider_fields(self) -> None:
        cfg = ProviderConfig(id="freebuff", name="FreeBuff", source="command", command="freebuff")
        self.assertEqual(cfg.source, "command")
        self.assertEqual(cfg.command, "freebuff")
        public = cfg.to_public()
        self.assertEqual(public["source"], "command")
        self.assertEqual(public["command"], "freebuff")

    def test_legacy_freebuff_type_migrates_to_command_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "providers": [
                            {"id": "freebuff", "name": "FreeBuff", "type": "freebuff"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            crud = ProviderCrudService(path=path)
            cfg = crud.get("freebuff")
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg.source, "command")
            self.assertEqual(cfg.command, "freebuff")

    def test_legacy_freebuff_multiple_models_migrates_to_one(self) -> None:
        """Legacy freebuff entries with several models keep only the first."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "providers": [
                            {
                                "id": "freebuff",
                                "name": "FreeBuff",
                                "source": "command",
                                "command": "freebuff",
                                "models": [
                                    "deepseek/deepseek-v4-flash",
                                    "mimo/mimo-v2.5",
                                    "google/gemini-2.5-flash-lite",
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            crud = ProviderCrudService(path=path)
            cfg = crud.get("freebuff")
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg.models, ["deepseek/deepseek-v4-flash"])

    def test_legacy_openai_provider_stays_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "providers": [
                            {"id": "claude", "name": "Claude", "type": "openai-compatible"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            crud = ProviderCrudService(path=path)
            cfg = crud.get("claude")
            self.assertEqual(cfg.source, "url")
            self.assertEqual(cfg.command, "")

    def test_body_to_config_command_defaults_to_freebuff(self) -> None:
        cfg = _body_to_config({"source": "command", "id": "fb"}, "fb")
        self.assertEqual(cfg.command, "freebuff")
        self.assertEqual(cfg.type, "freebuff")

    def test_body_to_config_url_type(self) -> None:
        cfg = _body_to_config(
            {"source": "url", "id": "ollama", "type": "ollama", "base_url": "http://localhost:11434/v1"},
            "ollama",
        )
        self.assertEqual(cfg.source, "url")
        self.assertEqual(cfg.type, "ollama")
        self.assertEqual(cfg.base_url, "http://localhost:11434/v1")

    def test_body_to_config_priority(self) -> None:
        cfg = _body_to_config({"source": "url", "id": "a", "priority": 3}, "a")
        self.assertEqual(cfg.priority, 3)

    def test_new_provider_appends_to_end_of_priority_order(self) -> None:
        """Creating a provider without priority must NOT jump to the front."""
        from gateway.routes.providers import create_provider

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.json"
            crud = ProviderCrudService(path=path)
            crud.create(ProviderConfig(id="a", name="A", priority=0))
            crud.create(ProviderConfig(id="b", name="B", priority=1))

            # Simulate the route assigning priority when body omits it.
            cfg = _body_to_config({"name": "C"}, "c")
            existing = crud.list()
            cfg.priority = max((p.priority for p in existing), default=-1) + 1
            crud.create(cfg)

            self.assertEqual([p.id for p in crud.ordered()], ["a", "b", "c"])

    def test_reorder_sets_priority_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.json"
            crud = ProviderCrudService(path=path)
            crud.create(ProviderConfig(id="a", name="A"))
            crud.create(ProviderConfig(id="b", name="B"))
            crud.create(ProviderConfig(id="c", name="C"))

            crud.reorder(["c", "a", "b"])
            ordered = crud.ordered()
            self.assertEqual([p.id for p in ordered], ["c", "a", "b"])
            self.assertEqual([p.priority for p in ordered], [0, 1, 2])

            # Persisted: reload sees the new order
            crud2 = ProviderCrudService(path=path)
            self.assertEqual([p.id for p in crud2.ordered()], ["c", "a", "b"])

    def test_reorder_unknown_provider_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.json"
            crud = ProviderCrudService(path=path)
            crud.create(ProviderConfig(id="a", name="A"))
            with self.assertRaises(KeyError):
                crud.reorder(["a", "ghost"])


class OpenAICompatibleProviderModelsTests(unittest.TestCase):
    def test_models_attr_does_not_shadow_method(self) -> None:
        """gateway_service calls await provider.models() — the config list must
        not shadow the async method (regression for TypeError: list not callable)."""
        import asyncio
        import httpx
        from providers.openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            "test",
            base_url="http://x/v1",
            models=["m1", "m2"],
        )
        try:
            result = asyncio.run(provider.models())
            self.assertEqual(result, ["m1", "m2"])
        finally:
            asyncio.run(provider.aclose())

    def test_models_fetches_from_api_when_empty(self) -> None:
        import asyncio
        from providers.openai_compatible import OpenAICompatibleProvider

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"data": [{"id": "a"}]}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def get(self, url, headers=None, timeout=None):
                return _FakeResponse()

            async def aclose(self):
                return None

        provider = OpenAICompatibleProvider("test", base_url="http://x/v1")
        provider._client = _FakeClient()
        try:
            result = asyncio.run(provider.models())
            self.assertEqual(result, ["a"])
        finally:
            asyncio.run(provider.aclose())


class RegistryCommandSkipTests(unittest.TestCase):
    def test_command_providers_other_than_freebuff_are_skipped(self) -> None:
        from registry import ProviderRegistry

        registry = ProviderRegistry()
        configs = [
            ProviderConfig(id="claude-code", name="Claude Code", source="command", command="claude"),
            ProviderConfig(id="freebuff", name="FreeBuff", source="command", command="freebuff"),
        ]
        with patch("registry.OpenAICompatibleProvider"):
            registry.build_from_config(configs)
        # freebuff command provider is not built without a factory; but the
        # non-freebuff command provider must NOT be turned into an HTTP provider.
        self.assertNotIn("claude-code", registry.all())
        self.assertNotIn("freebuff", registry.all())

    def test_command_provider_does_not_become_openai_provider(self) -> None:
        from registry import ProviderRegistry

        registry = ProviderRegistry()
        configs = [
            ProviderConfig(id="codex", name="Codex", source="command", command="codex"),
        ]
        with patch("registry.OpenAICompatibleProvider") as mock_provider:
            registry.build_from_config(configs)
        mock_provider.assert_not_called()
        self.assertNotIn("codex", registry.all())

    def test_reload_swaps_providers_in_place(self) -> None:
        from registry import ProviderRegistry

        registry = ProviderRegistry()
        registry.register("old", object())
        configs = [
            ProviderConfig(id="new", name="New", source="url", type="openai-compatible", base_url="http://x/v1"),
        ]
        with patch("registry.OpenAICompatibleProvider") as mock_cls:
            mock_cls.return_value = object()
            registry.reload(configs)
        self.assertNotIn("old", registry.all())
        self.assertIn("new", registry.all())
        self.assertIsNotNone(registry.get_default())

    def test_build_from_config_orders_by_priority(self) -> None:
        from registry import ProviderRegistry

        registry = ProviderRegistry()
        configs = [
            ProviderConfig(id="low", name="Low", source="url", priority=10, base_url="http://a/v1"),
            ProviderConfig(id="high", name="High", source="url", priority=1, base_url="http://b/v1"),
        ]
        with patch("registry.OpenAICompatibleProvider") as mock_cls:
            mock_cls.side_effect = lambda pid, **kw: type("P", (), {"id": pid})()
            registry.build_from_config(configs)
        self.assertEqual(registry.ordered_ids(), ["high", "low"])
        self.assertEqual(registry.get_default().id, "high")

    def test_disabled_provider_is_not_registered_or_selected(self) -> None:
        from registry import ProviderRegistry

        registry = ProviderRegistry()
        configs = [
            ProviderConfig(id="off", name="Off", enabled=False, priority=0, base_url="http://off/v1"),
            ProviderConfig(id="on", name="On", enabled=True, priority=1, base_url="http://on/v1"),
        ]
        with patch("registry.OpenAICompatibleProvider") as mock_cls:
            mock_cls.return_value = type("P", (), {"id": "on"})()
            registry.build_from_config(configs)
        self.assertEqual(registry.ordered_ids(), ["on"])
        self.assertEqual(registry.get_default().id, "on")


class FreebuffCatalogRouteTests(unittest.TestCase):
    def test_list_freebuff_models_returns_catalog(self) -> None:
        from gateway.compat.models import ALL_MODELS
        from gateway.routes.freebuff import list_freebuff_models

        result = _run(list_freebuff_models(None))
        self.assertEqual(result["models"], [m.id for m in ALL_MODELS])
        self.assertIn("deepseek/deepseek-v4-flash", result["models"])
        self.assertIn("google/gemini-2.5-flash-lite", result["models"])
        self.assertEqual(len(result["models"]), len(ALL_MODELS))


class FetchModelsRouteTests(unittest.TestCase):
    def test_fetch_models_returns_model_ids(self) -> None:
        from gateway.routes.providers import fetch_models

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"data": [{"id": "model-a"}, {"id": "model-b"}]}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                self.url = url
                return _FakeResponse()

        fake = _FakeClient()
        with patch("httpx.AsyncClient", return_value=fake):
            result = _run(fetch_models(None, {"base_url": "https://api.example.com/v1", "api_key": "k"}))
        self.assertEqual(result, {"models": ["model-a", "model-b"]})

    def test_fetch_models_handles_v1_suffix(self) -> None:
        from gateway.routes.providers import fetch_models

        captured = {}

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"data": [{"id": "m1"}]}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                captured["url"] = url
                return _FakeResponse()

        with patch("httpx.AsyncClient", return_value=_FakeClient()):
            result = _run(
                fetch_models(None, {"base_url": "https://api.example.com/v1", "api_key": "k"})
            )
        self.assertEqual(result, {"models": ["m1"]})
        self.assertEqual(captured["url"], "https://api.example.com/v1/models")

    def test_fetch_models_non_dict_json_rejected(self) -> None:
        from fastapi import HTTPException
        from gateway.routes.providers import fetch_models

        class _FakeResponse:
            status_code = 200

            def json(self):
                return ["not", "an", "object"]

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                return _FakeResponse()

        with patch("httpx.AsyncClient", return_value=_FakeClient()):
            with self.assertRaises(HTTPException) as ctx:
                _run(fetch_models(None, {"base_url": "https://api.example.com/v1"}))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_fetch_models_handles_chat_completions_suffix(self) -> None:
        from fastapi import HTTPException
        from gateway.routes.providers import fetch_models

        captured = {}

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"data": [{"id": "m1"}]}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                captured["url"] = url
                return _FakeResponse()

        with patch("httpx.AsyncClient", return_value=_FakeClient()):
            result = _run(
                fetch_models(
                    None,
                    {"base_url": "https://api.example.com/v1/chat/completions", "api_key": "k"},
                )
            )
        self.assertEqual(result, {"models": ["m1"]})
        self.assertEqual(captured["url"], "https://api.example.com/v1/models")

    def test_fetch_models_missing_base_url_rejected(self) -> None:
        from fastapi import HTTPException
        from gateway.routes.providers import fetch_models

        with self.assertRaises(HTTPException) as ctx:
            _run(fetch_models(None, {}))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_fetch_models_unreachable_returns_hint(self) -> None:
        from fastapi import HTTPException
        from gateway.routes.providers import fetch_models

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                import httpx

                raise httpx.ConnectError("boom")

        with patch("httpx.AsyncClient", return_value=_FakeClient()):
            with self.assertRaises(HTTPException) as ctx:
                _run(fetch_models(None, {"base_url": "https://down.example.com/v1"}))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("manually", str(ctx.exception.detail).lower())


if __name__ == "__main__":
    unittest.main()
