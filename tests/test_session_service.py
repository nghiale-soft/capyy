import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gateway.core.config import Settings
from gateway.services.session_service import SessionService


class FakePool:
    instances: list["FakePool"] = []

    def __init__(self, settings, *, default_index=0, on_default_change=None) -> None:
        self.settings = settings
        self.default_index = default_index
        self.on_default_change = on_default_change
        self.closed = False
        FakePool.instances.append(self)

    @property
    def account_count(self) -> int:
        # CodebuffAccountPool thật luôn tạo >= 1 account (token None nếu rỗng)
        tokens = self.settings.codebuff_tokens
        return len(tokens) if tokens else 1

    @property
    def default_client(self):
        return object()

    @property
    def default_sessions(self):
        return object()

    async def acquire_session(self, model, messages=None):
        return None

    async def aclose(self) -> None:
        self.closed = True


class SessionServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        FakePool.instances.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tokens_file = str(Path(self._tmp.name) / "tokens.json")

    def _settings(self, env_token: str | None = "env-a,env-b") -> Settings:
        return Settings(
            codebuff_token=env_token,
            local_api_key=None,
            tokens_file=self.tokens_file,
        )

    async def test_env_tokens_used_when_no_file(self) -> None:
        with patch(
            "gateway.services.session_service.CodebuffAccountPool",
            FakePool,
        ):
            service = SessionService(self._settings())

        self.assertEqual(service.account_count, 2)
        self.assertEqual(service.token_source, "env")
        self.assertEqual(
            FakePool.instances[0].settings.codebuff_token,
            "env-a,env-b",
        )

    async def test_file_tokens_override_env(self) -> None:
        Path(self.tokens_file).write_text(
            '{"version": 1, "tokens": ["file-a", "file-b", "file-c"]}',
            encoding="utf-8",
        )
        with patch(
            "gateway.services.session_service.CodebuffAccountPool",
            FakePool,
        ):
            service = SessionService(self._settings())

        self.assertEqual(service.account_count, 3)
        self.assertEqual(service.token_source, "file")
        self.assertEqual(
            FakePool.instances[0].settings.codebuff_token,
            "file-a,file-b,file-c",
        )

    async def test_persists_and_restores_active_token_index(self) -> None:
        Path(self.tokens_file).write_text(
            '{"version": 2, "tokens": ["file-a", "file-b"], "active_index": 1}',
            encoding="utf-8",
        )
        with patch("gateway.services.session_service.CodebuffAccountPool", FakePool):
            service = SessionService(self._settings())
            self.assertEqual(FakePool.instances[0].default_index, 1)
            service._persist_active_index(0)

        raw = __import__("json").loads(Path(self.tokens_file).read_text(encoding="utf-8"))
        self.assertEqual(raw["active_index"], 0)

    async def test_update_tokens_writes_file_and_reloads_pool(self) -> None:
        with patch(
            "gateway.services.session_service.CodebuffAccountPool",
            FakePool,
        ):
            service = SessionService(self._settings())
            await service.update_tokens(["new-a", "new-b", "new-c", "new-d"])

        self.assertEqual(service.account_count, 4)
        self.assertEqual(service.token_source, "file")
        self.assertTrue(Path(self.tokens_file).exists())
        self.assertTrue(FakePool.instances[0].closed, "old pool should be closed")
        self.assertEqual(
            FakePool.instances[1].settings.codebuff_token,
            "new-a,new-b,new-c,new-d",
        )

    async def test_update_tokens_normalizes_whitespace_and_empty(self) -> None:
        with patch(
            "gateway.services.session_service.CodebuffAccountPool",
            FakePool,
        ):
            service = SessionService(self._settings())
            await service.update_tokens([" a ", "", "b", "  "])

        self.assertEqual(service.account_count, 2)
        self.assertEqual(
            FakePool.instances[1].settings.codebuff_token,
            "a,b",
        )

    async def test_clear_tokens_removes_file_and_falls_back_to_env(self) -> None:
        with patch(
            "gateway.services.session_service.CodebuffAccountPool",
            FakePool,
        ):
            service = SessionService(self._settings())
            await service.update_tokens(["only-file"])
            self.assertEqual(service.token_source, "file")

            await service.clear_tokens()

        self.assertFalse(Path(self.tokens_file).exists())
        self.assertEqual(service.token_source, "env")
        self.assertEqual(service.account_count, 2)
        self.assertTrue(FakePool.instances[1].closed)
        self.assertEqual(
            FakePool.instances[2].settings.codebuff_token,
            "env-a,env-b",
        )

    async def test_no_tokens_anywhere_builds_empty_pool(self) -> None:
        with patch(
            "gateway.services.session_service.CodebuffAccountPool",
            FakePool,
        ):
            service = SessionService(self._settings(env_token=None))

        # pool thật vẫn tạo 1 account dummy (token None); không có nguồn token nào
        self.assertEqual(service.account_count, 1)
        self.assertEqual(service.token_source, "none")

    async def test_update_tokens_with_empty_list_clears_file_and_uses_env(self) -> None:
        with patch(
            "gateway.services.session_service.CodebuffAccountPool",
            FakePool,
        ):
            service = SessionService(self._settings())
            await service.update_tokens(["file-only"])
            self.assertEqual(service.token_source, "file")

            await service.update_tokens([])

        self.assertFalse(Path(self.tokens_file).exists())
        self.assertEqual(service.token_source, "env")
        self.assertEqual(service.account_count, 2)

    async def test_tokens_property_returns_pool_order(self) -> None:
        with patch(
            "gateway.services.session_service.CodebuffAccountPool",
            FakePool,
        ):
            service = SessionService(self._settings())
            await service.update_tokens(["abcdefgh1234", "xyz7890"])

        self.assertEqual(service.tokens, ["abcdefgh1234", "xyz7890"])

    async def test_remove_token_updates_file_and_pool(self) -> None:
        with patch(
            "gateway.services.session_service.CodebuffAccountPool",
            FakePool,
        ):
            service = SessionService(self._settings())
            await service.update_tokens(["a", "b", "c"])

            remaining = await service.remove_token(1)

        self.assertEqual(remaining, ["a", "c"])
        self.assertEqual(service.tokens, ["a", "c"])
        self.assertEqual(
            FakePool.instances[2].settings.codebuff_token,
            "a,c",
        )

    async def test_remove_token_out_of_range_raises(self) -> None:
        with patch(
            "gateway.services.session_service.CodebuffAccountPool",
            FakePool,
        ):
            service = SessionService(self._settings())
            await service.update_tokens(["a"])
            with self.assertRaises(IndexError):
                await service.remove_token(5)

    async def test_remove_last_token_falls_back_to_env(self) -> None:
        with patch(
            "gateway.services.session_service.CodebuffAccountPool",
            FakePool,
        ):
            service = SessionService(self._settings())
            await service.update_tokens(["only-file"])
            remaining = await service.remove_token(0)

        self.assertEqual(remaining, ["env-a", "env-b"])
        self.assertEqual(service.token_source, "env")
        self.assertFalse(Path(self.tokens_file).exists())


if __name__ == "__main__":
    unittest.main()
