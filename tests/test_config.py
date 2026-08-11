import unittest
from unittest.mock import patch

from gateway.core.config import Settings, load_settings


class ConfigTests(unittest.TestCase):
    def test_load_settings_ignores_legacy_token_environment_variables(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_TOKEN": "must-not-be-read",
                "CODEBUFF_TOKEN": "must-not-be-read-either",
            },
            clear=True,
        ):
            settings = load_settings()

        self.assertIsNone(settings.codebuff_token)

    def test_proxy_url_is_ignored_when_proxy_is_disabled(self) -> None:
        settings = Settings(
            codebuff_token="token",
            local_api_key=None,
            proxy_enabled=False,
            proxy_url="http://127.0.0.1:7890",
        )

        self.assertIsNone(settings.upstream_proxy_url)

    def test_proxy_url_is_used_when_proxy_is_enabled(self) -> None:
        settings = Settings(
            codebuff_token="token",
            local_api_key=None,
            proxy_enabled=True,
            proxy_url=" http://127.0.0.1:7890 ",
        )

        self.assertEqual(settings.upstream_proxy_url, "http://127.0.0.1:7890")

    def test_load_settings_reads_proxy_toggle_and_url(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_PROXY_ENABLED": "true",
                "FREEBUFF_PROXY_URL": "socks5://127.0.0.1:1080",
            },
        ):
            settings = load_settings()

        self.assertTrue(settings.proxy_enabled)
        self.assertEqual(settings.upstream_proxy_url, "socks5://127.0.0.1:1080")

    def test_freebuff_api_base_url_overrides_legacy_codebuff_base_url(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_API_BASE_URL": " https://freebuff-api.example.test/ ",
                "CODEBUFF_BASE_URL": "https://legacy.example.test",
            },
        ):
            settings = load_settings()

        self.assertEqual(settings.codebuff_api_url, "https://freebuff-api.example.test")

    def test_legacy_codebuff_base_url_is_still_supported(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "CODEBUFF_BASE_URL": "https://legacy.example.test/",
            },
            clear=True,
        ):
            settings = load_settings()

        self.assertEqual(settings.codebuff_api_url, "https://legacy.example.test")

    def test_codebuff_tokens_splits_comma_separated_tokens(self) -> None:
        settings = Settings(
            codebuff_token="token-a, token-b,,token-c ",
            local_api_key=None,
        )

        self.assertEqual(
            settings.codebuff_tokens,
            ("token-a", "token-b", "token-c"),
        )

    def test_load_settings_reads_max_tokens_retry_and_cooldown(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_MAX_TOKENS": "4096",
                "FREEBUFF_RETRY_ATTEMPTS": "2",
                "FREEBUFF_RETRY_BASE_DELAY": "0.5",
                "FREEBUFF_RETRY_MAX_DELAY": "5",
                "FREEBUFF_ACCOUNT_COOLDOWN": "30",
            },
        ):
            settings = load_settings()

        self.assertEqual(settings.max_tokens, 4096)
        self.assertEqual(settings.retry_attempts, 2)
        self.assertEqual(settings.retry_base_delay, 0.5)
        self.assertEqual(settings.retry_max_delay, 5.0)
        self.assertEqual(settings.account_cooldown, 30.0)

    def test_settings_defaults_max_tokens_retry_and_cooldown(self) -> None:
        settings = Settings(
            codebuff_token="token",
            local_api_key=None,
        )

        self.assertEqual(settings.max_tokens, 8192)
        self.assertEqual(settings.retry_attempts, 4)
        self.assertEqual(settings.account_cooldown, 60.0)

    def test_load_settings_reads_dashboard_port(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_DASHBOARD_PORT": "3333",
                "FREEBUFF_DASHBOARD_ENABLED": "false",
            },
        ):
            settings = load_settings()

        self.assertEqual(settings.dashboard_port, 3333)
        self.assertFalse(settings.dashboard_enabled)

    def test_settings_defaults_dashboard_port(self) -> None:
        settings = Settings(
            codebuff_token="token",
            local_api_key=None,
        )

        self.assertTrue(settings.dashboard_enabled)
        self.assertEqual(settings.dashboard_port, 2222)

    def test_load_settings_does_not_expose_custom_browser_user_agent(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "FREEBUFF_BROWSER_UA": "custom-client/1.0",
            },
            clear=True,
        ):
            settings = load_settings()

        self.assertFalse(hasattr(settings, "browser_user_agent"))


if __name__ == "__main__":
    unittest.main()
