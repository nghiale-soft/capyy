import logging
import unittest

from gateway.core.logging import ColorFormatter


class LoggingConfigTests(unittest.TestCase):
    def test_color_formatter_adds_ansi_color(self) -> None:
        formatter = ColorFormatter("%(levelname)s %(message)s")
        record = logging.LogRecord(
            "gateway.test",
            logging.INFO,
            __file__,
            1,
            "hello",
            (),
            None,
        )

        message = formatter.format(record)

        self.assertTrue(message.startswith("\033[32m"))
        self.assertTrue(message.endswith("\033[0m"))


if __name__ == "__main__":
    unittest.main()
