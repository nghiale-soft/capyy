import tempfile
import unittest
from pathlib import Path

from gateway.services.contributions import Contributions


class ContributionsTests(unittest.TestCase):
    def test_deduplicates_pending_reports_and_counts_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reports = Contributions(str(Path(directory) / "reports.json"), "org/capyy")
            first = reports.add(
                "tool-protocol", "Invalid declared tool request", "summary",
                {"error_code": "tool_not_declared", "emitted_tool": "Glob"},
            )
            second = reports.add(
                "tool-protocol", "Invalid declared tool request", "summary",
                {"error_code": "tool_not_declared", "emitted_tool": "Glob"},
            )

            self.assertEqual(first["id"], second["id"])
            self.assertEqual(second["occurrences"], 2)
            self.assertEqual(len(reports.list()), 1)

    def test_limits_metadata_values_before_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reports = Contributions(str(Path(directory) / "reports.json"), "org/capyy")
            item = reports.add("tool-protocol", "title", "summary", {"declared_tools": "x" * 1000})
            self.assertEqual(len(item["metadata"]["declared_tools"]), 280)


if __name__ == "__main__":
    unittest.main()
