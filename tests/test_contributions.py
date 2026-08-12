import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.core.config import Settings
from gateway.routes.contributions import router
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

    def test_approve_is_idempotent_and_keeps_the_same_issue_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reports = Contributions(str(Path(directory) / "reports.json"), "org/capyy")
            item = reports.add("provider", "No fallback", "summary", {"status_code": "429"})
            first = reports.issue_url(item["id"])
            second = reports.issue_url(item["id"])

            self.assertEqual(first, second)
            self.assertIn("github.com/org/capyy/issues/new", first)
            self.assertEqual(reports.list()[0]["status"], "approved")

    def test_approve_endpoint_is_not_404_after_a_client_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = FastAPI()
            app.include_router(router)
            app.state.settings = Settings(codebuff_token="", local_api_key=None)
            reports = Contributions(str(Path(directory) / "reports.json"), "org/capyy")
            item = reports.add("provider", "No fallback", "summary", {"status_code": "429"})
            app.state.contributions = reports

            with TestClient(app) as client:
                first = client.post(f"/api/contributions/{item['id']}/approve")
                retry = client.post(f"/api/contributions/{item['id']}/approve")

            self.assertEqual(first.status_code, 200)
            self.assertEqual(retry.status_code, 200)
            self.assertEqual(first.json()["issue_url"], retry.json()["issue_url"])


if __name__ == "__main__":
    unittest.main()
