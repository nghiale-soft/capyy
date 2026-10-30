import base64
import io
import json
import tempfile
import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from gateway.routes.images import router
from gateway.services.image_service import ImageTranslationService
from gateway.services.image_store import ImageSessionStore


def _source_image() -> bytes:
    image = Image.new("RGB", (300, 120), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((30, 40), "蛋蒸1-18", fill=(0, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _region(translated_text: str, region_id: int | None = None) -> dict:
    region = {
        "erase_box_2d": [200, 50, 800, 950],
        "box_2d": [200, 50, 800, 950],
        "original_text": "蛋蒸1-18",
        "translated_text": translated_text,
        "inpaint_strategy": "flat_color_patch",
        "layout_policy": "single_line_contain",
        "container_shape": "none",
        "bg_color_hex": "#ffffff",
        "text_color_hex": "#000000",
        "bold": True,
    }
    if region_id is not None:
        region["region_id"] = region_id
    return region


class _FakeGateway:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def resolve(self, model):
        return "fake-provider", model or "fake-model"

    async def chat(self, provider_id, payload, *, real_model=None):
        self.calls.append(payload)
        return {"choices": [{"message": {"content": self.responses.pop(0)}}]}


class ImageRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.gateway = _FakeGateway(
            [
                json.dumps({"regions": [_region("TRỨNG")]}),
                json.dumps({"regions": [_region("TRỨNG GÀ", region_id=0)]}),
            ]
        )
        self.store = ImageSessionStore(self._tmp.name)
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.state.settings = SimpleNamespace(local_api_key=None)
        self.app.state.image_store = self.store
        self.app.state.image_service = ImageTranslationService(
            self.gateway, store=self.store
        )
        self.client = TestClient(self.app)

    def test_translate_repair_metadata_content_and_delete(self) -> None:
        source = base64.b64encode(_source_image()).decode("utf-8")
        response = self.client.post(
            "/v1/images/translate",
            json={
                "image_base64": source,
                "target_language": "Vietnamese",
                "response_format": "json",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        image_id = data["id"]
        self.assertTrue(image_id.startswith("img_"))
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["regions"][0]["region_id"], 0)

        repair = self.client.post(
            f"/v1/images/{image_id}/repair",
            json={
                "instruction": "đổi thành TRỨNG GÀ",
                "region_ids": [0],
                "idempotency_key": "route-repair-1",
            },
        )
        self.assertEqual(repair.status_code, 200, repair.text)
        repaired = repair.json()["data"]
        self.assertEqual(repaired["version"], 2)
        self.assertEqual(repaired["regions"][0]["translated_text"], "TRỨNG GÀ")
        self.assertIn("repair_check", repaired["qc_report"])

        metadata = self.client.get(f"/v1/images/{image_id}")
        self.assertEqual(metadata.status_code, 200, metadata.text)
        meta = metadata.json()["data"]
        self.assertEqual(meta["version"], 2)
        self.assertEqual(len(meta["versions"]), 2)

        content = self.client.get(f"/v1/images/{image_id}/content")
        self.assertEqual(content.status_code, 200)
        self.assertEqual(content.headers["content-type"], "image/jpeg")
        self.assertEqual(content.headers["x-capyy-image-version"], "2")

        content_v1 = self.client.get(f"/v1/images/{image_id}/content?version=1")
        self.assertEqual(content_v1.status_code, 200)
        self.assertEqual(content_v1.headers["x-capyy-image-version"], "1")

        deleted = self.client.delete(f"/v1/images/{image_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get(f"/v1/images/{image_id}").status_code, 404)

    def test_translate_rejects_image_over_store_limit(self) -> None:
        self.store.max_bytes = 10
        source = base64.b64encode(_source_image()).decode("utf-8")
        response = self.client.post(
            "/v1/images/translate",
            json={
                "image_base64": source,
                "target_language": "Vietnamese",
                "response_format": "json",
            },
        )
        self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual(self.gateway.calls, [])

    def test_invalid_bbox_and_unknown_id(self) -> None:
        response = self.client.post(
            "/v1/images/img_00000000000000000000000000000000/repair",
            json={"instruction": "sửa", "bbox_2d": [1, 2, 3]},
        )
        self.assertEqual(response.status_code, 400)

        response = self.client.get("/v1/images/img_00000000000000000000000000000000")
        self.assertEqual(response.status_code, 404)

    def test_image_test_endpoint_pass(self) -> None:
        # 1: OCR translation, 2: QC inspection pass
        self.gateway.responses = [
            json.dumps({"regions": [_region("TRỨNG")]}),
            json.dumps({
                "approved": True,
                "qc_score": 95,
                "feedback": "Translation quality is great",
                "issues": [],
            }),
        ]
        source = base64.b64encode(_source_image()).decode("utf-8")
        response = self.client.post(
            "/v1/images/test",
            json={
                "image_base64": source,
                "target_language": "Vietnamese",
                "response_format": "json",
                "pass_threshold": 80,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["status"], "passed")
        self.assertTrue(data["approved"])
        self.assertGreaterEqual(data["data"]["qc_score"], 80)
        self.assertIn("image_base64", data["data"])

    def test_image_test_endpoint_fail_and_auto_repair(self) -> None:
        # 1: OCR translation, 2: QC inspection fail, 3: Auto-repair LLM call, 4: Auto-repair QC
        self.gateway.responses = [
            json.dumps({"regions": [_region("TRỨNG")]}),
            json.dumps({
                "approved": False,
                "qc_score": 50,
                "feedback": "Chữ tiếng Trung ở góc chưa được xóa sạch",
                "issues": [
                    {
                        "issue_type": "residual_text",
                        "severity": "high",
                        "description": "Còn sót chữ tiếng Trung ở góc trên",
                    }
                ],
            }),
            json.dumps({"regions": [_region("TRỨNG ĐÃ FIX", region_id=0)]}),
            json.dumps({
                "approved": True,
                "qc_score": 95,
                "feedback": "Fixed cleanly",
                "issues": [],
            }),
        ]
        source = base64.b64encode(_source_image()).decode("utf-8")
        response = self.client.post(
            "/v1/images/test",
            json={
                "image_base64": source,
                "target_language": "Vietnamese",
                "response_format": "image",
                "pass_threshold": 80,
                "auto_repair": True,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["x-capyy-status"], "fail")
        self.assertEqual(response.headers["x-capyy-qc-approved"], "false")
        data = response.json()
        self.assertEqual(data["status"], "failed")
        self.assertFalse(data["approved"])
        self.assertEqual(data["qc_score"], 50)
        self.assertEqual(len(data["issues"]), 1)
        self.assertTrue(data["auto_repair"]["triggered"])
        self.assertIn("Còn sót chữ tiếng Trung", data["auto_repair"]["instruction"])

    def test_image_feedback_fail_triggers_repair(self) -> None:
        # First create an image session
        self.gateway.responses = [
            json.dumps({"regions": [_region("TRỨNG")]}),
            # Feedback repair responses:
            json.dumps({"regions": [_region("TRỨNG GÀ SIÊU SẠCH", region_id=0)]}),
            json.dumps({
                "approved": True,
                "qc_score": 98,
                "feedback": "Repaired successfully",
                "issues": [],
            }),
        ]
        source = base64.b64encode(_source_image()).decode("utf-8")
        trans_res = self.client.post(
            "/v1/images/translate",
            json={"image_base64": source, "response_format": "json"},
        )
        self.assertEqual(trans_res.status_code, 200)
        image_id = trans_res.json()["data"]["id"]

        # Call feedback endpoint reporting fail with reason
        feedback_res = self.client.post(
            f"/v1/images/{image_id}/feedback",
            json={
                "status": "failed",
                "reason": "chữ dịch chưa chuẩn, sửa thành TRỨNG GÀ SIÊU SẠCH",
                "response_format": "json",
            },
        )
        self.assertEqual(feedback_res.status_code, 200, feedback_res.text)
        fb_data = feedback_res.json()
        self.assertEqual(fb_data["status"], "repaired")
        self.assertEqual(fb_data["data"]["version"], 2)
        self.assertEqual(
            fb_data["data"]["regions"][0]["translated_text"], "TRỨNG GÀ SIÊU SẠCH"
        )

        # Test feedback pass recording
        pass_res = self.client.post(
            "/v1/images/feedback",
            json={
                "image_id": image_id,
                "status": "passed",
            },
        )
        self.assertEqual(pass_res.status_code, 200)
        self.assertEqual(pass_res.json()["status"], "recorded")
        self.assertTrue(pass_res.json()["approved"])


if __name__ == "__main__":
    unittest.main()
