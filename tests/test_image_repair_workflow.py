import asyncio
import io
import json
import tempfile
import unittest

from PIL import Image, ImageDraw

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
        content = self.responses.pop(0)
        return {"choices": [{"message": {"content": content}}]}


class ImageRepairWorkflowTests(unittest.TestCase):
    def test_legacy_translate_without_store_keeps_three_tuple(self) -> None:
        gateway = _FakeGateway([json.dumps({"regions": [_region("TRỨNG")]})])
        service = ImageTranslationService(gateway)
        result = asyncio.run(service.translate_image(_source_image()))
        self.assertEqual(len(result), 3)
        out_bytes, regions, qc_report = result
        self.assertTrue(out_bytes)
        self.assertEqual(regions[0]["region_id"], 0)
        self.assertTrue(qc_report["deterministic"])

    def test_translate_repair_and_idempotent_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gateway = _FakeGateway(
                [
                    json.dumps({"regions": [_region("TRỨNG")]}),
                    json.dumps({"regions": [_region("TRỨNG GÀ", region_id=0)]}),
                ]
            )
            store = ImageSessionStore(tmp)
            service = ImageTranslationService(gateway, store=store)

            out_bytes, regions, qc_report, state = asyncio.run(
                service.translate_image_session(
                    _source_image(),
                    target_language="Vietnamese",
                    instruction="dịch sang tiếng Việt",
                )
            )
            image_id = state["id"]
            self.assertEqual(state["version"], 1)
            self.assertEqual(regions[0]["region_id"], 0)
            self.assertTrue(qc_report["deterministic"])
            self.assertTrue(out_bytes)

            repaired_bytes, repaired_regions, repair_report, repaired_state = (
                asyncio.run(
                    service.repair_image(
                        image_id,
                        instruction="đổi thành TRỨNG GÀ",
                        region_ids=[0],
                        idempotency_key="repair-1",
                    )
                )
            )
            self.assertEqual(repaired_state["version"], 2)
            self.assertEqual(repaired_regions[0]["translated_text"], "TRỨNG GÀ")
            self.assertTrue(repaired_bytes)
            self.assertIn("repair_check", repair_report)

            stored = store.get(image_id)
            self.assertEqual(stored["version"], 2)
            self.assertEqual(stored["regions"][0]["translated_text"], "TRỨNG GÀ")
            self.assertEqual(len(stored["versions"]), 2)
            self.assertEqual(store.get_original(image_id), _source_image())

            # Replaying the same idempotency key must not call the model again.
            calls_before = len(gateway.calls)
            replay_bytes, replay_regions, _, replay_state = asyncio.run(
                service.repair_image(
                    image_id,
                    instruction="đổi thành TRỨNG GÀ",
                    region_ids=[0],
                    idempotency_key="repair-1",
                )
            )
            self.assertEqual(len(gateway.calls), calls_before)
            self.assertEqual(replay_state["version"], 2)
            self.assertEqual(replay_regions[0]["translated_text"], "TRỨNG GÀ")
            self.assertTrue(replay_bytes)

    def test_bbox_selects_overlapping_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gateway = _FakeGateway(
                [
                    json.dumps({"regions": [_region("TRỨNG")]}),
                    json.dumps({"regions": [_region("TRỨNG GÀ", region_id=0)]}),
                ]
            )
            store = ImageSessionStore(tmp)
            service = ImageTranslationService(gateway, store=store)
            _, _, _, state = asyncio.run(
                service.translate_image_session(_source_image())
            )

            _, regions, qc_report, repaired_state = asyncio.run(
                service.repair_image(
                    state["id"],
                    instruction="đổi thành TRỨNG GÀ",
                    bbox_2d=[180, 40, 820, 960],
                )
            )
            self.assertEqual(repaired_state["version"], 2)
            self.assertEqual(regions[0]["translated_text"], "TRỨNG GÀ")
            self.assertTrue(qc_report["repair_check"]["changed"])

    def test_repair_response_without_ids_maps_to_explicit_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gateway = _FakeGateway(
                [
                    json.dumps({"regions": [_region("TRỨNG")]}),
                    json.dumps({"regions": [_region("TRỨNG GÀ")]}),
                ]
            )
            store = ImageSessionStore(tmp)
            service = ImageTranslationService(gateway, store=store)
            _, _, _, state = asyncio.run(
                service.translate_image_session(_source_image())
            )

            _, regions, _, repaired_state = asyncio.run(
                service.repair_image(
                    state["id"],
                    instruction="đổi thành TRỨNG GÀ",
                    region_ids=[0],
                )
            )
            self.assertEqual(repaired_state["version"], 2)
            self.assertEqual(regions[0]["translated_text"], "TRỨNG GÀ")
            self.assertEqual(regions[0]["region_id"], 0)

    def test_description_only_repair_lets_model_pick_region(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gateway = _FakeGateway(
                [
                    json.dumps({"regions": [_region("TRỨNG")]}),
                    json.dumps({"regions": [_region("TRỨNG GÀ", region_id=0)]}),
                ]
            )
            store = ImageSessionStore(tmp)
            service = ImageTranslationService(gateway, store=store)
            _, _, _, state = asyncio.run(
                service.translate_image_session(_source_image())
            )

            _, regions, qc_report, repaired_state = asyncio.run(
                service.repair_image(
                    state["id"],
                    instruction="chữ ở góc dưới trái bị sai, sửa thành TRỨNG GÀ",
                )
            )
            self.assertEqual(repaired_state["version"], 2)
            self.assertEqual(regions[0]["translated_text"], "TRỨNG GÀ")
            self.assertTrue(qc_report["repair_check"]["changed"])

    def test_unknown_region_and_unsupported_base_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gateway = _FakeGateway([json.dumps({"regions": [_region("TRỨNG")]})])
            store = ImageSessionStore(tmp)
            service = ImageTranslationService(gateway, store=store)
            _, _, _, state = asyncio.run(
                service.translate_image_session(_source_image())
            )

            with self.assertRaises(ValueError):
                asyncio.run(
                    service.repair_image(
                        state["id"], instruction="sửa", region_ids=[99]
                    )
                )

            with self.assertRaises(ValueError):
                asyncio.run(
                    service.repair_image(
                        state["id"],
                        instruction="sửa",
                        region_ids=[0],
                        base="current",
                    )
                )


if __name__ == "__main__":
    unittest.main()
