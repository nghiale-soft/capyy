import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from gateway.services.image_store import (
    IMAGE_ID_RE,
    ImageSessionExpired,
    ImageSessionNotFound,
    ImageSessionStore,
    ImageSessionTooLarge,
)


def _jpeg_bytes(color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class ImageSessionStoreTests(unittest.TestCase):
    def test_create_get_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ImageSessionStore(tmp)
            state = store.create(
                _jpeg_bytes(),
                output_bytes=_jpeg_bytes((240, 240, 240)),
                regions=[{"region_id": 0, "translated_text": "XIN CHÀO"}],
                target_language="Vietnamese",
                instruction="test",
                model="fake-model",
                provider="fake-provider",
            )

            image_id = state["id"]
            self.assertRegex(image_id, IMAGE_ID_RE)
            self.assertEqual(state["version"], 1)
            self.assertTrue((Path(tmp) / image_id / "original.jpg").exists())
            self.assertTrue((Path(tmp) / image_id / "output_v1.jpg").exists())

            loaded = store.get(image_id)
            self.assertEqual(loaded["regions"][0]["translated_text"], "XIN CHÀO")
            self.assertEqual(store.get_original(image_id), _jpeg_bytes())
            self.assertEqual(
                store.get_version_bytes(image_id), _jpeg_bytes((240, 240, 240))
            )

            self.assertTrue(store.delete(image_id))
            with self.assertRaises(ImageSessionNotFound):
                store.get(image_id)

    def test_save_version_and_idempotency_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ImageSessionStore(tmp)
            state = store.create(
                _jpeg_bytes(),
                output_bytes=_jpeg_bytes(),
                regions=[{"region_id": 0, "translated_text": "A"}],
                target_language="Vietnamese",
            )
            image_id = state["id"]
            initial_expires_at = state["expires_at"]

            state = store.save_version(
                image_id,
                _jpeg_bytes((240, 240, 240)),
                regions=[{"region_id": 0, "translated_text": "B"}],
                instruction="repair",
                idempotency_key="repair-1",
                qc_report={"approved": True, "qc_score": 100},
            )
            self.assertEqual(state["version"], 2)
            self.assertEqual(state["regions"][0]["translated_text"], "B")
            self.assertGreaterEqual(state["expires_at"], initial_expires_at)
            self.assertTrue((Path(tmp) / image_id / "output_v2.jpg").exists())

            found = store.find_version_by_idempotency_key(image_id, "repair-1")
            self.assertIsNotNone(found)
            self.assertEqual(found["version"], 2)
            self.assertEqual(found["qc_report"]["approved"], True)

    def test_expired_session_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ImageSessionStore(tmp)
            state = store.create(
                _jpeg_bytes(),
                output_bytes=_jpeg_bytes(),
                regions=[],
                target_language="Vietnamese",
            )
            image_id = state["id"]
            state_path = Path(tmp) / image_id / "state.json"
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            payload["expires_at"] = 1
            state_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ImageSessionExpired):
                store.get(image_id)
            self.assertFalse((Path(tmp) / image_id).exists())

    def test_invalid_id_cannot_escape_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ImageSessionStore(tmp)
            with self.assertRaises(ImageSessionNotFound):
                store.get("../../etc/passwd")
            with self.assertRaises(ImageSessionNotFound):
                store.get("img_not-a-valid-id")

    def test_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ImageSessionStore(tmp, max_bytes=10)
            with self.assertRaises(ImageSessionTooLarge):
                store.create(
                    b"x" * 20,
                    output_bytes=b"y" * 20,
                    regions=[],
                    target_language="Vietnamese",
                )

    def test_cleanup_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ImageSessionStore(tmp)
            state = store.create(
                _jpeg_bytes(),
                output_bytes=_jpeg_bytes(),
                regions=[],
                target_language="Vietnamese",
            )
            state_path = Path(tmp) / state["id"] / "state.json"
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            payload["expires_at"] = 1
            state_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(store.cleanup_expired(), 1)
            self.assertFalse((Path(tmp) / state["id"]).exists())


if __name__ == "__main__":
    unittest.main()
