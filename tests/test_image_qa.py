import io
import unittest

from PIL import Image, ImageDraw

from gateway.services.image_qa import inspect_repair_effect, inspect_translation


def _render(text: str | None = None, fill: tuple[int, int, int] = (0, 0, 0)) -> bytes:
    image = Image.new("RGB", (300, 120), (255, 255, 255))
    if text:
        draw = ImageDraw.Draw(image)
        draw.text((30, 40), text, fill=fill)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


REGIONS = [
    {
        "region_id": 0,
        "box_2d": [200, 50, 800, 950],
        "erase_box_2d": [200, 50, 800, 950],
        "translated_text": "HẤP 1-18 TRỨNG",
    }
]


class DeterministicQaTests(unittest.TestCase):
    def test_unchanged_source_text_is_flagged(self) -> None:
        original = _render("蛋蒸1-18")
        report = inspect_translation(original, original, REGIONS)
        self.assertFalse(report["approved"])
        self.assertEqual(report["issues"][0]["type"], "leftover_text")
        self.assertEqual(report["issues"][0]["region_id"], 0)

    def test_cleanly_erased_text_passes(self) -> None:
        original = _render("蛋蒸1-18")
        erased = _render(None)
        report = inspect_translation(original, erased, REGIONS)
        self.assertTrue(report["approved"])
        self.assertEqual(report["issues"], [])

    def test_new_translation_is_not_flagged(self) -> None:
        original = _render("蛋蒸1-18")
        translated = _render("HẤP 1-18 TRỨNG", fill=(190, 30, 20))
        report = inspect_translation(original, translated, REGIONS)
        self.assertTrue(report["approved"])


    def test_preserved_container_border_is_not_leftover_text(self) -> None:
        def bubble(translated: bool) -> bytes:
            image = Image.new("RGB", (360, 180), (160, 120, 80))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle(
                (25, 25, 335, 145),
                radius=18,
                fill=(255, 255, 255),
                outline=(0, 0, 0),
                width=4,
            )
            if translated:
                draw.rectangle((70, 50, 220, 120), fill=(255, 255, 255))
                draw.rounded_rectangle(
                    (25, 25, 335, 145),
                    radius=18,
                    outline=(0, 0, 0),
                    width=4,
                )
                draw.text((120, 70), "NEW", fill=(190, 30, 20))
            else:
                draw.text((120, 70), "OLD", fill=(0, 0, 0))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=95)
            return buf.getvalue()

        regions = [
            {
                "region_id": 0,
                "box_2d": [139, 69, 806, 931],
                "erase_box_2d": [139, 69, 806, 931],
                "translated_text": "NEW",
            }
        ]
        report = inspect_translation(bubble(False), bubble(True), regions)
        self.assertTrue(report["approved"])
        self.assertEqual(report["issues"], [])

        unchanged = inspect_translation(bubble(False), bubble(False), regions)
        self.assertFalse(unchanged["approved"])
        self.assertEqual(unchanged["issues"][0]["type"], "leftover_text")

    def test_repair_effect_detects_change_and_no_change(self) -> None:
        before = _render("CŨ")
        after = _render("MỚI")
        changed = inspect_repair_effect(before, after, REGIONS, [0])
        self.assertTrue(changed["changed"])

        unchanged = inspect_repair_effect(before, before, REGIONS, [0])
        self.assertFalse(unchanged["changed"])
        self.assertEqual(unchanged["issues"][0]["type"], "no_visible_change")


if __name__ == "__main__":
    unittest.main()
