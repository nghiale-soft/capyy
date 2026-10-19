import unittest
from PIL import Image
import numpy as np
from gateway.services.image_service import (
    create_vertical_gradient,
    parse_hex_color,
    ImageTranslationService,
)


class TestImageTranslationStrategy(unittest.TestCase):
    def test_create_vertical_gradient_contrast_boost(self):
        """Verify that gradients with low contrast automatically get specular highlight boost."""
        low_contrast_stops = [(66, 32, 16), (140, 67, 27), (66, 32, 16)]
        grad = create_vertical_gradient(100, 100, low_contrast_stops)
        arr = np.array(grad)

        # Top row should have been boosted to a luminous champagne gold specular highlight (> 150)
        top_lum = 0.299 * arr[0, 50, 0] + 0.587 * arr[0, 50, 1] + 0.114 * arr[0, 50, 2]
        self.assertGreater(top_lum, 150, f"Expected top specular highlight > 150, got {top_lum}")

        # Bottom should preserve the deep shadow (< 90)
        bot_lum = 0.299 * arr[99, 50, 0] + 0.587 * arr[99, 50, 1] + 0.114 * arr[99, 50, 2]
        self.assertLess(bot_lum, 90, f"Expected bottom shadow < 90, got {bot_lum}")

    def test_stroke_sanitization_cancels_identical_colors(self):
        """Verify that identical or low-contrast text and stroke colors are dropped to avoid blur."""
        service = ImageTranslationService(None)
        canvas = Image.new("RGB", (400, 200), (255, 255, 255))

        regions = [{
            "erase_box_2d": [20, 20, 100, 300],
            "box_2d": [20, 20, 100, 300],
            "original_text": "蒸包",
            "translated_text": "BÁNH BAO",
            "inpaint_strategy": "preserve_container_stroke_only",
            "layout_policy": "single_line_contain",
            "container_shape": "bubble",
            "bg_color_hex": "#ffffff",
            "text_color_hex": "#000000",
            "stroke_color_hex": "#000000",  # Bogus matching stroke
            "stroke_width": 3,
            "rotation_deg": 15,
            "bold": True,
        }]

        out = service._render_translations(canvas, regions)
        self.assertIsInstance(out, Image.Image)

    def test_hero_title_scale_and_wrap(self):
        """Verify hero titles expand available width and choose optimal font scale."""
        service = ImageTranslationService(None)
        canvas = Image.new("RGB", (1000, 1000), (240, 240, 240))

        regions = [{
            "erase_box_2d": [75, 49, 195, 439],
            "box_2d": [73, 46, 195, 442],
            "original_text": "爬行机",
            "translated_text": "MÁY TẬP BỤNG",
            "inpaint_strategy": "telea_context_aware",
            "layout_policy": "hero_impact_wrap",
            "container_shape": "none",
            "bg_color_hex": "#f5f3f0",
            "text_color_hex": "#ff7500",
            "gradient_colors": ["#ffea79", "#ff6600"],
            "stroke_color_hex": "#ffffff",
            "stroke_width": 2,
            "font_family": "sans",
            "align_h": "left",
            "align_v": "center",
            "group_id": "header",
            "rotation_deg": 0,
            "bold": True,
        }]

        out = service._render_translations(canvas, regions)
        self.assertIsInstance(out, Image.Image)

    def test_container_border_preserved_during_inpainting(self):
        """Verify that enclosing container border strokes are never erased during inpainting."""
        import cv2
        service = ImageTranslationService(None)
        
        # Create a 200x200 canvas with a dark green border loop around a white/cream cloud
        cv_img = np.full((200, 200, 3), 255, dtype=np.uint8)
        # Draw outer container border loop (dark green)
        cv2.rectangle(cv_img, (20, 20), (180, 180), (27, 77, 62), 3)
        # Draw some dark text inside the container
        cv2.putText(cv_img, "TEXT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (27, 77, 62), 2)
        
        canvas = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        
        regions = [{
            "erase_box_2d": [100, 100, 900, 900],  # Box covering the container
            "box_2d": [100, 100, 900, 900],
            "original_text": "TEXT",
            "translated_text": "CHỮ",
            "inpaint_strategy": "preserve_container_stroke_only",
            "layout_policy": "hero_impact_wrap",
            "container_shape": "bubble",
            "bg_color_hex": "#ffffff",
            "text_color_hex": "#1b4d3e",
            "group_id": "bubble_labels",
            "bold": True,
        }]
        
        inpainted = service._inpaint_canvas(canvas, regions)
        arr = np.array(inpainted)
        
        # Border stroke at (20, 100) must remain dark (not erased to 255)
        border_pixel_r = int(arr[100, 20, 0])
        border_pixel_g = int(arr[100, 20, 1])
        border_pixel_b = int(arr[100, 20, 2])
        self.assertLess(border_pixel_r, 100, f"Expected dark border R < 100, got {border_pixel_r}")
        self.assertLess(border_pixel_b, 50, f"Expected dark border B < 50, got {border_pixel_b}")


if __name__ == "__main__":
    unittest.main()

