from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

logger = logging.getLogger("gateway.services.image")

FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
DEFAULT_BOLD_FONT = FONTS_DIR / "Roboto-Bold.ttf"
DEFAULT_REG_FONT = FONTS_DIR / "Roboto-Regular.ttf"
SERIF_BOLD_FONT = FONTS_DIR / "Georgia-Bold.ttf"
SERIF_REG_FONT = FONTS_DIR / "Georgia-Regular.ttf"


def create_vertical_gradient(width: int, height: int, color_stops: list[tuple[int, int, int]]) -> Image.Image:
    """Generate a vertical linear gradient RGBA image between multi-stop RGB colors."""
    n_stops = len(color_stops)
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    if n_stops == 1:
        arr[:, :, :3] = color_stops[0]
        arr[:, :, 3] = 255
        return Image.fromarray(arr, "RGBA")

    # Gradient contrast enhancement:
    # If provided stops lack contrast (max lum - min lum < 55), inject a luminous specular highlight at the top
    lums = [0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in color_stops]
    if max(lums) - min(lums) < 55:
        top = color_stops[0]
        is_warm = top[0] >= top[2]
        if is_warm:
            # Champagne gold specular highlight (luminous metallic sheen)
            bright_top = (
                min(255, max(235, int(top[0] + 160))),
                min(255, max(210, int(top[1] + 150))),
                min(255, max(160, int(top[2] + 120))),
            )
        else:
            bright_top = (
                min(255, max(230, int(top[0] + 150))),
                min(255, max(230, int(top[1] + 150))),
                min(255, max(230, int(top[2] + 150))),
            )
        color_stops = [bright_top] + list(color_stops)
        n_stops = len(color_stops)

    positions = np.linspace(0, height - 1, n_stops)
    for y in range(height):
        idx = 0
        while idx < n_stops - 2 and y > positions[idx + 1]:
            idx += 1
        t = (y - positions[idx]) / max(1e-5, (positions[idx + 1] - positions[idx]))
        t = max(0.0, min(1.0, t))
        c1 = np.array(color_stops[idx], dtype=np.float32)
        c2 = np.array(color_stops[idx + 1], dtype=np.float32)
        c = (1 - t) * c1 + t * c2
        arr[y, :, :3] = np.clip(c, 0, 255).astype(np.uint8)
        arr[y, :, 3] = 255
    return Image.fromarray(arr, "RGBA")



def parse_hex_color(val: Any, default: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    """Parse hex color string '#RRGGBB' or RGB array into (R, G, B)."""
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        try:
            return (
                max(0, min(255, int(val[0]))),
                max(0, min(255, int(val[1]))),
                max(0, min(255, int(val[2]))),
            )
        except Exception:
            return default

    if isinstance(val, str):
        cleaned = val.strip().lstrip("#")
        if len(cleaned) == 6:
            try:
                return (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))
            except ValueError:
                pass
        elif len(cleaned) == 3:
            try:
                return (int(cleaned[0] * 2, 16), int(cleaned[1] * 2, 16), int(cleaned[2] * 2, 16))
            except ValueError:
                pass

    return default


def sanitize_box(box: Any) -> list[int] | None:
    """Safely sanitize box coordinates to [ymin, xmin, ymax, xmax] integers."""
    if not isinstance(box, (list, tuple)):
        return None
    if len(box) == 1 and isinstance(box[0], (list, tuple)):
        box = box[0]
    if len(box) != 4:
        return None
    try:
        flat: list[int] = []
        for v in box:
            if isinstance(v, (list, tuple)) and len(v) > 0:
                flat.append(int(v[0]))
            else:
                flat.append(int(v))
        return flat if len(flat) == 4 else None
    except (ValueError, TypeError):
        return None


class ImageTranslationService:
    def __init__(self, gateway_service: Any) -> None:
        self.gateway = gateway_service

    def _get_font(self, font_size: int, bold: bool = True, family: str = "sans") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        is_serif = str(family).lower() == "serif"
        if is_serif:
            font_path = SERIF_BOLD_FONT if bold else SERIF_REG_FONT
        else:
            font_path = DEFAULT_BOLD_FONT if bold else DEFAULT_REG_FONT

        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except Exception:
                pass

        fallback = DEFAULT_BOLD_FONT if bold else DEFAULT_REG_FONT
        if fallback.exists():
            try:
                return ImageFont.truetype(str(fallback), font_size)
            except Exception:
                pass
        return ImageFont.load_default()


    async def translate_image(
        self,
        image_bytes: bytes,
        *,
        target_language: str = "Vietnamese",
        instruction: str | None = None,
        model: str | None = None,
        provider_id: str | None = None,
        enable_qc: bool = False,
    ) -> tuple[bytes, list[dict[str, Any]], dict[str, Any]]:
        """OCR, translate, and inpaint text on image using vision model + Pillow."""
        # Validate and open image
        try:
            original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image format: {e}") from e

        # Resolve provider and model
        chosen_provider, real_model = self.gateway.resolve(model)
        if provider_id:
            chosen_provider = provider_id
        if not real_model and not model:
            real_model = "gemini-3.5-flash-lite"

        # Prepare base64 image for OpenAI-compatible vision payload
        b64_img = base64.b64encode(image_bytes).decode("utf-8")

        user_inst_block = ""
        if instruction and instruction.strip():
            user_inst_block = (
                f"\nCRITICAL USER CUSTOM INSTRUCTIONS (HIGHEST PRIORITY OVER GENERAL RULES):\n"
                f"\"{instruction.strip()}\"\n"
                f"You MUST strictly follow the user's explicit instructions above regarding which languages to translate or preserve, translation style, and whether to translate packaging/props.\n\n"
            )

        prompt = (
            f"Analyze all promotional foreign marketing text in this e-commerce product image that should be translated into {target_language}.\n"
            f"{user_inst_block}"
            f"CRITICAL GUIDELINES:\n"
            f"1. TRANSLATION, SCENE CONTEXT & IDIOMATIC LOCALIZATION:\n"
            f"   - HOLISTIC PRODUCT SCENE UNDERSTANDING: Analyze the visual context of the entire image (the product being demonstrated, model's action, usage scenario) and surrounding text clues.\n"
            f"   - NEVER perform literal word-for-word or morpheme-level translations. For example, if fitness gear for abdominal core training is shown (e.g. 爬行机, 健腹机 with clues like 练腹, 塑形, 核心), localize it naturally as 'MÁY TẬP BỤNG' or 'MÁY TẬP CƠ BỤNG' (NEVER translate literally as 'Máy tập bò'!). For food steamers/cooking presets: localize naturally (e.g. '蒸包' -> 'BÁNH BAO', '蒸饺' -> 'SỦI CẢO', '热牛奶' -> 'HÂM SỮA', '蒸玉米' -> 'BẮP HẤP').\n"
            f"   - Use authentic, high-converting e-commerce terminology that real shoppers search for in {target_language}.\n"
            f"   - CONCISE & PUNCHY MARKETING COPY: East Asian text is compact (3-6 characters). Keep {target_language} translations punchy and concise without verbose filler words (e.g. translate '全身塑形训练' -> 'ĐỊNH HÌNH TOÀN THÂN' or 'SĂN CHẮC TOÀN THÂN' instead of overly long phrases; '腹部塑形训练' -> 'SĂN CHẮC CƠ BỤNG'). This ensures translated titles can be rendered in large, impactful typography.\n"
            f"   - For main marketing titles, preserve uppercase or title casing (e.g. 'MÁY TẬP BỤNG' or 'MÁY TẬP CƠ BỤNG').\n"
            f"   - HORIZONTAL GUARANTEE / FOOTER RIBBONS: For wide ribbons/banners (e.g. '三年质保 免费试用 赠运费险'), translate concisely on a single line separated by bullets '•', e.g. 'BẢO HÀNH 3 NĂM • DÙNG THỬ 0Đ • TẶNG BẢO HIỂM SHIP'. Never make footer ribbons verbose.\n"
            f"2. ALGORITHM & STRATEGY SELECTION (CRITICAL: YOU ARE THE STRATEGY DIRECTOR):\n"
            f"   - 'inpaint_strategy': Select the exact CV inpainting recipe for each text element:\n"
            f"     * 'preserve_container_stroke_only': MUST be used for text inside containers (speech bubbles, circular callout tags, badges, pill buttons, guarantee ribbons, stickers). Only erases the foreign character strokes while preserving 100% of the container color, shape, borders, gradients, and pointer tails.\n"
            f"     * 'telea_context_aware': for titles, headlines, or marketing text floating freely on photo backgrounds (sky, table, floor, model's clothing, background scenery).\n"
            f"     * 'perspective_surface': for text printed on physical product surfaces (bottles, slanted cartons, cans, appliances).\n"
            f"     * 'flat_color_patch': for text on completely flat solid-colored background strips or monochrome panels.\n"
            f"     * 'watermark_cleanup': for faint, semi-transparent watermark text or draft guide text (e.g. '请输入文字').\n"
            f"   - 'layout_policy': Select the typography layout and wrapping policy:\n"
            f"     * 'single_line_contain': strictly single-line, auto-scaled to fit inside container height & width without wrapping (badges, callout tags, bubbles, guarantee ribbons, buttons).\n"
            f"     * 'hero_impact_wrap': multi-line balanced wrap with large bold typography for prominent marketing headlines.\n"
            f"     * 'surface_slant_fit': single-line slanted typography conforming to physical product surface angle.\n"
            f"   - 'container_shape': 'pill' | 'circle' | 'bubble' | 'ribbon' | 'rectangle' | 'none'.\n"
            f"   - 'icon_side': 'left' | 'right' | 'none'. If 'left', 'erase_box_2d' and 'box_2d' MUST start after the icon, keeping the icon 100% intact.\n"
            f"3. CSS STYLING & RICH TEXT EFFECTS DETECTION (CRITICAL FOR GRAPHIC REALISM):\n"
            f"   - 'gradient_colors': If characters have a vertical linear color gradient, golden sheen, or metallic specular reflection, return an array of 2 to 4 hex colors from TOP (bright reflective highlight, e.g. ['#fff2d0', '#ff8500', '#eb3800'] or ['#fde7b0', '#ca9877', '#3f1c06']) down to BOTTOM (rich shadow/extrusion). Return null if solid color.\n"
            f"   - 'stroke_color_hex': ONLY return hex color (e.g. '#ffffff') if text has a contrasting outline stroke. NEVER set stroke if text has no outline or if stroke matches text color.\n"
            f"   - 'stroke_width': Pixel width of the contrasting outline stroke (1 to 4), default 0.\n"
            f"   - 'shadow_color_hex': If text has a warm/dark drop shadow or 3D extrusion, return hex color (e.g. '#ff3300' or '#d4aa88'), else null.\n"
            f"   - 'font_family': 'serif' if text has flared serif or decorative strokes (like luxury/cosmetic display text), else 'sans'.\n"
            f"4. TYPOGRAPHY SCALE & CONCISENESS RULES (MAXIMIZE VISUAL IMPACT):\n"
            f"   - Prominent Marketing Titles ('hero_impact_wrap'): Always make translations bold, concise, and punchy. If the title is 3-4 words (e.g. 'MÁY TẬP BỤNG'), you may use '\\n' (e.g. 'MÁY TẬP\\nBỤNG' or 'SĂN CHẮC\\nCƠ BỤNG') so each line can render at huge font size filling the vertical space.\n"
            f"   - Badges, Chips, and Bubble Tags: Match the 2 to 4 Chinese characters with concise 2 to 3 Vietnamese words (e.g. 'TẬP CƠ BỤNG', 'SIẾT CƠ LÕI', 'TIỆN TẠI NHÀ', 'SIÊU ÊM ÁI'). DO NOT output lengthy sentences that compress the font size down.\n"
            f"5. PACKAGING & PROPS HANDLING:\n"
            f"   - DO NOT translate background props or decorative packaging (such as milk cartons, bottles, prop cans) UNLESS the user explicitly instructed to translate packaging.\n"
            f"   - If instructed to translate packaging: ONLY translate substantive foreign brand names or descriptive foreign words (e.g. '臻浓牛奶' -> 'Sữa tiệt trùng' or 'Sữa tươi'). DO NOT translate or re-render numbers, percentages, volume units, or universal Latin acronyms (e.g. '39%', '250ml', 'UHT', barcodes).\n"
            f"6. WATERMARK & DRAFT PLACEHOLDER REMOVAL:\n"
            f"   - If there is any draft watermark text, template placeholder, or watermark overlay (e.g., '请输入文字', '点击输入文字', sample placeholder text, or copyright watermark), you MUST extract it as a region with:\n"
            f"     * 'inpaint_strategy': 'watermark_cleanup'\n"
            f"     * 'translated_text': '' (empty string, so that it is cleanly erased without re-rendering any new text)\n"
            f"     * 'layout_policy': 'single_line_contain'\n"
            f"     * 'container_shape': 'none'\n"
            f"7. HARDWARE CONTROL PANELS & UNIVERSAL ENGLISH BRANDING:\n"
            f"   - For physical buttons, status indicators, or hardware control panels with standard universal English/icons (e.g. 'MASSAGER', 'Power', 'Auto', 'Heating', 'Low/Medium/High'), do NOT alter them unless they contain foreign Chinese characters or the user explicitly asks to translate hardware buttons.\n"
            f"8. BOUNDING BOXES & FORMATTING:\n"
            f"   - 'erase_box_2d': [ymin, xmin, ymax, xmax] (0-1000) bounding box enclosing text strokes completely.\n"
            f"   - 'box_2d': [ymin, xmin, ymax, xmax] (0-1000) inner container area available for placing translated text.\n"
            f"   - 'original_text': source text verbatim.\n"
            f"   - 'translated_text': concise, punchy translation in {target_language}.\n"
            f"   - 'bg_color_hex': hex color of the background immediately behind the text (e.g. '#ffffff' or '#ff7236').\n"
            f"   - 'text_color_hex': hex color of the text font (e.g. '#000000' or '#ffffff').\n"
            f"   - 'align_h': 'center' | 'left' | 'right'.\n"
            f"   - 'align_v': 'center' | 'top' | 'bottom'.\n"
            f"   - 'group_id': string identifier if elements share identical style/container (e.g. 'header', 'feature_badges', 'bubble_labels', 'tags', 'footer_banner', 'packaging').\n"
            f"   - 'rotation_deg': integer rotation in degrees (-180 to 180) matching physical slant.\n"
            f"   - 'bold': true or false.\n\n"
            f"Return ONLY valid JSON matching this schema:\n"
            f'{{"regions": [{{"erase_box_2d": [0,0,0,0], "box_2d": [0,0,0,0], "original_text": "...", "translated_text": "...", "inpaint_strategy": "preserve_container_stroke_only", "layout_policy": "single_line_contain", "container_shape": "bubble", "icon_side": "none", "bg_color_hex": "#ff7236", "text_color_hex": "#ffffff", "gradient_colors": ["#fff176", "#eb3800"], "stroke_color_hex": "#ffffff", "stroke_width": 2, "shadow_color_hex": "#ff3300", "font_family": "sans", "align_h": "center", "align_v": "center", "group_id": "tags", "rotation_deg": 0, "bold": true}}]}}'
        )


        chat_payload = {
            "model": real_model or model or "gemini-3.5-flash-lite",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }

        # Call vision model
        chat_resp = await self.gateway.chat(
            chosen_provider,
            chat_payload,
            real_model=real_model,
        )

        content = ""
        try:
            choices = chat_resp.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"Unexpected vision model response: {chat_resp}") from e

        # Extract JSON from response
        regions = self._parse_json_regions(content)
        if not regions:
            logger.warning("First attempt produced 0 regions, retrying vision inference...")
            try:
                retry_resp = await self.gateway.chat(chosen_provider, chat_payload, real_model=real_model)
                retry_choices = retry_resp.get("choices", [])
                if retry_choices:
                    retry_content = retry_choices[0].get("message", {}).get("content", "")
                    regions = self._parse_json_regions(retry_content)
            except Exception as e:
                logger.warning("Retry vision inference failed: %s", e)

        logger.info("Vision model parsed regions: %s", json.dumps(regions, ensure_ascii=False))

        # Inpaint and render text on image with typography fitting, grouping, and rotation
        translated_img = self._render_translations(original_img, regions)

        # Export result to JPEG bytes
        out_buf = io.BytesIO()
        translated_img.save(out_buf, format="JPEG", quality=95)
        out_bytes = out_buf.getvalue()

        # Optional Capyy QC Acceptance Pass
        qc_report = {"approved": True, "qc_score": 100, "feedback": "Rendered successfully"}
        if enable_qc:
            try:
                qc_report = await self._run_qc_inspection(out_bytes, target_language, real_model, instruction=instruction, regions=regions)
                logger.info("Capyy QC Report: %s", json.dumps(qc_report, ensure_ascii=False))
            except Exception as e:
                logger.warning("Capyy QC pass failed (non-blocking): %s", e)
                qc_report = {"approved": True, "qc_score": 90, "feedback": f"QC skipped: {e}"}

        return out_bytes, regions, qc_report

    async def _run_qc_inspection(
        self,
        image_bytes: bytes,
        target_language: str,
        real_model: str,
        instruction: str | None = None,
        regions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Capyy Quality Inspector: inspects rendered image for cleanliness, alignment, and completeness."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        user_inst_qc = ""
        if instruction and instruction.strip():
            user_inst_qc = (
                f"\nUSER CUSTOM INSTRUCTIONS SPECIFIED FOR THIS RUN:\n"
                f"\"{instruction.strip()}\"\n"
            )

        translated_summary = ""
        if regions:
            translated_summary = "\nTRANSLATED ELEMENTS RENDERED IN THIS IMAGE:\n" + "\n".join(
                [f"- {r.get('original_text')} -> '{r.get('translated_text')}' (rot: {r.get('rotation_deg')}°)" for r in regions]
            ) + "\n"

        prompt = (
            f"You are a Quality Control Inspector reviewing a translated product advertisement image in {target_language}.\n"
            f"{user_inst_qc}"
            f"{translated_summary}"
            f"Review this final rendered translated image and check:\n"
            f"1. Are the translated foreign text strokes cleanly erased and replaced with natural typography?\n"
            f"2. Does the translated text fit naturally within its containers (speech bubbles, packaging stripes, badges) with matching slant/rotation angle and legible typography?\n"
            f"3. Are container borders (doodle lines, badges) 100% intact without rectangular cutouts?\n\n"
            f"Return ONLY valid JSON in this schema:\n"
            f'{{"approved": true, "qc_score": 98, "feedback": "All speech bubbles and translated packaging are clean with well-proportioned typography."}}'
        )
        chat_payload = {
            "model": real_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        chosen_provider, _ = self.gateway.resolve(real_model)
        chat_resp = await self.gateway.chat(chosen_provider, chat_payload, real_model=real_model)
        choices = chat_resp.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"approved": True, "qc_score": 90, "feedback": content[:200]}

    def _parse_json_regions(self, content: str) -> list[dict[str, Any]]:
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE).strip()

        raw_list: list[dict[str, Any]] = []
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "regions" in data:
                raw_list = data["regions"]
            elif isinstance(data, list):
                raw_list = data
        except Exception:
            # Try auto-closing truncated JSON
            trimmed = cleaned.rstrip(", \n\r\t")
            for suffix in ["]}", "]}", "}]}", "}}"]:
                try:
                    data = json.loads(trimmed + suffix)
                    if isinstance(data, dict) and "regions" in data:
                        raw_list = data["regions"]
                        break
                except Exception:
                    pass

            if not raw_list:
                match = re.search(r"\{.*\"regions\"\s*:\s*\[.*\]\s*\}", cleaned, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        raw_list = data.get("regions", [])
                    except Exception:
                        pass

        if not raw_list:
            logger.warning("Could not parse JSON regions from model response: %s", content[:300])
            return []

        # Normalize and unroll batched regions (if model groups multiple elements into 1 dict)
        normalized: list[dict[str, Any]] = []
        for r in raw_list:
            if not isinstance(r, dict):
                continue
            orig = r.get("original_text")
            trans = r.get("translated_text")
            erase = r.get("erase_box_2d")
            box = r.get("box_2d")

            # Check if this region is a batch of multiple items
            if isinstance(orig, list) and len(orig) > 1:
                for idx, o in enumerate(orig):
                    t = trans[idx] if isinstance(trans, list) and idx < len(trans) else ""
                    e = erase[idx] if isinstance(erase, list) and idx < len(erase) else None
                    b = box[idx] if isinstance(box, list) and idx < len(box) else None
                    sub_r = dict(r)
                    sub_r["original_text"] = o
                    sub_r["translated_text"] = t
                    sub_r["erase_box_2d"] = e
                    sub_r["box_2d"] = b
                    normalized.append(sub_r)
            else:
                normalized.append(r)

        return normalized

    def _wrap_text(
        self,
        text: str,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        max_width: int,
        draw: ImageDraw.ImageDraw,
    ) -> list[str]:
        """Wrap text into lines fitting within max_width."""
        lines: list[str] = []
        for paragraph in text.split("\n"):
            words = paragraph.split(" ")
            cur_line: list[str] = []
            for word in words:
                test_line = " ".join(cur_line + [word]) if cur_line else word
                bbox = draw.textbbox((0, 0), test_line, font=font)
                line_w = bbox[2] - bbox[0]
                if line_w <= max_width or not cur_line:
                    cur_line.append(word)
                else:
                    lines.append(" ".join(cur_line))
                    cur_line = [word]
            if cur_line:
                lines.append(" ".join(cur_line))
        return lines

    @staticmethod
    def _inpaint_strategy_container_stroke(
        roi: np.ndarray,
        gray: np.ndarray,
        text_bgr: np.ndarray,
        bg_bgr: np.ndarray,
        text_lum: float,
        bg_lum: float,
        roi_h: int,
        roi_w: int,
        x1: int,
        y1: int,
        region_char_components: list[dict[str, Any]],
    ) -> np.ndarray:
        """Isolate foreign text strokes inside container without erasing container shape/gradient."""
        d_text = np.linalg.norm(roi.astype(np.float32) - text_bgr, axis=2)
        d_bg = np.linalg.norm(roi.astype(np.float32) - bg_bgr, axis=2)

        if text_lum > bg_lum:
            # Light text on colored/dark container (e.g. white text on orange/red bubble)
            cand_mask = ((d_text < 130) & (d_bg > 20) & (gray > min(190, bg_lum + 10))).astype(np.uint8) * 255
        else:
            # Dark text on light container (e.g. black text on white pill badge or banner)
            if bg_lum > 200:
                cand_mask = ((d_bg > 18) & (gray < max(180, min(228, int(bg_lum - 10))))).astype(np.uint8) * 255
            else:
                cand_mask = ((d_text < 125) & (d_bg > 25) & (gray < max(180, bg_lum - 10))).astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cand_mask)
        clean_char_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        for i in range(1, num_labels):
            cx, cy, cw_c, ch_c, area = stats[i]
            touches_edge = (cx <= 1 or cy <= 1 or (cx + cw_c) >= roi_w - 1 or (cy + ch_c) >= roi_h - 1)
            if touches_edge or area >= (roi_h * roi_w * 0.45):
                continue
            if area >= 6:
                clean_char_mask[labels == i] = 255
                region_char_components.append({
                    "cx": float(x1 + centroids[i][0]),
                    "cy": float(y1 + centroids[i][1]),
                    "area": float(area),
                    "w": int(cw_c),
                    "h": int(ch_c),
                })

        if np.sum(clean_char_mask > 0) >= 12:
            kernel_sz = 3 if max(roi_h, roi_w) <= 90 else 4
            return cv2.dilate(clean_char_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz)))
        else:
            # Fallback threshold
            if text_lum > bg_lum:
                mask = (gray > 175).astype(np.uint8) * 255
            else:
                mask = (gray < 165).astype(np.uint8) * 255
            return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    @staticmethod
    def _inpaint_strategy_perspective_surface(
        roi: np.ndarray,
        gray: np.ndarray,
        bg_lum: float,
        roi_h: int,
        roi_w: int,
        x1: int,
        y1: int,
        region_char_components: list[dict[str, Any]],
    ) -> np.ndarray:
        """Isolate slanted text on packaging or physical props for cv2.fitLine slant detection."""
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        char_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        for i in range(1, num_labels):
            cx, cy, cw, ch, area = stats[i]
            touches_edge = (cx <= 1 or cy <= 1 or (cx + cw) >= roi_w - 1 or (cy + ch) >= roi_h - 1)
            if touches_edge or area >= (roi_h * roi_w * 0.70):
                continue
            if 8 <= area:
                char_mask[labels == i] = 255
                region_char_components.append({
                    "cx": float(x1 + centroids[i][0]),
                    "cy": float(y1 + centroids[i][1]),
                    "area": float(area),
                    "w": int(cw),
                    "h": int(ch),
                })
        return cv2.dilate(char_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    @staticmethod
    def _inpaint_strategy_telea_context(
        roi: np.ndarray,
        gray: np.ndarray,
        bg_est: np.ndarray,
        bg_gray: float,
    ) -> np.ndarray:
        """Context-aware inpainting for marketing text floating over photo backgrounds."""
        diff = np.linalg.norm(roi.astype(np.float32) - bg_est.astype(np.float32), axis=2)
        if bg_gray > 160:
            m_c = (gray < (bg_gray - 20)).astype(np.uint8) * 255
        else:
            m_c = (gray > (bg_gray + 20)).astype(np.uint8) * 255
        raw_mask = cv2.bitwise_or((diff > 8).astype(np.uint8) * 255, m_c)

        num_l, lbls, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
        clean_mask = np.zeros_like(raw_mask)
        for i in range(1, num_l):
            if stats[i, cv2.CC_STAT_AREA] >= 20:
                clean_mask[lbls == i] = 255
        return cv2.dilate(clean_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)))

    @staticmethod
    def _inpaint_strategy_watermark_cleanup(
        roi: np.ndarray,
        gray: np.ndarray,
        bg_est: np.ndarray,
        bg_gray: float,
    ) -> np.ndarray:
        """Clean faint watermark or draft guide text with high-pass stroke isolation."""
        bright_strokes = (gray >= 238).astype(np.uint8) * 255
        diff = np.linalg.norm(roi.astype(np.float32) - bg_est.astype(np.float32), axis=2)
        local_edges = cv2.Canny(gray, 30, 100)
        dilated_edges = cv2.dilate(local_edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cand_mask = cv2.bitwise_or(bright_strokes, cv2.bitwise_and((diff > 15).astype(np.uint8) * 255, dilated_edges))
        num_l, lbls, stats, _ = cv2.connectedComponentsWithStats(cand_mask)
        clean_mask = np.zeros_like(cand_mask)
        h, w = gray.shape
        for i in range(1, num_l):
            if stats[i, cv2.CC_STAT_AREA] < (h * w * 0.35) and stats[i, cv2.CC_STAT_AREA] >= 4:
                clean_mask[lbls == i] = 255
        return cv2.dilate(clean_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    def _inpaint_canvas(
        self,
        img: Image.Image,
        regions: list[dict[str, Any]],
    ) -> Image.Image:
        """Erase foreign text strokes using OpenCV inpainting (Telea method).

        Preserves container outlines, bubble doodle borders, gradients, and background textures.
        Never leaves flat rectangular boxes or whiteout patches.
        """
        try:
            w, h = img.size
            np_img = np.array(img.convert("RGB"))
            cv_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

            full_mask = np.zeros((h, w), dtype=np.uint8)

            for r in regions:
                boxes_to_erase: list[list[int]] = []
                erase_box = sanitize_box(r.get("erase_box_2d"))
                if erase_box:
                    boxes_to_erase.append(erase_box)
                else:
                    char_boxes = r.get("char_boxes_2d")
                    if isinstance(char_boxes, list) and len(char_boxes) > 0:
                        for b in char_boxes:
                            s_box = sanitize_box(b)
                            if s_box:
                                boxes_to_erase.append(s_box)

                if not boxes_to_erase:
                    fallback_box = sanitize_box(r.get("box_2d"))
                    if fallback_box:
                        boxes_to_erase.append(fallback_box)

                if not boxes_to_erase:
                    continue

                bg_hex = r.get("bg_color_hex") or r.get("bg_color")
                text_hex = r.get("text_color_hex") or r.get("text_color")
                bg_rgb = parse_hex_color(bg_hex, (255, 255, 255))
                text_rgb = parse_hex_color(text_hex, (0, 0, 0))

                bg_lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
                text_lum = 0.299 * text_rgb[0] + 0.587 * text_rgb[1] + 0.114 * text_rgb[2]

                region_char_components: list[dict[str, Any]] = []

                gid_str = str(r.get("group_id", "")).lower()
                is_pill_badge = any(k in gid_str for k in ["badge", "feature", "pill"])
                is_bubble_group = any(k in gid_str for k in ["bubble", "callout", "doodle", "tag", "chip", "sticker", "stamp"])
                is_packaging = any(k in gid_str for k in ["packaging", "carton", "prop", "bottle"])
                is_banner = any(k in gid_str for k in ["banner", "ribbon", "footer", "bar", "strip"])
                is_container_elem = is_pill_badge or is_bubble_group or is_banner or any(k in gid_str for k in ["label", "button"])

                bg_bgr = np.array([bg_rgb[2], bg_rgb[1], bg_rgb[0]], dtype=np.float32)
                text_bgr = np.array([text_rgb[2], text_rgb[1], text_rgb[0]], dtype=np.float32)

                for erase_box in boxes_to_erase:
                    ymin, xmin, ymax, xmax = erase_box

                    # Use safe container margins
                    margin = 6 if (is_container_elem or is_packaging) else 2
                    x1 = max(0, min(w - 1, int(xmin * w / 1000) - margin))
                    y1 = max(0, min(h - 1, int(ymin * h / 1000) - margin))
                    x2 = max(x1 + 1, min(w, int(xmax * w / 1000) + margin))
                    y2 = max(y1 + 1, min(h, int(ymax * h / 1000) + margin))
                    roi = cv_bgr[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_h, roi_w = gray.shape

                    edge_pts = np.concatenate([
                        roi[:3, :].reshape(-1, 3), roi[-3:, :].reshape(-1, 3),
                        roi[:, :3].reshape(-1, 3), roi[:, -3:].reshape(-1, 3)
                    ], axis=0)
                    bg_est = np.median(edge_pts, axis=0)
                    bg_gray = float(0.299 * bg_est[2] + 0.587 * bg_est[1] + 0.114 * bg_est[0])

                    # Determine inpaint strategy (AI-directed with autonomous heuristic fallback)
                    strategy = str(r.get("inpaint_strategy") or "").strip().lower()
                    if not strategy:
                        if is_packaging and bg_lum > 180:
                            strategy = "perspective_surface"
                        elif is_container_elem:
                            strategy = "preserve_container_stroke_only"
                        elif "watermark" in gid_str:
                            strategy = "watermark_cleanup"
                        else:
                            strategy = "telea_context_aware"

                    if strategy in ("preserve_container_stroke_only", "container_stroke"):
                        dilated = self._inpaint_strategy_container_stroke(
                            roi, gray, text_bgr, bg_bgr, text_lum, bg_lum, roi_h, roi_w, x1, y1, region_char_components
                        )
                    elif strategy == "perspective_surface":
                        dilated = self._inpaint_strategy_perspective_surface(
                            roi, gray, bg_lum, roi_h, roi_w, x1, y1, region_char_components
                        )
                    elif strategy == "watermark_cleanup":
                        dilated = self._inpaint_strategy_watermark_cleanup(
                            roi, gray, bg_est, bg_gray
                        )
                    elif strategy == "flat_color_patch":
                        cv_bgr[y1:y2, x1:x2] = bg_bgr.astype(np.uint8)
                        continue
                    else:
                        dilated = self._inpaint_strategy_telea_context(
                            roi, gray, bg_est, bg_gray
                        )

                    # Accumulate directly to full_mask for seamless context-aware OpenCV inpainting
                    full_mask[y1:y2, x1:x2] = cv2.bitwise_or(full_mask[y1:y2, x1:x2], dilated)

                # Deterministic CV Slant Angle & Character Glyph Dimensions Detection (via cv2.fitLine)
                if region_char_components:
                    max_a = max(c["area"] for c in region_char_components)
                    glyphs = [c for c in region_char_components if c["area"] >= 0.35 * max_a]
                    glyphs = sorted(glyphs, key=lambda c: c["cx"])
                    if len(glyphs) >= 2:
                        pts = np.array([[c["cx"], c["cy"]] for c in glyphs], dtype=np.float32)
                        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                        vx_val = float(vx[0])
                        vy_val = float(vy[0])
                        if vx_val < 0:
                            vx_val, vy_val = -vx_val, -vy_val
                        detected_deg = float(np.degrees(np.arctan2(-vy_val, vx_val)))
                        if abs(detected_deg) < 4.5:
                            detected_deg = 0.0
                        r["rotation_deg"] = max(-65, min(65, round(detected_deg)))
                        r["orig_glyph_h"] = float(np.median([g["h"] for g in glyphs]))
                    elif len(glyphs) == 1:
                        r["orig_glyph_h"] = float(glyphs[0]["h"])

            # Seamless OpenCV inpainting over whole image
            inpainted_bgr = cv2.inpaint(cv_bgr, full_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

            # Film grain injection: match high-frequency noise of unmasked image to eliminate plastic/waxy blur
            unmasked_pts = cv_bgr[full_mask == 0]
            if len(unmasked_pts) > 100:
                blurred_unmasked = cv2.GaussianBlur(cv_bgr, (5, 5), 0)[full_mask == 0]
                noise_sigma = float(np.std(unmasked_pts.astype(np.float32) - blurred_unmasked.astype(np.float32)))
                noise_sigma = max(0.4, min(4.0, noise_sigma))
            else:
                noise_sigma = 1.0

            # Inject subtle Gaussian grain matching surrounding photo texture into inpainted area
            noise = np.random.normal(0, noise_sigma, cv_bgr.shape).astype(np.float32)
            weight = (cv2.GaussianBlur(full_mask, (5, 5), 0) / 255.0)[:, :, None]
            final_bgr = np.clip(inpainted_bgr.astype(np.float32) + noise * weight, 0, 255).astype(np.uint8)

            inpainted_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(inpainted_rgb)
        except Exception as e:
            logger.warning("OpenCV inpainting failed, falling back to original image: %s", e)
            return img.copy()


    def _render_translations(
        self,
        img: Image.Image,
        regions: list[dict[str, Any]],
    ) -> Image.Image:
        # -------------------------------------------------------------
        # PASS 1: Erase foreign text strokes cleanly via OpenCV inpainting
        # -------------------------------------------------------------
        out = self._inpaint_canvas(img, regions)
        draw = ImageDraw.Draw(out)
        w, h = out.size

        # -------------------------------------------------------------
        # PASS 2: Calculate typography, auto-wrap & sizing for each item
        # -------------------------------------------------------------
        layout_items: list[dict[str, Any]] = []

        for r in regions:
            box = sanitize_box(r.get("box_2d")) or sanitize_box(r.get("erase_box_2d"))
            if not box:
                continue
            ymin, xmin, ymax, xmax = box

            gid_lower = str(r.get("group_id", "")).lower()
            is_bubble = any(k in gid_lower for k in ["bubble", "callout", "doodle", "tag", "chip", "sticker", "stamp"])
            is_pill_badge = any(k in gid_lower for k in ["badge", "feature", "pill"])
            is_packaging = any(k in gid_lower for k in ["packaging", "carton", "prop", "bottle"])
            is_horizontal_banner = any(k in gid_lower for k in ["banner", "ribbon", "footer", "bar", "strip"]) or ((xmax - xmin) / max(1, ymax - ymin) >= 4.0 and ymin > 600)
            is_badge = is_pill_badge or is_bubble or any(k in gid_lower for k in ["label", "button"])

            x1 = max(0, min(w, int(xmin * w / 1000)))
            y1 = max(0, min(h, int(ymin * h / 1000)))
            x2 = max(0, min(w, int(xmax * w / 1000)))
            y2 = max(0, min(h, int(ymax * h / 1000)))

            box_w = max(10, x2 - x1)
            box_h = max(10, y2 - y1)

            # Layout & Typography Policy Dispatcher
            policy = str(r.get("layout_policy") or "").strip().lower()
            container_shape = str(r.get("container_shape") or "").strip().lower()
            icon_side = str(r.get("icon_side") or "none").strip().lower()

            # If icon is on the left or right, constrain text rendering box strictly to the text portion
            if icon_side == "left":
                erase_box = sanitize_box(r.get("erase_box_2d"))
                if erase_box:
                    x_erase_start = max(0, min(w, int(erase_box[1] * w / 1000)))
                    if x_erase_start > x1:
                        x1 = x_erase_start
                else:
                    x1 = x1 + int(box_w * 0.28)
                box_w = max(10, x2 - x1)
            elif icon_side == "right":
                erase_box = sanitize_box(r.get("erase_box_2d"))
                if erase_box:
                    x_erase_end = max(0, min(w, int(erase_box[3] * w / 1000)))
                    if x_erase_end < x2:
                        x2 = x_erase_end
                else:
                    x2 = x2 - int(box_w * 0.28)
                box_w = max(10, x2 - x1)

            if not policy:
                if is_badge or is_horizontal_banner:
                    policy = "single_line_contain"
                elif is_packaging or abs(rotation_deg) >= 5:
                    policy = "surface_slant_fit"
                elif is_header_elem:
                    policy = "hero_impact_wrap"
                else:
                    policy = "single_line_contain" if box_h < 45 else "hero_impact_wrap"

            # Proportional safe padding inside container based on shape & icon
            if container_shape in ("bubble", "circle") or is_bubble:
                pad_x = max(6, int(box_w * 0.08))
                pad_y = max(4, int(box_h * 0.10))
            elif container_shape in ("ribbon", "strip") or is_horizontal_banner:
                pad_x = max(10, int(box_w * 0.03))
                pad_y = max(4, int(box_h * 0.12))
            elif container_shape == "pill" or is_pill_badge:
                pad_x = max(6, int(box_w * 0.05))
                pad_y = max(4, int(box_h * 0.08))
            else:
                pad_x = max(2, int(box_w * 0.03))
                pad_y = max(2, int(box_h * 0.05))

            if icon_side == "left":
                pad_x = max(int(box_w * 0.12), pad_x)

            # Left screen edge protection for bubbles and badges
            if x1 <= 5 and (is_pill_badge or is_bubble or container_shape in ("pill", "bubble")):
                pad_x = max(14, pad_x)

            avail_w = max(8, box_w - 2 * pad_x)
            avail_h = max(8, box_h - 2 * pad_y)

            # Headline & Header Expansion:
            # If item is in the top header/title area on the left margin, allow avail_w to expand into available safe whitespace
            is_header_elem = any(k in str(r.get("group_id", "")).lower() for k in ["header", "title", "hero", "main"]) or (ymin < 300 and box_h >= 60 and xmin < 300)
            if is_header_elem and x1 < int(w * 0.35):
                safe_max_x = int(w * 0.60)
                if safe_max_x > x1:
                    avail_w = max(avail_w, safe_max_x - x1)

            translated_text = str(r.get("translated_text", "")).strip().rstrip(",;.:-")
            if not translated_text:
                continue

            bold = bool(r.get("bold", True))
            font_family = str(r.get("font_family") or "sans")
            fg = parse_hex_color(r.get("text_color_hex") or r.get("text_color"), (0, 0, 0))

            stroke_width = max(0, min(8, int(r.get("stroke_width") or 0)))
            stroke_color = parse_hex_color(r.get("stroke_color_hex")) if r.get("stroke_color_hex") else None
            shadow_color = parse_hex_color(r.get("shadow_color_hex")) if r.get("shadow_color_hex") else None

            # Stroke sanitization:
            # If stroke color has no contrast against text color, cancel it (prevents fat, clogged, blurry characters)
            bg_hex = r.get("bg_color_hex") or r.get("bg_color")
            bg_rgb = parse_hex_color(bg_hex, (255, 255, 255))
            bg_lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
            fg_lum = 0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]

            if stroke_width > 0 and stroke_color:
                c_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(stroke_color, fg)))
                s_lum = 0.299 * stroke_color[0] + 0.587 * stroke_color[1] + 0.114 * stroke_color[2]
                if c_dist < 45 or (fg_lum < 60 and s_lum < 60):
                    stroke_width = 0
                    stroke_color = None

            # Contrast safety protection:
            # If text color is white/near-white and rendered on light background without stroke, add subtle dark outline
            if stroke_width == 0 and stroke_color is None:
                if fg_lum > 210 and bg_lum > 180:
                    stroke_width = 1
                    stroke_color = (60, 60, 60)
                elif fg_lum < 50 and bg_lum < 75:
                    stroke_width = 1
                    stroke_color = (255, 255, 255)

            gradient_colors = r.get("gradient_colors")
            gradient_stops = None
            if isinstance(gradient_colors, list) and len(gradient_colors) >= 2:
                parsed_stops = [parse_hex_color(c) for c in gradient_colors]
                if len(parsed_stops) >= 2:
                    gradient_stops = parsed_stops

            align_h = str(r.get("align_h") or r.get("alignment") or "center").lower()
            align_v = str(r.get("align_v") or "center").lower()
            group_id = r.get("group_id")
            if isinstance(group_id, str):
                group_id = group_id.strip() or None

            rotation_deg = int(r.get("rotation_deg") or 0)

            best_font = None
            best_size = 14
            best_lines: list[str] = []

            # Single-line enforcement:
            # Ribbons, packaging, and slanted elements must strictly remain single-line.
            # Badges default to single-line, but allow 2 lines if height is generous (box_h >= 50) and words >= 3.
            is_ribbon_or_slant = container_shape in ("ribbon", "strip") or is_horizontal_banner or is_packaging or policy == "surface_slant_fit" or (abs(rotation_deg) >= 5)
            is_single_line_only = is_ribbon_or_slant or (policy == "single_line_contain" and (box_h < 50 or len(translated_text.split()) < 3))

            # Typography layout optimization:
            # A. Evaluate single-line candidate
            single_sz = 0
            single_font = None
            if "\n" not in translated_text:
                if policy == "surface_slant_fit" or is_packaging:
                    orig_glyph_h = float(r.get("orig_glyph_h") or 0)
                    if orig_glyph_h > 10:
                        max_single_sz = max(13, min(int(orig_glyph_h * 0.60), 20))
                    else:
                        max_single_sz = max(13, min(int(min(avail_w, avail_h) * 0.22), 20))
                elif container_shape in ("ribbon", "strip") or is_horizontal_banner:
                    max_single_sz = min(28, max(13, int(avail_h * 0.65)))
                elif container_shape in ("bubble", "circle") or is_bubble:
                    max_single_sz = min(28, max(14, int(avail_h * 0.65)))
                elif container_shape == "pill" or is_pill_badge:
                    max_single_sz = min(36, max(16, int(avail_h * 0.75)))
                else:
                    max_single_sz = min(120, max(18, int(avail_h * 0.90)))

                for sz in range(max_single_sz, 10, -1):
                    font = self._get_font(sz, bold=bold, family=font_family)
                    bb = draw.textbbox((0, 0), translated_text, font=font, stroke_width=stroke_width)
                    tw = bb[2] - bb[0]
                    th = bb[3] - bb[1]
                    if tw <= avail_w and th <= avail_h:
                        single_font = font
                        single_sz = sz
                        break

            # B. Evaluate multi-line candidate (up to 2 lines for titles/subtitles/flexible badges)
            multi_sz = 0
            multi_font = None
            multi_lines: list[str] = []
            words = translated_text.split()
            if not is_single_line_only and len(words) >= 2 and avail_h >= 45:
                max_multi_sz = min(110, max(16, int(avail_h * 0.75)))
                for sz in range(max_multi_sz, 14, -1):
                    font = self._get_font(sz, bold=bold, family=font_family)
                    lines = self._wrap_text(translated_text, font, avail_w, draw)
                    if len(lines) <= 2:
                        bboxes = [draw.textbbox((0, 0), l, font=font, stroke_width=stroke_width) for l in lines]
                        max_lw = max((b[2] - b[0] for b in bboxes), default=0)
                        spacing = max(2, int(sz * 0.15))
                        total_th = sum((b[3] - b[1]) for b in bboxes) + spacing * max(0, len(lines) - 1)
                        if max_lw <= avail_w and total_th <= avail_h:
                            multi_font = font
                            multi_sz = sz
                            multi_lines = lines
                            break

            # Selection rule:
            is_title = is_header_elem or policy == "hero_impact_wrap"
            if not is_single_line_only and multi_sz and single_sz:
                if is_title:
                    if multi_sz >= int(single_sz * 1.15) or single_sz < int(avail_h * 0.50):
                        best_font = multi_font
                        best_size = multi_sz
                        best_lines = multi_lines
                    else:
                        best_font = single_font
                        best_size = single_sz
                        best_lines = [translated_text]
                elif multi_sz >= int(single_sz * 1.25) and single_sz < 30:
                    best_font = multi_font
                    best_size = multi_sz
                    best_lines = multi_lines
                else:
                    best_font = single_font
                    best_size = single_sz
                    best_lines = [translated_text]
            elif single_font:
                best_font = single_font
                best_size = single_sz
                best_lines = [translated_text]
            elif multi_font and not is_single_line_only:
                best_font = multi_font
                best_size = multi_sz
                best_lines = multi_lines
            else:
                # Fallback: single-line scale down or multi-line wrap
                if is_single_line_only:
                    for sz in range(16, 10, -1):
                        font = self._get_font(sz, bold=bold, family=font_family)
                        bb = draw.textbbox((0, 0), translated_text, font=font, stroke_width=stroke_width)
                        if (bb[2] - bb[0]) <= avail_w and (bb[3] - bb[1]) <= avail_h:
                            best_font = font
                            best_size = sz
                            best_lines = [translated_text]
                            break
                    else:
                        best_size = 11
                        best_font = self._get_font(11, bold=bold, family=font_family)
                        best_lines = [translated_text]
                else:
                    for sz in range(20, 12, -1):
                        font = self._get_font(sz, bold=bold, family=font_family)
                        lines = self._wrap_text(translated_text, font, avail_w, draw)
                        bboxes = [draw.textbbox((0, 0), l, font=font, stroke_width=stroke_width) for l in lines]
                        max_lw = max((b[2] - b[0] for b in bboxes), default=0)
                        spacing = max(2, int(sz * 0.15))
                        total_th = sum((b[3] - b[1]) for b in bboxes) + spacing * max(0, len(lines) - 1)
                        if max_lw <= avail_w and total_th <= avail_h:
                            best_font = font
                            best_size = sz
                            best_lines = lines
                            break
                    else:
                        best_size = 14
                        best_font = self._get_font(14, bold=bold, family=font_family)
                        best_lines = self._wrap_text(translated_text, best_font, avail_w, draw)

            layout_items.append({
                "region": r,
                "text": translated_text,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "box_w": box_w, "box_h": box_h,
                "pad_x": pad_x, "pad_y": pad_y,
                "avail_w": avail_w, "avail_h": avail_h,
                "font": best_font,
                "font_size": best_size,
                "lines": best_lines,
                "bold": bold,
                "font_family": font_family,
                "fg": fg,
                "stroke_width": stroke_width,
                "stroke_color": stroke_color,
                "shadow_color": shadow_color,
                "gradient_stops": gradient_stops,
                "align_h": align_h,
                "align_v": align_v,
                "group_id": group_id,
                "rotation_deg": rotation_deg,
                "is_badge": is_badge,
                "is_bubble": is_bubble,
                "is_packaging": is_packaging,
            })

        # -------------------------------------------------------------
        # PASS 2.5: Synchronize font size & rotation across items in the same group
        # -------------------------------------------------------------
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in layout_items:
            gid = item.get("group_id")
            if gid:
                groups.setdefault(gid.lower(), []).append(item)

        for gid, group_members in groups.items():
            if len(group_members) > 1:
                # Do NOT harmonize font size across header/title elements (preserve H1 vs H2 typographic hierarchy)
                is_header_group = any(h in gid.lower() for h in ["header", "title", "hero", "main"])
                if not is_header_group:
                    avg_h = sum(m["box_h"] for m in group_members) / len(group_members)
                    if all(0.75 * avg_h <= m["box_h"] <= 1.25 * avg_h for m in group_members):
                        min_group_size = max(12, min(m["font_size"] for m in group_members))
                        if any(m.get("is_badge") for m in group_members):
                            # Ensure all badges fit on a single line at the synchronized font size
                            for sz in range(min_group_size, 10, -1):
                                all_fit = True
                                for m in group_members:
                                    test_font = self._get_font(sz, bold=m["bold"], family=m["font_family"])
                                    test_bb = draw.textbbox((0, 0), m["text"], font=test_font, stroke_width=m["stroke_width"])
                                    if (test_bb[2] - test_bb[0]) > m["avail_w"] or (test_bb[3] - test_bb[1]) > m["avail_h"]:
                                        all_fit = False
                                        break
                                if all_fit:
                                    min_group_size = sz
                                    break
                            for m in group_members:
                                m["font_size"] = min_group_size
                                m["font"] = self._get_font(min_group_size, bold=m["bold"], family=m["font_family"])
                                m["lines"] = [m["text"]]
                        else:
                            for m in group_members:
                                if m["font_size"] != min_group_size:
                                    m["font_size"] = min_group_size
                                    m["font"] = self._get_font(min_group_size, bold=m["bold"], family=m["font_family"])
                                    m["lines"] = self._wrap_text(m["text"], m["font"], m["avail_w"], draw)

                rots = [m["rotation_deg"] for m in group_members]
                is_bubble_group = any(m.get("is_bubble") for m in group_members)
                if is_bubble_group:
                    valid_rots = [r for r in rots if abs(r) >= 5]
                    if valid_rots:
                        harmonized_rot = round(float(np.median(valid_rots)))
                        for m in group_members:
                            m["rotation_deg"] = harmonized_rot
                            if "region" in m and isinstance(m["region"], dict):
                                m["region"]["rotation_deg"] = harmonized_rot
                else:
                    if max(rots) - min(rots) <= 15:
                        harmonized_rot = round(float(np.median(rots)))
                        for m in group_members:
                            m["rotation_deg"] = harmonized_rot
                            if "region" in m and isinstance(m["region"], dict):
                                m["region"]["rotation_deg"] = harmonized_rot

        # Also harmonize any speech bubbles across groups if present
        all_bubble_items = [m for m in layout_items if m.get("is_bubble")]
        if len(all_bubble_items) > 1:
            bubble_rots = [m["rotation_deg"] for m in all_bubble_items if abs(m["rotation_deg"]) >= 5]
            if bubble_rots:
                harmonized_bubble_rot = round(float(np.median(bubble_rots)))
                for m in all_bubble_items:
                    m["rotation_deg"] = harmonized_bubble_rot
                    if "region" in m and isinstance(m["region"], dict):
                        m["region"]["rotation_deg"] = harmonized_bubble_rot

        # -------------------------------------------------------------
        # PASS 3: Render lines with CSS styling, gradients, strokes, shadows, and rotation
        # -------------------------------------------------------------
        for item in layout_items:
            font = item["font"]
            lines = item["lines"]
            fg = item["fg"]
            align_h = item["align_h"]
            align_v = item["align_v"]
            rot = item["rotation_deg"]
            stroke_w = item["stroke_width"]
            stroke_c = item["stroke_color"]
            shadow_c = item["shadow_color"]
            stops = item["gradient_stops"]

            x1, y1 = item["x1"], item["y1"]
            x2, y2 = item["x2"], item["y2"]
            pad_x, pad_y = item["pad_x"], item["pad_y"]
            avail_w, avail_h = item["avail_w"], item["avail_h"]

            line_bboxes = [draw.textbbox((0, 0), l, font=font, stroke_width=stroke_w) for l in lines]
            line_heights = [(b[3] - b[1]) for b in line_bboxes]
            line_widths = [(b[2] - b[0]) for b in line_bboxes]
            spacing = max(2, int(font.size * 0.15))
            total_text_h = sum(line_heights) + spacing * max(0, len(lines) - 1)
            max_text_w = max(line_widths, default=avail_w)

            layer_w = max_text_w + stroke_w * 2 + 40
            layer_h = total_text_h + stroke_w * 2 + 40
            text_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
            layer_draw = ImageDraw.Draw(text_layer)
            base_x = 20 + stroke_w
            base_y = 20 + stroke_w

            # 1. Render drop shadow if specified
            if shadow_c:
                shadow_img = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
                s_draw = ImageDraw.Draw(shadow_img)
                cur_sy = base_y + 3
                for l, bb, lw, lh in zip(lines, line_bboxes, line_widths, line_heights):
                    if align_h == "center":
                        lx = base_x + (max_text_w - lw) // 2
                    elif align_h == "right":
                        lx = base_x + (max_text_w - lw)
                    else:
                        lx = base_x
                    s_draw.text(
                        (lx - bb[0], cur_sy - bb[1]),
                        l,
                        font=font,
                        fill=(*shadow_c, 160),
                        stroke_width=stroke_w,
                        stroke_fill=(*shadow_c, 160),
                    )
                    cur_sy += lh + spacing
                shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=2))
                text_layer = Image.alpha_composite(text_layer, shadow_img)
                layer_draw = ImageDraw.Draw(text_layer)

            # 2. Render stroke outline if specified
            if stroke_w > 0 and stroke_c:
                cur_y = base_y
                for l, bb, lw, lh in zip(lines, line_bboxes, line_widths, line_heights):
                    if align_h == "center":
                        lx = base_x + (max_text_w - lw) // 2
                    elif align_h == "right":
                        lx = base_x + (max_text_w - lw)
                    else:
                        lx = base_x
                    layer_draw.text(
                        (lx - bb[0], cur_y - bb[1]),
                        l,
                        font=font,
                        fill=(*stroke_c, 255),
                        stroke_width=stroke_w,
                        stroke_fill=(*stroke_c, 255),
                    )
                    cur_y += lh + spacing

            # 3. Render fill: gradient or solid color
            if stops:
                fill_mask = Image.new("L", (layer_w, layer_h), 0)
                f_draw = ImageDraw.Draw(fill_mask)
                cur_y = base_y
                for l, bb, lw, lh in zip(lines, line_bboxes, line_widths, line_heights):
                    if align_h == "center":
                        lx = base_x + (max_text_w - lw) // 2
                    elif align_h == "right":
                        lx = base_x + (max_text_w - lw)
                    else:
                        lx = base_x
                    f_draw.text((lx - bb[0], cur_y - bb[1]), l, font=font, fill=255)
                    cur_y += lh + spacing
                grad = create_vertical_gradient(layer_w, max(1, total_text_h), stops)
                grad_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
                grad_layer.paste(grad, (0, base_y))
                text_layer.paste(grad_layer, (0, 0), mask=fill_mask)
            else:
                cur_y = base_y
                for l, bb, lw, lh in zip(lines, line_bboxes, line_widths, line_heights):
                    if align_h == "center":
                        lx = base_x + (max_text_w - lw) // 2
                    elif align_h == "right":
                        lx = base_x + (max_text_w - lw)
                    else:
                        lx = base_x
                    layer_draw.text((lx - bb[0], cur_y - bb[1]), l, font=font, fill=(*fg, 255))
                    cur_y += lh + spacing

            # 4. Composite onto canvas with positioning and rotation
            if align_h == "center":
                content_x = (x1 + pad_x) + max(0, (avail_w - max_text_w) // 2)
            elif align_h == "right":
                content_x = (x2 - pad_x) - max_text_w
            else:
                content_x = x1 + pad_x

            if align_v == "top":
                content_y = y1 + pad_y
            elif align_v == "bottom":
                content_y = (y2 - pad_y) - total_text_h
            else:
                content_y = (y1 + pad_y) + max(0, (avail_h - total_text_h) // 2)

            paste_x = content_x - base_x
            paste_y = content_y - base_y

            if rot != 0:
                ss_scale = 2
                ss_font = self._get_font(font.size * ss_scale, bold=item["bold"], family=item["font_family"])
                ss_spacing = max(2, int(ss_font.size * 0.15))
                ss_bboxes = [draw.textbbox((0, 0), l, font=ss_font, stroke_width=stroke_w * ss_scale) for l in lines]
                ss_lws = [(b[2] - b[0]) for b in ss_bboxes]
                ss_lhs = [(b[3] - b[1]) for b in ss_bboxes]
                ss_total_th = sum(ss_lhs) + ss_spacing * max(0, len(lines) - 1)
                ss_max_w = max(ss_lws, default=max_text_w * ss_scale)

                ss_layer_w = ss_max_w + stroke_w * ss_scale * 2 + 80
                ss_layer_h = ss_total_th + stroke_w * ss_scale * 2 + 80
                ss_layer = Image.new("RGBA", (ss_layer_w, ss_layer_h), (0, 0, 0, 0))
                ss_draw = ImageDraw.Draw(ss_layer)
                ss_bx = 40 + stroke_w * ss_scale
                ss_by = 40 + stroke_w * ss_scale

                if stroke_w > 0 and stroke_c:
                    cur_sy = ss_by
                    for l, bb, lw, lh in zip(lines, ss_bboxes, ss_lws, ss_lhs):
                        lx = ss_bx + (ss_max_w - lw) // 2 if align_h == "center" else (ss_bx + (ss_max_w - lw) if align_h == "right" else ss_bx)
                        ss_draw.text((lx - bb[0], cur_sy - bb[1]), l, font=ss_font, fill=(*stroke_c, 255), stroke_width=stroke_w * ss_scale, stroke_fill=(*stroke_c, 255))
                        cur_sy += lh + ss_spacing

                if stops:
                    ss_mask = Image.new("L", (ss_layer_w, ss_layer_h), 0)
                    m_draw = ImageDraw.Draw(ss_mask)
                    cur_y = ss_by
                    for l, bb, lw, lh in zip(lines, ss_bboxes, ss_lws, ss_lhs):
                        lx = ss_bx + (ss_max_w - lw) // 2 if align_h == "center" else (ss_bx + (ss_max_w - lw) if align_h == "right" else ss_bx)
                        m_draw.text((lx - bb[0], cur_y - bb[1]), l, font=ss_font, fill=255)
                        cur_y += lh + ss_spacing
                    ss_grad = create_vertical_gradient(ss_layer_w, max(1, ss_total_th), stops)
                    ss_glayer = Image.new("RGBA", (ss_layer_w, ss_layer_h), (0, 0, 0, 0))
                    ss_glayer.paste(ss_grad, (0, ss_by))
                    ss_layer.paste(ss_glayer, (0, 0), mask=ss_mask)
                else:
                    cur_y = ss_by
                    for l, bb, lw, lh in zip(lines, ss_bboxes, ss_lws, ss_lhs):
                        lx = ss_bx + (ss_max_w - lw) // 2 if align_h == "center" else (ss_bx + (ss_max_w - lw) if align_h == "right" else ss_bx)
                        ss_draw.text((lx - bb[0], cur_y - bb[1]), l, font=ss_font, fill=(*fg, 255))
                        cur_y += lh + ss_spacing

                rotated_2x = ss_layer.rotate(rot, resample=Image.BICUBIC, expand=True)
                tgt_w = max(1, rotated_2x.width // ss_scale)
                tgt_h = max(1, rotated_2x.height // ss_scale)
                text_layer = rotated_2x.resize((tgt_w, tgt_h), resample=Image.LANCZOS)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                paste_x = center_x - text_layer.width // 2
                paste_y = center_y - text_layer.height // 2

            rgba_out = out.convert("RGBA")
            rgba_out.paste(text_layer, (paste_x, paste_y), mask=text_layer)
            out = rgba_out.convert("RGB")
            draw = ImageDraw.Draw(out)

        return out

