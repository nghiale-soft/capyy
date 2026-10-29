from __future__ import annotations

import io
import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("gateway.services.image_qa")


def _sanitize_box(box: Any) -> list[int] | None:
    """Mirror of image_service.sanitize_box to avoid a circular import."""
    if not isinstance(box, (list, tuple)):
        return None
    if len(box) == 1 and isinstance(box[0], (list, tuple)):
        box = box[0]
    if len(box) != 4:
        return None
    try:
        flat: list[int] = []
        for value in box:
            if isinstance(value, (list, tuple)) and len(value) > 0:
                flat.append(int(value[0]))
            else:
                flat.append(int(value))
        return flat if len(flat) == 4 else None
    except (ValueError, TypeError):
        return None


def _to_gray(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _inner_source_ink(
    gray: np.ndarray,
    *,
    edge_trim: int,
) -> tuple[np.ndarray, int, float]:
    """Return a conservative mask of text-like ink inside a crop.

    The outer ``edge_trim`` pixels are removed first. Container borders live on
    the crop edge; trimming them prevents the border itself from being reported
    as leftover text while still leaving the body of a glyph that touched the
    border. Large frame-like components are then discarded.
    """
    h, w = gray.shape
    if h <= edge_trim * 2 + 4 or w <= edge_trim * 2 + 4:
        return np.zeros_like(gray, dtype=np.uint8), 0, 0.0

    # Use the dominant color of the whole crop, not the border, as the local
    # background.  For a text box inside a white speech bubble the border is
    # the wood/product behind the bubble; using it would classify the whole
    # white bubble as "ink" and make every translated label look unerased.
    bg_gray = float(np.median(gray))

    ink = (np.abs(gray.astype(np.float32) - bg_gray) > 42).astype(np.uint8) * 255
    if edge_trim > 0:
        ink[:edge_trim, :] = 0
        ink[-edge_trim:, :] = 0
        ink[:, :edge_trim] = 0
        ink[:, -edge_trim:] = 0

    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    clean = np.zeros_like(ink)
    inner_h, inner_w = h - 2 * edge_trim, w - 2 * edge_trim
    inner_area = max(1, inner_h * inner_w)
    for idx in range(1, num):
        x, y, cw, ch, area = stats[idx]
        if area < 6:
            continue
        # A full frame or a huge flat region is a container, not a glyph.
        if area >= 0.55 * inner_area:
            continue
        if cw >= 0.92 * inner_w and ch >= 0.92 * inner_h:
            continue
        if cw >= 0.92 * inner_w and area >= 0.12 * inner_area:
            continue
        if ch >= 0.92 * inner_h and area >= 0.12 * inner_area:
            continue
        # Container outlines, carton edges and label frames touch the crop
        # boundary.  They are deliberately preserved by the renderer, so they
        # must not be counted as surviving source text.  A glyph inside the
        # box normally has a small margin; the model's box_2d is used first for
        # exactly that reason.
        if (
            x <= edge_trim
            or y <= edge_trim
            or x + cw >= w - edge_trim
            or y + ch >= h - edge_trim
        ):
            continue
        fill = area / float(max(1, cw * ch))
        # Drop long thin lines (bubble borders) and large sparse graphics
        # (product artwork) while keeping compact glyph-sized components.
        if cw > 0.60 * inner_w and ch > 0.60 * inner_h:
            continue
        if cw > 0.60 * inner_w and ch < 0.45 * inner_h:
            continue
        if ch > 0.60 * inner_h and cw < 0.35 * inner_w:
            continue
        if fill < 0.24 and (cw > 0.35 * inner_w or ch > 0.35 * inner_h):
            continue
        clean[labels == idx] = 255
    return clean, int(np.count_nonzero(clean)), bg_gray


def _box_to_pixels(
    box: list[int], width: int, height: int, margin: int = 2
) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = box
    x1 = max(0, min(width - 1, int(xmin * width / 1000) - margin))
    y1 = max(0, min(height - 1, int(ymin * height / 1000) - margin))
    x2 = max(x1 + 1, min(width, int(xmax * width / 1000) + margin))
    y2 = max(y1 + 1, min(height, int(ymax * height / 1000) + margin))
    return x1, y1, x2, y2


def inspect_translation(
    original_bytes: bytes,
    output_bytes: bytes,
    regions: list[dict[str, Any]],
    *,
    target_language: str = "Vietnamese",
) -> dict[str, Any]:
    """Cheap, deterministic post-render inspection.

    This is intentionally *not* an LLM QA pass. It looks for the most common
    mechanical failure: source-language ink that survived inpainting unchanged.
    It can be run on every image for effectively zero token cost.

    The returned shape is compatible with the existing ``qc_report`` contract
    (``approved`` / ``qc_score`` / ``feedback``) and adds ``issues`` plus
    ``deterministic`` for clients that want region-level detail.
    """
    issues: list[dict[str, Any]] = []
    try:
        original = Image.open(io.BytesIO(original_bytes)).convert("RGB")
        output = Image.open(io.BytesIO(output_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - QA must never block delivery
        logger.warning("Deterministic QA could not decode images: %s", exc)
        return {
            "approved": True,
            "qc_score": 100,
            "feedback": f"Deterministic QA skipped: {exc}",
            "deterministic": True,
            "issues": [],
            "qa_error": str(exc),
        }

    if original.size != output.size:
        return {
            "approved": False,
            "qc_score": 0,
            "feedback": "Rendered image size differs from the source image",
            "deterministic": True,
            "issues": [
                {
                    "region_id": None,
                    "type": "size_mismatch",
                    "severity": "high",
                    "bbox_2d": None,
                }
            ],
        }

    orig_gray = _to_gray(original)
    out_gray = _to_gray(output)
    width, height = original.size

    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        region_id = region.get("region_id", index)
        box = _sanitize_box(region.get("box_2d")) or _sanitize_box(
            region.get("erase_box_2d")
        )
        if not box:
            continue

        x1, y1, x2, y2 = _box_to_pixels(box, width, height, margin=2)
        crop_h, crop_w = y2 - y1, x2 - x1
        if crop_h < 12 or crop_w < 12:
            continue

        edge_trim = max(2, min(6, min(crop_h, crop_w) // 18))
        source_ink, source_area, bg_gray = _inner_source_ink(
            orig_gray[y1:y2, x1:x2], edge_trim=edge_trim
        )
        if source_area < 12:
            continue

        out_crop = out_gray[y1:y2, x1:x2].astype(np.float32)
        orig_crop = orig_gray[y1:y2, x1:x2].astype(np.float32)

        # A pixel is "leftover" only when it was source ink, is still visually
        # unchanged from the source, and remains far from the local background.
        unchanged = np.abs(out_crop - orig_crop) < 7
        still_ink = np.abs(out_crop - bg_gray) > 30
        leftover = (source_ink > 0) & unchanged & still_ink
        leftover_area = int(np.count_nonzero(leftover))
        # Scale the noise floor with the source ink, not the crop area: a small
        # leftover glyph in a large rectangle must still be visible to QA.
        if leftover_area < max(8, int(source_area * 0.08)):
            continue

        ratio = leftover_area / float(max(1, source_area))
        translated = str(region.get("translated_text", "")).strip()
        # Rendering new text can legitimately reuse the source color, so require
        # stronger evidence when the region is supposed to contain a translation.
        # Real results routinely keep tiny anti-aliased fragments; one whole
        # leftover glyph is still well above this threshold.
        threshold = 0.30 if translated else 0.20
        if ratio < threshold:
            continue

        ys, xs = np.where(leftover)
        if ys.size and xs.size:
            ly1, ly2 = int(ys.min()) + y1, int(ys.max()) + y1
            lx1, lx2 = int(xs.min()) + x1, int(xs.max()) + x1
            issue_box = [
                int(ly1 * 1000 / max(1, height)),
                int(lx1 * 1000 / max(1, width)),
                int((ly2 + 1) * 1000 / max(1, height)),
                int((lx2 + 1) * 1000 / max(1, width)),
            ]
        else:
            issue_box = box

        issues.append(
            {
                "region_id": region_id,
                "type": "leftover_text",
                "severity": "high" if ratio >= 0.55 else "medium",
                "bbox_2d": issue_box,
                "leftover_ratio": round(float(ratio), 3),
                "leftover_pixels": leftover_area,
                "source_ink_pixels": source_area,
                "message": "Source text ink appears unchanged after inpainting",
            }
        )

    high = sum(1 for issue in issues if issue.get("severity") == "high")
    medium = sum(1 for issue in issues if issue.get("severity") == "medium")
    low = sum(1 for issue in issues if issue.get("severity") == "low")
    score = max(0, 100 - high * 25 - medium * 10 - low * 3)
    approved = high == 0 and medium == 0
    if issues:
        feedback = (
            f"Deterministic QA flagged {len(issues)} region(s) in {target_language}"
        )
    else:
        feedback = "Deterministic QA passed"

    return {
        "approved": approved,
        "qc_score": score,
        "feedback": feedback,
        "deterministic": True,
        "issues": issues,
    }


def inspect_repair_effect(
    before_bytes: bytes,
    after_bytes: bytes,
    regions: list[dict[str, Any]],
    target_region_ids: list[int],
) -> dict[str, Any]:
    """Check that a repair actually changed at least one targeted region."""
    try:
        before = _to_gray(Image.open(io.BytesIO(before_bytes)))
        after = _to_gray(Image.open(io.BytesIO(after_bytes)))
    except Exception as exc:  # noqa: BLE001 - QA must never block delivery
        return {
            "changed": False,
            "issues": [
                {
                    "type": "repair_check_failed",
                    "severity": "medium",
                    "message": str(exc),
                }
            ],
        }
    if before.shape != after.shape:
        return {
            "changed": True,
            "issues": [],
        }

    height, width = before.shape
    targets = set(target_region_ids)
    changed = False
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        region_id = region.get("region_id", index)
        if targets and region_id not in targets:
            continue
        box = _sanitize_box(region.get("erase_box_2d")) or _sanitize_box(
            region.get("box_2d")
        )
        if not box:
            continue
        x1, y1, x2, y2 = _box_to_pixels(box, width, height, margin=1)
        diff = np.abs(
            before[y1:y2, x1:x2].astype(np.int16) - after[y1:y2, x1:x2].astype(np.int16)
        )
        if diff.size and (
            float(diff.mean()) >= 0.5 or int(np.count_nonzero(diff > 25)) >= 8
        ):
            changed = True
            break

    issues: list[dict[str, Any]] = []
    if not changed:
        issues.append(
            {
                "type": "no_visible_change",
                "severity": "medium",
                "message": "Repair produced no visible change in the targeted region",
            }
        )
    return {"changed": changed, "issues": issues}
