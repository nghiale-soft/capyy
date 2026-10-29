from __future__ import annotations

import json
import logging
import math
import subprocess
import tempfile
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)

from ..deps import check_and_reserve_credit, check_local_auth, deduct_credit

router = APIRouter(prefix="/v1/videos", tags=["videos"])
logger = logging.getLogger("gateway.routes.videos")


def _probe_video_duration(video_bytes: bytes, segments: list[dict[str, Any]] | None = None) -> float:
    """Best-effort probing of video duration in seconds."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
            f.write(video_bytes)
            f.flush()
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", f.name],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(probe.stdout or "{}")
            dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
            if dur > 0.0:
                return dur
    except Exception:
        pass
    if segments:
        max_ms = max((s.get("end_ms", 0) for s in segments), default=0)
        if max_ms > 0:
            return float(max_ms) / 1000.0
    return 10.0


@router.post(
    "/translate",
    dependencies=[Depends(check_local_auth)],
    summary="Translate & Dub e-commerce product video with AI speech & subtitles",
)
async def translate_video_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Video file (MP4, MOV, WebM)"),
    target_language: str = Form("Vietnamese"),
    voice: str = Form("female", description="'female' (Hoai My) or 'male' (Nam Minh)"),
    instruction: str | None = Form(None),
    model: str | None = Form(None),
    cover_old_sub: bool = Form(True),
) -> Response:
    video_service = getattr(request.app.state, "video_service", None)
    if not video_service:
        raise HTTPException(status_code=500, detail="Video translation service is not initialized")

    try:
        video_bytes = await file.read()
        if not video_bytes:
            raise HTTPException(status_code=400, detail="Empty video file uploaded")

        # 1. Pre-check credit balance (minimum 5 credits for first 10s block)
        user_id = await check_and_reserve_credit(request, resource_type="video", required_credits=5)

        out_bytes, segments = await video_service.translate_video(
            video_bytes,
            target_language=target_language,
            voice=voice,
            instruction=instruction,
            model=model,
            cover_old_sub=cover_old_sub,
        )

        # 2. Calculate duration and deduct credits (5 credits per 10-second block)
        vid_duration = _probe_video_duration(video_bytes, segments)
        blocks_10s = max(1, math.ceil(vid_duration / 10.0))
        credits_to_deduct = blocks_10s * 5

        tx_ref = f"vid_{uuid.uuid4().hex[:12]}"
        await deduct_credit(
            user_id=user_id,
            amount=credits_to_deduct,
            reference_id=tx_ref,
            description=f"Cappy AI dịch video ({vid_duration:.1f}s - {credits_to_deduct} credits)",
            request=request,
        )

        headers = {
            "Content-Disposition": 'inline; filename="translated_dubbed.mp4"',
            "X-Segments-Count": str(len(segments)),
            "X-Credits-Deducted": str(credits_to_deduct),
            "X-Video-Duration": f"{vid_duration:.2f}",
        }
        return Response(content=out_bytes, media_type="video/mp4", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Video translation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Video translation error: {e}") from e
