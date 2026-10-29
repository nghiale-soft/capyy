from __future__ import annotations

import logging
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

from ..deps import check_local_auth

router = APIRouter(prefix="/v1/videos", tags=["videos"])
logger = logging.getLogger("gateway.routes.videos")


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

        out_bytes, segments = await video_service.translate_video(
            video_bytes,
            target_language=target_language,
            voice=voice,
            instruction=instruction,
            model=model,
            cover_old_sub=cover_old_sub,
        )

        headers = {
            "Content-Disposition": 'inline; filename="translated_dubbed.mp4"',
            "X-Segments-Count": str(len(segments)),
        }
        return Response(content=out_bytes, media_type="video/mp4", headers=headers)
    except Exception as e:
        logger.exception("Video translation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Video translation error: {e}") from e
