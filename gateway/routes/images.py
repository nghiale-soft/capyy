from __future__ import annotations

import base64
import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field

from ..deps import check_local_auth

router = APIRouter(prefix="/v1/images", tags=["images"])
logger = logging.getLogger("gateway.routes.images")


class ImageTranslateJsonRequest(BaseModel):
    image_base64: str | None = Field(default=None, description="Base64-encoded image string")
    image_url: str | None = Field(default=None, description="Direct URL of image to download and translate")
    target_language: str = Field(default="Vietnamese", description="Target translation language")
    instruction: str | None = Field(default=None, description="Custom translation instructions (e.g. 'chỉ dịch tiếng Trung, dịch cả bao bì')")
    description: str | None = Field(default=None, description="Alias for instruction")
    model: str | None = Field(default=None, description="Vision model to use (default: gemini-3.5-flash-lite)")
    provider: str | None = Field(default=None, description="Provider ID (default: gemini)")
    response_format: str = Field(default="image", description="'image' for raw JPEG binary, 'json' for base64 response")
    enable_qc: bool = Field(default=False, description="Run Capyy final acceptance inspection pass")


@router.post("/translate")
async def translate_image_endpoint(
    request: Request,
    _auth: None = Depends(check_local_auth),
    file: UploadFile | None = File(None),
    target_language: str = Form("Vietnamese"),
    instruction: str | None = Form(None),
    description: str | None = Form(None),
    model: str | None = Form(None),
    provider: str | None = Form(None),
    response_format: str = Form("image"),
    enable_qc: bool = Form(False),
) -> Response:
    """Translate text inside an image using Vision OCR and local Pillow inpainting.

    Accepts either:
    1. Multipart form-data with file upload (`file=@image.jpg`).
    2. JSON payload with `image_base64` or `image_url`.
    """
    image_bytes: bytes | None = None
    req_target_lang = target_language
    req_instruction = instruction or description
    req_model = model
    req_provider = provider
    req_format = response_format
    req_enable_qc = enable_qc

    # Check if request is JSON
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            json_req = ImageTranslateJsonRequest(**body)
            req_target_lang = json_req.target_language
            req_instruction = json_req.instruction or json_req.description
            req_model = json_req.model
            req_provider = json_req.provider
            req_format = json_req.response_format
            req_enable_qc = json_req.enable_qc

            if json_req.image_base64:
                # Strip data URL prefix if present
                raw_b64 = json_req.image_base64
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]
                image_bytes = base64.b64decode(raw_b64)
            elif json_req.image_url:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(json_req.image_url)
                    resp.raise_for_status()
                    image_bytes = resp.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse JSON image request: {e}") from e
    elif file is not None:
        image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="No image provided. Upload a file or provide 'image_base64'/'image_url' in JSON body.",
        )

    image_service = request.app.state.image_service
    try:
        translated_bytes, regions, qc_report = await image_service.translate_image(
            image_bytes,
            target_language=req_target_lang,
            instruction=req_instruction,
            model=req_model,
            provider_id=req_provider,
            enable_qc=req_enable_qc,
        )
    except Exception as e:
        logger.exception("Image translation failed")
        raise HTTPException(status_code=500, detail=f"Image translation failed: {e}") from e

    if req_format.lower() == "json":
        b64_result = base64.b64encode(translated_bytes).decode("utf-8")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content={
                "data": {
                    "image_base64": f"data:image/jpeg;base64,{b64_result}",
                    "format": "jpeg",
                    "regions_count": len(regions),
                    "regions": regions,
                    "qc_report": qc_report,
                }
            }
        )

    return Response(
        content=translated_bytes,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": 'inline; filename="translated.jpg"',
            "X-Capyy-QC-Approved": str(qc_report.get("approved", True)),
            "X-Capyy-QC-Score": str(qc_report.get("qc_score", 100)),
        },
    )
