from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx
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
from pydantic import BaseModel, Field

from ..deps import check_local_auth
from ..services.image_store import (
    ImageSessionError,
    ImageSessionExpired,
    ImageSessionNotFound,
    ImageSessionStore,
    ImageSessionTooLarge,
)

router = APIRouter(prefix="/v1/images", tags=["images"])
logger = logging.getLogger("gateway.routes.images")


class ImageTranslateJsonRequest(BaseModel):
    image_base64: str | None = Field(
        default=None, description="Base64-encoded image string"
    )
    image_url: str | None = Field(
        default=None, description="Direct URL of image to download and translate"
    )
    target_language: str = Field(
        default="Vietnamese", description="Target translation language"
    )
    instruction: str | None = Field(
        default=None,
        description="Custom translation instructions (e.g. 'chỉ dịch tiếng Trung, dịch cả bao bì')",
    )
    description: str | None = Field(default=None, description="Alias for instruction")
    model: str | None = Field(
        default=None, description="Vision model to use (default: gemini-3.5-flash-lite)"
    )
    provider: str | None = Field(
        default=None, description="Provider ID (default: gemini)"
    )
    response_format: str = Field(
        default="image",
        description="'image' for raw JPEG binary, 'json' for base64 response",
    )
    enable_qc: bool = Field(
        default=False, description="Run Capyy final acceptance inspection pass"
    )


class ImageRepairJsonRequest(BaseModel):
    instruction: str = Field(
        min_length=1,
        max_length=2000,
        description="What to repair, e.g. 'chữ 蛋 còn sót ở góc dưới trái, xoá giúp'",
    )
    region_ids: list[int] | None = Field(
        default=None,
        max_length=100,
        description="Stable region ids from the translate response",
    )
    bbox_2d: list[int] | None = Field(
        default=None,
        description="Optional [ymin, xmin, ymax, xmax] (0-1000) to select overlapping regions",
    )
    base: str = Field(
        default="original",
        description="Pixel base for re-rendering. Only 'original' is supported.",
    )
    response_format: str = Field(default="json", description="'json' or 'image'")
    enable_qc: bool = Field(
        default=False, description="Run the optional LLM inspection too"
    )
    idempotency_key: str | None = Field(default=None, max_length=128)


def _image_store(request: Request) -> ImageSessionStore:
    store = getattr(request.app.state, "image_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Image session storage is not configured on this server",
        )
    return store


def _version_url(image_id: str, version: int) -> str:
    return f"/v1/images/{image_id}/content?version={int(version)}"


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    image_id = str(state.get("id") or "")
    versions = []
    for item in state.get("versions") or []:
        versions.append(
            {
                "version": int(item.get("version") or 0),
                "created_at": item.get("created_at"),
                "instruction": item.get("instruction"),
                "region_count": len(item.get("regions") or []),
                "url": _version_url(image_id, int(item.get("version") or 0)),
            }
        )
    return {
        "id": image_id,
        "version": int(state.get("version") or 0),
        "created_at": state.get("created_at"),
        "expires_at": state.get("expires_at"),
        "target_language": state.get("target_language"),
        "instruction": state.get("instruction"),
        "model": state.get("model"),
        "provider": state.get("provider"),
        "regions": state.get("regions") or [],
        "versions": versions,
        "url": _version_url(image_id, int(state.get("version") or 0)),
    }


def _repair_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ImageSessionExpired):
        return HTTPException(status_code=410, detail=str(exc))
    if isinstance(exc, ImageSessionNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ImageSessionTooLarge):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, ImageSessionError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=502, detail=f"Repair failed: {exc}")


@router.post("/translate")
async def translate_image_endpoint(
    request: Request,
    _auth: None = Depends(check_local_auth),
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    target_language: str = Form("Vietnamese"),
    target_lang: str | None = Form(None),
    instruction: str | None = Form(None),
    description: str | None = Form(None),
    model: str | None = Form(None),
    provider: str | None = Form(None),
    response_format: str = Form("image"),
    enable_qc: bool = Form(False),
) -> Response:
    """Translate text inside an image using Vision OCR and local Pillow inpainting.

    Accepts either:
    1. Multipart form-data with file upload (`file=@image.jpg` or `image=@image.jpg`).
    2. JSON payload with `image_base64` or `image_url`.
    """
    image_bytes: bytes | None = None
    upload_file = file or image
    req_target_lang = target_lang or target_language
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
            raise HTTPException(
                status_code=400, detail=f"Failed to parse JSON image request: {e}"
            ) from e
    elif upload_file is not None:
        image_bytes = await upload_file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="No image provided. Upload a file or provide 'image_base64'/'image_url' in JSON body.",
        )

    store = getattr(request.app.state, "image_store", None)
    if store is not None and len(image_bytes) > store.max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {store.max_bytes} byte storage limit",
        )

    image_service = request.app.state.image_service
    session: dict[str, Any] | None = None
    try:
        if store is not None:
            (
                translated_bytes,
                regions,
                qc_report,
                session,
            ) = await image_service.translate_image_session(
                image_bytes,
                target_language=req_target_lang,
                instruction=req_instruction,
                model=req_model,
                provider_id=req_provider,
                enable_qc=req_enable_qc,
            )
        else:
            translated_bytes, regions, qc_report = await image_service.translate_image(
                image_bytes,
                target_language=req_target_lang,
                instruction=req_instruction,
                model=req_model,
                provider_id=req_provider,
                enable_qc=req_enable_qc,
            )
    except ImageSessionTooLarge as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except Exception as e:
        logger.exception("Image translation failed")
        raise HTTPException(
            status_code=500, detail=f"Image translation failed: {e}"
        ) from e

    if req_format.lower() == "json":
        b64_result = base64.b64encode(translated_bytes).decode("utf-8")
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content={
                "data": {
                    "id": session.get("id") if session else None,
                    "version": session.get("version") if session else None,
                    "expires_at": session.get("expires_at") if session else None,
                    "image_base64": f"data:image/jpeg;base64,{b64_result}",
                    "format": "jpeg",
                    "regions_count": len(regions),
                    "regions": regions,
                    "qc_report": qc_report,
                }
            }
        )

    headers = {
        "Content-Disposition": 'inline; filename="translated.jpg"',
        "X-Capyy-QC-Approved": str(qc_report.get("approved", True)),
        "X-Capyy-QC-Score": str(qc_report.get("qc_score", 100)),
    }
    if session:
        headers["X-Capyy-Image-Id"] = str(session.get("id"))
        headers["X-Capyy-Image-Version"] = str(session.get("version"))
        headers["X-Capyy-Image-Expires-At"] = str(session.get("expires_at"))
    return Response(
        content=translated_bytes,
        media_type="image/jpeg",
        headers=headers,
    )


@router.post("/{image_id}/repair")
async def repair_image_endpoint(
    image_id: str,
    payload: ImageRepairJsonRequest,
    request: Request,
    _auth: None = Depends(check_local_auth),
) -> Response:
    """Repair one or more regions of an existing translation (user-initiated)."""
    _image_store(request)
    bbox = payload.bbox_2d
    if bbox is not None:
        if len(bbox) != 4:
            raise HTTPException(
                status_code=400,
                detail="bbox_2d must contain exactly 4 values [ymin, xmin, ymax, xmax]",
            )
        if any(value < 0 or value > 1000 for value in bbox):
            raise HTTPException(
                status_code=400, detail="bbox_2d values must be within 0-1000"
            )
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise HTTPException(
                status_code=400,
                detail="bbox_2d must satisfy ymin < ymax and xmin < xmax",
            )

    try:
        (
            out_bytes,
            regions,
            qc_report,
            state,
        ) = await request.app.state.image_service.repair_image(
            image_id,
            instruction=payload.instruction.strip(),
            region_ids=payload.region_ids,
            bbox_2d=bbox,
            base=payload.base,
            enable_qc=payload.enable_qc,
            idempotency_key=payload.idempotency_key,
        )
    except Exception as e:
        logger.warning("Image repair failed for %s: %s", image_id, e)
        raise _repair_error(e) from e

    version = int(state.get("version") or 0)
    headers = {
        "Content-Disposition": 'inline; filename="repaired.jpg"',
        "X-Capyy-Image-Id": str(state.get("id") or image_id),
        "X-Capyy-Image-Version": str(version),
        "X-Capyy-QC-Approved": str(qc_report.get("approved", True)),
        "X-Capyy-QC-Score": str(qc_report.get("qc_score", 100)),
    }
    if payload.response_format.lower() == "image":
        return Response(content=out_bytes, media_type="image/jpeg", headers=headers)

    from fastapi.responses import JSONResponse

    b64_result = base64.b64encode(out_bytes).decode("utf-8")
    return JSONResponse(
        content={
            "data": {
                "id": state.get("id") or image_id,
                "version": version,
                "parent_version": max(1, version - 1),
                "expires_at": state.get("expires_at"),
                "image_base64": f"data:image/jpeg;base64,{b64_result}",
                "format": "jpeg",
                "regions_count": len(regions),
                "regions": regions,
                "qc_report": qc_report,
                "url": _version_url(str(state.get("id") or image_id), version),
            }
        }
    )


@router.get("/{image_id}")
async def get_image_session(
    image_id: str,
    request: Request,
    _auth: None = Depends(check_local_auth),
) -> Response:
    """Return editable session metadata, including every stored version."""
    store = _image_store(request)
    try:
        state = await asyncio.to_thread(store.get, image_id)
    except Exception as e:
        raise _repair_error(e) from e
    from fastapi.responses import JSONResponse

    return JSONResponse(content={"data": _public_state(state)})


@router.get("/{image_id}/content")
async def get_image_content(
    image_id: str,
    request: Request,
    version: int | None = None,
    _auth: None = Depends(check_local_auth),
) -> Response:
    """Return the current (or requested) rendered JPEG for a stored session."""
    store = _image_store(request)
    try:
        image_bytes = await asyncio.to_thread(
            store.get_version_bytes, image_id, version
        )
        state = await asyncio.to_thread(store.get, image_id)
    except Exception as e:
        raise _repair_error(e) from e
    resolved_version = version or int(state.get("version") or 0)
    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": 'inline; filename="translated.jpg"',
            "X-Capyy-Image-Id": image_id,
            "X-Capyy-Image-Version": str(resolved_version),
        },
    )


@router.delete("/{image_id}")
async def delete_image_session(
    image_id: str,
    request: Request,
    _auth: None = Depends(check_local_auth),
) -> dict[str, Any]:
    """Purge a stored session and all of its rendered versions."""
    store = _image_store(request)
    try:
        deleted = await asyncio.to_thread(store.delete, image_id)
    except Exception as e:
        raise _repair_error(e) from e
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Image session not found: {image_id}"
        )
    return {"deleted": True, "id": image_id}
