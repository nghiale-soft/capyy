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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..deps import check_and_reserve_credit, check_local_auth, deduct_credit
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


class ImageTestJsonRequest(BaseModel):
    image_base64: str | None = Field(
        default=None, description="Base64-encoded image string"
    )
    image_url: str | None = Field(
        default=None, description="Direct URL of image to download and test"
    )
    target_language: str = Field(
        default="Vietnamese", description="Target translation language"
    )
    instruction: str | None = Field(
        default=None,
        description="Custom translation instructions (e.g. 'dịch sang tiếng Việt, giữ nguyên logo')",
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
        description="'image' for raw JPEG binary on pass, 'json' for base64 response",
    )
    pass_threshold: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Minimum QC score required to pass (default 80)",
    )
    auto_repair: bool = Field(
        default=True,
        description="Automatically trigger repair in background when test fails",
    )


class ImageFeedbackJsonRequest(BaseModel):
    image_id: str | None = Field(
        default=None, description="ID of the image session to feedback/repair"
    )
    status: str = Field(
        default="failed",
        description="'failed' (yêu cầu sửa ảnh) hoặc 'passed' (ảnh đạt yêu cầu)",
    )
    approved: bool | None = Field(
        default=None,
        description="Optional boolean: False = failed/repair, True = passed",
    )
    reason: str | None = Field(
        default=None,
        description="Lý do chưa đạt / mô tả lỗi cụ thể để AI sửa",
    )
    instruction: str | None = Field(
        default=None, description="Alias for reason"
    )
    region_ids: list[int] | None = Field(
        default=None,
        description="Optional list of specific region IDs to repair",
    )
    response_format: str = Field(
        default="image",
        description="'image' for raw JPEG binary of repaired image, 'json' for base64 response",
    )
    enable_qc: bool = Field(
        default=True,
        description="Run QC evaluation on the repaired image",
    )


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

    # Pre-check credit balance before AI inpainting (1 credit required)
    user_id = await check_and_reserve_credit(request, resource_type="image", required_credits=1)

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

    # Deduct 1 credit for completed image translation
    tx_ref = f"img_{session.get('id') if session else 'tx_' + str(abs(hash(translated_bytes[:32])))}"
    await deduct_credit(
        user_id=user_id,
        amount=1,
        reference_id=tx_ref,
        description="Cappy AI dịch ảnh sản phẩm",
        request=request,
    )

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


@router.post("/test")
async def test_image_endpoint(
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
    pass_threshold: int = Form(80),
    auto_repair: bool = Form(True),
) -> Response:
    """Test and self-evaluate an image translation.

    - If QC PASSES (score >= pass_threshold and no severe defects):
      Returns translated JPEG binary directly (or JSON if requested).
    - If QC FAILS:
      Returns failure JSON detailing issues and defect areas, and automatically
      triggers an asynchronous self-repair task in the background.
    """
    image_bytes: bytes | None = None
    upload_file = file or image
    req_target_lang = target_lang or target_language
    req_instruction = instruction or description
    req_model = model
    req_provider = provider
    req_format = response_format
    req_pass_threshold = pass_threshold
    req_auto_repair = auto_repair

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            json_req = ImageTestJsonRequest(**body)
            req_target_lang = json_req.target_language
            req_instruction = json_req.instruction or json_req.description
            req_model = json_req.model
            req_provider = json_req.provider
            req_format = json_req.response_format
            req_pass_threshold = json_req.pass_threshold
            req_auto_repair = json_req.auto_repair

            if json_req.image_base64:
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
                status_code=400, detail=f"Failed to parse JSON test request: {e}"
            ) from e
    elif upload_file is not None:
        image_bytes = await upload_file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="No image provided. Upload a file or provide 'image_base64'/'image_url' in JSON body.",
        )

    # Pre-check credit balance (1 coin for test run)
    user_id = await check_and_reserve_credit(request, resource_type="image", required_credits=1)

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
                enable_qc=True,
            )
        else:
            translated_bytes, regions, qc_report = await image_service.translate_image(
                image_bytes,
                target_language=req_target_lang,
                instruction=req_instruction,
                model=req_model,
                provider_id=req_provider,
                enable_qc=True,
            )
    except ImageSessionTooLarge as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except Exception as e:
        logger.exception("Image test translation failed")
        raise HTTPException(
            status_code=500, detail=f"Image test translation failed: {e}"
        ) from e

    tx_ref = f"img_test_{session.get('id') if session else 'tx_' + str(abs(hash(translated_bytes[:32])))}"
    await deduct_credit(
        user_id=user_id,
        amount=1,
        reference_id=tx_ref,
        description="Cappy AI dịch và tự chấm điểm test ảnh",
        request=request,
    )

    session_id = str(session.get("id")) if session else ""
    version = int(session.get("version", 1)) if session else 1
    qc_score = int(qc_report.get("qc_score", 100) or 0)
    qc_approved = bool(qc_report.get("approved", True))
    issues = qc_report.get("issues") or []
    has_severe = any(
        isinstance(i, dict) and str(i.get("severity", "")).lower() in ("high", "critical", "severe")
        for i in issues
    )
    is_passed = qc_approved and (qc_score >= req_pass_threshold) and not has_severe

    if is_passed:
        if req_format.lower() == "json":
            b64_result = base64.b64encode(translated_bytes).decode("utf-8")
            return JSONResponse(
                content={
                    "status": "passed",
                    "approved": True,
                    "data": {
                        "id": session_id,
                        "version": version,
                        "qc_score": qc_score,
                        "qc_report": qc_report,
                        "image_base64": f"data:image/jpeg;base64,{b64_result}",
                        "format": "jpeg",
                        "regions_count": len(regions),
                        "regions": regions,
                        "url": _version_url(session_id, version) if session_id else None,
                    },
                }
            )

        headers = {
            "Content-Disposition": 'inline; filename="tested_pass.jpg"',
            "X-Capyy-Status": "pass",
            "X-Capyy-QC-Approved": "true",
            "X-Capyy-QC-Score": str(qc_score),
        }
        if session:
            headers["X-Capyy-Image-Id"] = session_id
            headers["X-Capyy-Image-Version"] = str(version)
            headers["X-Capyy-Image-Expires-At"] = str(session.get("expires_at"))
        return Response(
            content=translated_bytes,
            media_type="image/jpeg",
            headers=headers,
        )

    # QC Failed: compile issues and trigger auto repair
    issue_descriptions = [
        str(i.get("description"))
        for i in issues
        if isinstance(i, dict) and i.get("description")
    ]
    feedback_text = str(qc_report.get("feedback") or "")
    if feedback_text and issue_descriptions:
        repair_instruction = f"Khắc phục lỗi QC: {feedback_text} | Chi tiết: {'; '.join(issue_descriptions)}"
    elif feedback_text:
        repair_instruction = f"Khắc phục lỗi QC: {feedback_text}"
    elif issue_descriptions:
        repair_instruction = f"Khắc phục lỗi QC: {'; '.join(issue_descriptions)}"
    else:
        repair_instruction = "Khắc phục lỗi sót chữ và căn chỉnh lại các vùng chữ chưa đạt chuẩn."

    auto_repair_info: dict[str, Any] = {"triggered": False}
    if req_auto_repair and session_id and store is not None:
        auto_repair_info = {
            "triggered": True,
            "instruction": repair_instruction,
        }

        async def _run_bg_repair(sid: str, instr: str) -> None:
            try:
                logger.info("Auto-repair triggered for failed test session %s: %s", sid, instr)
                await image_service.repair_image(
                    sid,
                    instruction=instr,
                    enable_qc=True,
                )
                logger.info("Auto-repair completed successfully for session %s", sid)
            except Exception as err:
                logger.warning("Auto-repair background task failed for %s: %s", sid, err)

        asyncio.create_task(_run_bg_repair(session_id, repair_instruction))

    return JSONResponse(
        status_code=200,
        headers={
            "X-Capyy-Status": "fail",
            "X-Capyy-Image-Id": session_id,
            "X-Capyy-QC-Approved": "false",
            "X-Capyy-QC-Score": str(qc_score),
        },
        content={
            "status": "failed",
            "approved": False,
            "id": session_id,
            "version": version,
            "qc_score": qc_score,
            "feedback": feedback_text,
            "issues": issues,
            "auto_repair": auto_repair_info,
            "message": "Ảnh chưa đạt yêu cầu QC. Hệ thống đang tự động sửa lỗi (auto-repair) trong nền.",
        },
    )


async def _execute_feedback(
    image_id: str,
    status: str,
    approved: bool | None,
    reason: str | None,
    instruction: str | None,
    region_ids: list[int] | None,
    response_format: str,
    enable_qc: bool,
    request: Request,
) -> Response:
    store = _image_store(request)
    try:
        await asyncio.to_thread(store.get, image_id)
    except Exception as e:
        raise _repair_error(e) from e

    st = str(status).strip().lower()
    is_fail = (
        st in ("failed", "fail", "error", "rejected", "fix", "repair", "false", "0")
        or (approved is False)
    )

    if not is_fail:
        return JSONResponse(
            content={
                "status": "recorded",
                "id": image_id,
                "approved": True,
                "message": "Phản hồi ảnh đạt yêu cầu đã được ghi nhận thành công.",
            }
        )

    # User reported failure -> trigger repair
    feedback_reason = (reason or instruction or "").strip()
    if not feedback_reason:
        feedback_reason = "Sửa lại các lỗi còn sót trong ảnh theo phản hồi người dùng: xóa sạch chữ sót và căn chỉnh chữ tự nhiên."

    try:
        (
            repaired_bytes,
            regions,
            qc_report,
            state,
        ) = await request.app.state.image_service.repair_image(
            image_id,
            instruction=feedback_reason,
            region_ids=region_ids,
            enable_qc=enable_qc,
        )
    except Exception as e:
        logger.warning("Feedback repair failed for %s: %s", image_id, e)
        raise _repair_error(e) from e

    version = int(state.get("version") or 0)
    headers = {
        "Content-Disposition": f'inline; filename="repaired_v{version}.jpg"',
        "X-Capyy-Image-Id": str(state.get("id") or image_id),
        "X-Capyy-Image-Version": str(version),
        "X-Capyy-QC-Approved": str(qc_report.get("approved", True)),
        "X-Capyy-QC-Score": str(qc_report.get("qc_score", 100)),
        "X-Capyy-Status": "repaired",
    }
    if response_format.lower() == "image":
        return Response(content=repaired_bytes, media_type="image/jpeg", headers=headers)

    b64_result = base64.b64encode(repaired_bytes).decode("utf-8")
    return JSONResponse(
        content={
            "status": "repaired",
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
            },
        }
    )


@router.post("/feedback")
async def image_feedback_endpoint(
    request: Request,
    _auth: None = Depends(check_local_auth),
    image_id: str | None = Form(None),
    id: str | None = Form(None),
    status: str = Form("failed"),
    approved: bool | None = Form(None),
    reason: str | None = Form(None),
    instruction: str | None = Form(None),
    response_format: str = Form("image"),
    enable_qc: bool = Form(True),
) -> Response:
    """Submit user feedback for an image session (with image_id in body/form)."""
    target_id = image_id or id
    req_status = status
    req_approved = approved
    req_reason = reason
    req_instruction = instruction
    req_region_ids: list[int] | None = None
    req_format = response_format
    req_enable_qc = enable_qc

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            json_req = ImageFeedbackJsonRequest(**body)
            target_id = json_req.image_id or body.get("id") or target_id
            req_status = json_req.status
            req_approved = json_req.approved
            req_reason = json_req.reason
            req_instruction = json_req.instruction
            req_region_ids = json_req.region_ids
            req_format = json_req.response_format
            req_enable_qc = json_req.enable_qc
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to parse JSON feedback request: {e}"
            ) from e

    if not target_id:
        raise HTTPException(
            status_code=400,
            detail="Missing required 'image_id' in feedback request.",
        )

    return await _execute_feedback(
        image_id=target_id,
        status=req_status,
        approved=req_approved,
        reason=req_reason,
        instruction=req_instruction,
        region_ids=req_region_ids,
        response_format=req_format,
        enable_qc=req_enable_qc,
        request=request,
    )


@router.post("/{image_id}/feedback")
async def image_feedback_with_id_endpoint(
    image_id: str,
    request: Request,
    _auth: None = Depends(check_local_auth),
    status: str = Form("failed"),
    approved: bool | None = Form(None),
    reason: str | None = Form(None),
    instruction: str | None = Form(None),
    response_format: str = Form("image"),
    enable_qc: bool = Form(True),
) -> Response:
    """Submit user feedback for a specific image session (image_id in URL path)."""
    req_status = status
    req_approved = approved
    req_reason = reason
    req_instruction = instruction
    req_region_ids: list[int] | None = None
    req_format = response_format
    req_enable_qc = enable_qc

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            json_req = ImageFeedbackJsonRequest(**body)
            req_status = json_req.status
            req_approved = json_req.approved
            req_reason = json_req.reason
            req_instruction = json_req.instruction
            req_region_ids = json_req.region_ids
            req_format = json_req.response_format
            req_enable_qc = json_req.enable_qc
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to parse JSON feedback request: {e}"
            ) from e

    return await _execute_feedback(
        image_id=image_id,
        status=req_status,
        approved=req_approved,
        reason=req_reason,
        instruction=req_instruction,
        region_ids=req_region_ids,
        response_format=req_format,
        enable_qc=req_enable_qc,
        request=request,
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
