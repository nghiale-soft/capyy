from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..deps import get_billing_service

router = APIRouter(prefix="/v1/auth", tags=["auth"])
logger = logging.getLogger("gateway.routes.auth")


class GoogleLoginRequest(BaseModel):
    token: str | None = Field(
        default=None,
        description="Google id_token or OAuth2 access_token",
    )
    id_token: str | None = Field(
        default=None,
        description="Alias for Google id_token",
    )
    access_token: str | None = Field(
        default=None,
        description="Alias for Google OAuth2 access_token",
    )


@router.post("/google", summary="Login with Google (Gmail) and receive JWT accessToken + coin balance")
async def google_login_endpoint(
    payload: GoogleLoginRequest,
    request: Request,
) -> dict[str, Any]:
    raw_token = (payload.token or payload.id_token or payload.access_token or "").strip()
    if not raw_token:
        raise HTTPException(
            status_code=400,
            detail="Missing Google token ('token', 'id_token', or 'access_token')",
        )

    billing = get_billing_service(request)
    profile = await billing.verify_google_token(raw_token)
    email = profile["email"]
    name = profile.get("name", "")
    picture = profile.get("picture", "")

    # Use Gmail address as userId so it is human-readable in wallets DB
    user_id = email

    # Fetch or initialize wallet with one-time 20 trial coins
    balance = 0
    try:
        balance = await billing.get_balance(user_id)
        if balance == 0:
            balance = await billing.grant_trial(user_id)
    except Exception as e:
        logger.warning("Could not fetch/grant balance from Pricing Gateway during login for %s: %s", user_id, e)

    access_token = billing.create_access_token(
        user_id=user_id,
        email=email,
        name=name,
        picture=picture,
    )

    return {
        "accessToken": access_token,
        "tokenType": "Bearer",
        "user": {
            "userId": user_id,
            "email": email,
            "name": name,
            "picture": picture,
            "balance": balance,
        },
    }


@router.get("/me", summary="Get current logged-in user profile and live coin balance")
async def get_me_endpoint(request: Request) -> dict[str, Any]:
    billing = get_billing_service(request)
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer accessToken")

    token = auth.split(" ", 1)[1].strip()
    claims = billing.verify_access_token(token)
    if not claims or not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid or expired accessToken")

    user_id = str(claims["sub"])
    balance = 0
    try:
        balance = await billing.get_balance(user_id)
    except Exception as e:
        logger.warning("Could not fetch balance for %s: %s", user_id, e)

    return {
        "user": {
            "userId": user_id,
            "email": claims.get("email", user_id),
            "name": claims.get("name", ""),
            "picture": claims.get("picture", ""),
            "balance": balance,
        }
    }
