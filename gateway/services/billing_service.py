from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx
from fastapi import HTTPException, Request

logger = logging.getLogger("gateway.services.billing")


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


class BillingService:
    def __init__(self, settings: Any) -> None:
        self.settings = settings

    @property
    def billing_enabled(self) -> bool:
        return bool(getattr(self.settings, "billing_enabled", False))

    @property
    def pricing_gateway_url(self) -> str:
        url = getattr(self.settings, "pricing_gateway_url", "http://pricing-gateway:8088")
        return (url or "http://pricing-gateway:8088").rstrip("/")

    @property
    def pricing_gateway_timeout(self) -> float:
        return float(getattr(self.settings, "pricing_gateway_timeout", 5.0))

    @property
    def free_trial_credits(self) -> int:
        return int(getattr(self.settings, "free_trial_credits", 20))

    @property
    def jwt_secret(self) -> str:
        return str(
            getattr(
                self.settings,
                "jwt_secret",
                "cappy-jwt-secret-key-chinhthien-2026",
            )
        )

    @property
    def jwt_expire_hours(self) -> int:
        return int(getattr(self.settings, "jwt_expire_hours", 720))

    def create_access_token(
        self,
        user_id: str,
        email: str,
        name: str = "",
        picture: str = "",
    ) -> str:
        """Create an HS256 JWT accessToken."""
        now = int(time.time())
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": user_id,
            "email": email,
            "name": name,
            "picture": picture,
            "iat": now,
            "exp": now + self.jwt_expire_hours * 3600,
        }
        segments = [
            _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        ]
        signing_input = ".".join(segments).encode("ascii")
        signature = hmac.new(
            self.jwt_secret.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        segments.append(_b64url_encode(signature))
        return ".".join(segments)

    def verify_access_token(self, token: str) -> dict[str, Any] | None:
        """Verify an HS256 JWT accessToken and return payload if valid."""
        parts = token.split(".")
        if len(parts) != 3:
            return None
        try:
            signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
            expected_sig = hmac.new(
                self.jwt_secret.encode("utf-8"), signing_input, hashlib.sha256
            ).digest()
            actual_sig = _b64url_decode(parts[2])
            if not hmac.compare_digest(expected_sig, actual_sig):
                return None
            payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
            exp = payload.get("exp")
            if exp is not None and int(time.time()) > int(exp):
                return None
            return payload
        except Exception:
            return None

    async def verify_google_token(self, google_token: str) -> dict[str, str]:
        """Verify Google id_token or OAuth2 access_token and return user profile."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 1. If token looks like a JWT (3 segments), verify via tokeninfo as id_token
            if google_token.count(".") == 2:
                resp = await client.get(
                    "https://oauth2.googleapis.com/tokeninfo",
                    params={"id_token": google_token},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    email = data.get("email")
                    if email:
                        return {
                            "email": email.strip().lower(),
                            "name": data.get("name", email.split("@")[0]),
                            "picture": data.get("picture", ""),
                            "sub": data.get("sub", email),
                        }

            # 2. Otherwise verify as OAuth2 access_token via Google userinfo endpoint
            resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {google_token}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                email = data.get("email")
                if email:
                    return {
                        "email": email.strip().lower(),
                        "name": data.get("name", email.split("@")[0]),
                        "picture": data.get("picture", ""),
                        "sub": data.get("sub", email),
                    }

        raise HTTPException(
            status_code=401,
            detail="Google token không hợp lệ hoặc đã hết hạn.",
        )

    def extract_user_id(self, request: Request) -> str:
        headers = request.headers

        # 1. Check Authorization: Bearer <accessToken>
        auth = headers.get("authorization") or headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1].strip()
            local_key = getattr(self.settings, "local_api_key", None)
            if token and token != local_key:
                jwt_payload = self.verify_access_token(token)
                if jwt_payload and jwt_payload.get("sub"):
                    return str(jwt_payload["sub"])
                # Fallback: if X-User-Id is not set, use raw bearer token
                x_uid = (
                    headers.get("x-user-id")
                    or headers.get("X-User-Id")
                    or headers.get("x-client-id")
                    or headers.get("X-Client-Id")
                )
                if not x_uid:
                    return token

        # 2. Check X-User-Id / X-Client-Id headers
        user_id = (
            headers.get("x-user-id")
            or headers.get("X-User-Id")
            or headers.get("x-client-id")
            or headers.get("X-Client-Id")
        )
        if user_id and user_id.strip():
            return user_id.strip()

        return "anonymous_local"

    async def get_balance(self, user_id: str) -> int:
        """Fetch current credit balance for user from Pricing Gateway."""
        url = f"{self.pricing_gateway_url}/api/v1/credits/{user_id}/balance"
        async with httpx.AsyncClient(timeout=self.pricing_gateway_timeout) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return 0
            resp.raise_for_status()
            data = resp.json()
            return int(data.get("data", {}).get("balance", 0))

    async def grant_trial(self, user_id: str, amount: int | None = None) -> int:
        """Grant free trial credits to new user (one-time only enforced by Pricing Gateway)."""
        amt = amount if amount is not None else self.free_trial_credits
        url = f"{self.pricing_gateway_url}/api/v1/credits/{user_id}/grant-trial"
        async with httpx.AsyncClient(timeout=self.pricing_gateway_timeout) as client:
            resp = await client.post(url, json={"amount": amt})
            resp.raise_for_status()
            data = resp.json()
            return int(data.get("data", {}).get("balance", amt))

    async def deduct_credit(
        self,
        user_id: str,
        amount: int,
        reference_id: str,
        description: str,
    ) -> dict[str, Any] | None:
        """Deduct credits from user wallet after successful operation."""
        if not self.billing_enabled:
            return None
        if user_id == "anonymous_local":
            return None

        url = f"{self.pricing_gateway_url}/api/v1/credits/deduct"
        payload = {
            "userId": user_id,
            "amount": amount,
            "referenceId": reference_id,
            "description": description,
        }
        try:
            async with httpx.AsyncClient(timeout=self.pricing_gateway_timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error("Failed to deduct %d credits for %s: %s", amount, user_id, e)
            return None

    async def check_and_reserve(
        self,
        request: Request,
        resource_type: str = "image",
        required_credits: int | None = None,
    ) -> str:
        """Pre-check credit balance before starting heavy AI processing."""
        if not self.billing_enabled:
            return "anonymous_local"

        user_id = self.extract_user_id(request)
        required = required_credits or (1 if resource_type == "image" else 5)

        try:
            current_balance = await self.get_balance(user_id)
            # If user has 0 credits, attempt granting trial (if new user)
            if current_balance == 0:
                current_balance = await self.grant_trial(user_id)
        except Exception as e:
            logger.error("Pricing Gateway error while checking credit for %s: %s", user_id, e)
            raise HTTPException(
                status_code=503,
                detail="Billing service temporarily unavailable",
            ) from e

        if current_balance < required:
            raise HTTPException(
                status_code=402,
                detail={
                    "code": "PAYMENT_REQUIRED",
                    "message": (
                        f"Số dư của bạn ({current_balance} credits) không đủ thực hiện yêu cầu "
                        f"({required} credits). Vui lòng nạp thêm."
                    ),
                    "current_balance": current_balance,
                    "required_credits": required,
                    "checkout_url": f"/billing/checkout?userId={user_id}",
                },
            )

        return user_id
