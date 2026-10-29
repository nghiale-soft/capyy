from __future__ import annotations

import logging
import math
import uuid
from typing import Any

import httpx
from fastapi import HTTPException, Request

from ..core.config import Settings

logger = logging.getLogger("gateway.services.billing")


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

    def extract_user_id(self, request: Request) -> str:
        headers = request.headers
        user_id = (
            headers.get("x-user-id")
            or headers.get("X-User-Id")
            or headers.get("x-client-id")
            or headers.get("X-Client-Id")
        )
        if user_id and user_id.strip():
            return user_id.strip()

        auth = headers.get("authorization") or headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1].strip()
            local_key = getattr(self.settings, "local_api_key", None)
            if token and token != local_key:
                return token

        return "anonymous_local"

    async def has_active_subscription(self, user_id: str) -> bool:
        """Check if user has an active Pro subscription in Pricing Gateway."""
        if not self.billing_enabled:
            return True
        if user_id == "anonymous_local":
            return False

        url = f"{self.pricing_gateway_url}/api/v1/subscriptions/{user_id}/is-active"
        try:
            async with httpx.AsyncClient(timeout=self.pricing_gateway_timeout) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    return bool(data.get("data", {}).get("hasActiveSubscription", False))
        except Exception as e:
            logger.warning("Failed to check subscription status for %s: %s", user_id, e)
        return False

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
        """Grant free trial credits to new user."""
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

        # Check if user has active subscription (unlimited Pro)
        if await self.has_active_subscription(user_id):
            logger.info("User %s has active subscription; skipping credit deduction", user_id)
            return {"subscription": True, "deducted": 0}

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

        # Pro subscribers have unlimited access
        if await self.has_active_subscription(user_id):
            return user_id

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
