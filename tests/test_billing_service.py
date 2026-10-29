from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from gateway.core.config import Settings
from gateway.services.billing_service import BillingService


def _mock_request(headers: dict[str, str] | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/images/translate",
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()],
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_billing_disabled_by_default():
    settings = SimpleNamespace(billing_enabled=False)
    service = BillingService(settings)

    req = _mock_request()
    user_id = await service.check_and_reserve(req, resource_type="image", required_credits=1)
    assert user_id == "anonymous_local"

    deduct_result = await service.deduct_credit(user_id, 1, "ref1", "test")
    assert deduct_result is None


@pytest.mark.asyncio
async def test_user_id_extraction_headers():
    settings = SimpleNamespace(billing_enabled=True, local_api_key="secret-gateway-key")
    service = BillingService(settings)

    # 1. X-User-Id
    req1 = _mock_request({"x-user-id": "user_123"})
    assert service.extract_user_id(req1) == "user_123"

    # 2. X-Client-Id
    req2 = _mock_request({"X-Client-Id": "client_abc"})
    assert service.extract_user_id(req2) == "client_abc"

    # 3. Bearer token as user ID (when not local_api_key)
    req3 = _mock_request({"Authorization": "Bearer usr_token_xyz"})
    assert service.extract_user_id(req3) == "usr_token_xyz"

    # 4. Bearer token matching local_api_key should fallback to anonymous
    req4 = _mock_request({"Authorization": "Bearer secret-gateway-key"})
    assert service.extract_user_id(req4) == "anonymous_local"


@pytest.mark.asyncio
async def test_pro_subscription_bypasses_credits():
    settings = SimpleNamespace(
        billing_enabled=True,
        pricing_gateway_url="http://pricing-gateway:8088",
        pricing_gateway_timeout=5.0,
    )
    service = BillingService(settings)

    req = _mock_request({"X-User-Id": "vip_user"})

    with patch.object(service, "has_active_subscription", new=AsyncMock(return_value=True)):
        user_id = await service.check_and_reserve(req, resource_type="video", required_credits=15)
        assert user_id == "vip_user"

        res = await service.deduct_credit(user_id, 15, "vid_ref", "video translation")
        assert res == {"subscription": True, "deducted": 0}


@pytest.mark.asyncio
async def test_normal_user_sufficient_balance():
    settings = SimpleNamespace(
        billing_enabled=True,
        pricing_gateway_url="http://pricing-gateway:8088",
        pricing_gateway_timeout=5.0,
    )
    service = BillingService(settings)

    req = _mock_request({"X-User-Id": "user_normal"})

    with patch.object(service, "has_active_subscription", new=AsyncMock(return_value=False)), \
         patch.object(service, "get_balance", new=AsyncMock(return_value=50)), \
         patch.object(httpx.AsyncClient, "post", new=AsyncMock(return_value=httpx.Response(200, json={"success": True}))):
        
        user_id = await service.check_and_reserve(req, resource_type="image", required_credits=1)
        assert user_id == "user_normal"


@pytest.mark.asyncio
async def test_new_user_auto_grants_trial():
    settings = SimpleNamespace(
        billing_enabled=True,
        pricing_gateway_url="http://pricing-gateway:8088",
        pricing_gateway_timeout=5.0,
        free_trial_credits=20,
    )
    service = BillingService(settings)

    req = _mock_request({"X-User-Id": "brand_new_user"})

    with patch.object(service, "has_active_subscription", new=AsyncMock(return_value=False)), \
         patch.object(service, "get_balance", new=AsyncMock(return_value=0)), \
         patch.object(service, "grant_trial", new=AsyncMock(return_value=20)):
        
        user_id = await service.check_and_reserve(req, resource_type="image", required_credits=1)
        assert user_id == "brand_new_user"
        service.grant_trial.assert_awaited_once_with("brand_new_user")


@pytest.mark.asyncio
async def test_user_insufficient_credits_raises_402():
    settings = SimpleNamespace(
        billing_enabled=True,
        pricing_gateway_url="http://pricing-gateway:8088",
        pricing_gateway_timeout=5.0,
        free_trial_credits=20,
    )
    service = BillingService(settings)

    req = _mock_request({"X-User-Id": "broke_user"})

    # User has 0 balance, and grant_trial returns 0 (since trial already used previously)
    with patch.object(service, "has_active_subscription", new=AsyncMock(return_value=False)), \
         patch.object(service, "get_balance", new=AsyncMock(return_value=0)), \
         patch.object(service, "grant_trial", new=AsyncMock(return_value=0)):
        
        with pytest.raises(HTTPException) as exc_info:
            await service.check_and_reserve(req, resource_type="image", required_credits=1)

        assert exc_info.value.status_code == 402
        detail = exc_info.value.detail
        assert detail["code"] == "PAYMENT_REQUIRED"
        assert detail["current_balance"] == 0
        assert detail["required_credits"] == 1
        assert "checkout_url" in detail
