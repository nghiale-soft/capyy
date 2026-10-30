from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from gateway.routes.auth import router as auth_router
from gateway.services.billing_service import BillingService


def _mock_request(headers: dict[str, str] | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/images/translate",
        "headers": [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in (headers or {}).items()
        ],
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
async def test_jwt_create_verify_and_extract_user_id():
    settings = SimpleNamespace(
        billing_enabled=True,
        local_api_key="secret-gateway-key",
        jwt_secret="my-test-jwt-secret",
        jwt_expire_hours=24,
    )
    service = BillingService(settings)

    # Create JWT accessToken for Gmail user
    token = service.create_access_token(
        user_id="admin@gmail.com",
        email="admin@gmail.com",
        name="Admin User",
        picture="https://example.com/avatar.png",
    )
    claims = service.verify_access_token(token)
    assert claims is not None
    assert claims["sub"] == "admin@gmail.com"
    assert claims["email"] == "admin@gmail.com"

    # Extract userId from Bearer JWT accessToken
    req_jwt = _mock_request({"Authorization": f"Bearer {token}"})
    assert service.extract_user_id(req_jwt) == "admin@gmail.com"

    # Tampered token should fail verification
    assert service.verify_access_token(token + "tamper") is None

    # Fallback headers
    req1 = _mock_request({"x-user-id": "user_123"})
    assert service.extract_user_id(req1) == "user_123"

    req2 = _mock_request({"X-Client-Id": "client_abc"})
    assert service.extract_user_id(req2) == "client_abc"

    req4 = _mock_request({"Authorization": "Bearer secret-gateway-key"})
    assert service.extract_user_id(req4) == "anonymous_local"


@pytest.mark.asyncio
async def test_normal_user_sufficient_balance():
    settings = SimpleNamespace(
        billing_enabled=True,
        pricing_gateway_url="http://pricing-gateway:8088",
        pricing_gateway_timeout=5.0,
    )
    service = BillingService(settings)

    req = _mock_request({"X-User-Id": "shopper@gmail.com"})

    with patch.object(service, "get_balance", new=AsyncMock(return_value=50)), \
         patch.object(
             httpx.AsyncClient,
             "post",
             new=AsyncMock(
                 return_value=httpx.Response(
                     200,
                     json={"success": True},
                     request=httpx.Request("POST", "http://pricing-gateway:8088/api/v1/credits/deduct"),
                 )
             ),
         ):
        user_id = await service.check_and_reserve(req, resource_type="image", required_credits=1)
        assert user_id == "shopper@gmail.com"

        res = await service.deduct_credit(user_id, 1, "img_ref", "image translation")
        assert res is not None and res["success"] is True


@pytest.mark.asyncio
async def test_new_user_auto_grants_trial():
    settings = SimpleNamespace(
        billing_enabled=True,
        pricing_gateway_url="http://pricing-gateway:8088",
        pricing_gateway_timeout=5.0,
        free_trial_credits=20,
    )
    service = BillingService(settings)

    req = _mock_request({"X-User-Id": "newuser@gmail.com"})

    with patch.object(service, "get_balance", new=AsyncMock(return_value=0)), \
         patch.object(service, "grant_trial", new=AsyncMock(return_value=20)):
        user_id = await service.check_and_reserve(req, resource_type="image", required_credits=1)
        assert user_id == "newuser@gmail.com"
        service.grant_trial.assert_awaited_once_with("newuser@gmail.com")


@pytest.mark.asyncio
async def test_user_insufficient_credits_raises_402():
    settings = SimpleNamespace(
        billing_enabled=True,
        pricing_gateway_url="http://pricing-gateway:8088",
        pricing_gateway_timeout=5.0,
        free_trial_credits=20,
    )
    service = BillingService(settings)

    req = _mock_request({"X-User-Id": "broke@gmail.com"})

    with patch.object(service, "get_balance", new=AsyncMock(return_value=0)), \
         patch.object(service, "grant_trial", new=AsyncMock(return_value=0)):
        with pytest.raises(HTTPException) as exc_info:
            await service.check_and_reserve(req, resource_type="image", required_credits=1)

        assert exc_info.value.status_code == 402
        detail = exc_info.value.detail
        assert detail["code"] == "PAYMENT_REQUIRED"
        assert detail["current_balance"] == 0
        assert detail["required_credits"] == 1
        assert "checkout_url" in detail


def test_google_login_and_me_routes():
    app = FastAPI()
    app.include_router(auth_router)
    settings = SimpleNamespace(
        billing_enabled=True,
        pricing_gateway_url="http://pricing-gateway:8088",
        pricing_gateway_timeout=5.0,
        free_trial_credits=20,
        jwt_secret="route-test-secret",
        jwt_expire_hours=24,
    )
    service = BillingService(settings)
    app.state.settings = settings
    app.state.billing_service = service
    client = TestClient(app)

    fake_profile = {
        "email": "testlogin@gmail.com",
        "name": "Test Login",
        "picture": "https://lh3.googleusercontent.com/a/default",
        "sub": "1092837465",
    }

    with patch.object(service, "verify_google_token", new=AsyncMock(return_value=fake_profile)), \
         patch.object(service, "get_balance", new=AsyncMock(return_value=0)), \
         patch.object(service, "grant_trial", new=AsyncMock(return_value=20)):
        resp = client.post("/v1/auth/google", json={"token": "fake.google.token"})
        assert resp.status_code == 200
        body = resp.json()
        assert "accessToken" in body
        assert body["tokenType"] == "Bearer"
        assert body["user"]["email"] == "testlogin@gmail.com"
        assert body["user"]["balance"] == 20
        access_token = body["accessToken"]

    with patch.object(service, "get_balance", new=AsyncMock(return_value=19)):
        me_resp = client.get(
            "/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert me_resp.status_code == 200
        me_data = me_resp.json()["user"]
        assert me_data["userId"] == "testlogin@gmail.com"
        assert me_data["balance"] == 19


@pytest.mark.asyncio
async def test_app_id_and_secret_key_flow():
    settings = SimpleNamespace(
        billing_enabled=True,
        pricing_gateway_url="http://pricing-gateway:8088",
        pricing_gateway_timeout=5.0,
        app_credentials=(
            ("cross-border", "cb_secret_chinhthien_2026"),
            ("dopa-english", "dopa_secret_chinhthien_2026"),
        ),
    )
    service = BillingService(settings)

    # 1. Valid appId + secretKey -> wallet 'app:cross-border' and deducts coin normally
    req_valid = _mock_request({
        "X-App-Id": "cross-border",
        "X-Secret-Key": "cb_secret_chinhthien_2026",
    })
    with patch.object(service, "get_balance", new=AsyncMock(return_value=100)), \
         patch.object(
             httpx.AsyncClient,
             "post",
             new=AsyncMock(
                 return_value=httpx.Response(
                     200,
                     json={"success": True},
                     request=httpx.Request("POST", "http://pricing-gateway:8088/api/v1/credits/deduct"),
                 )
             ),
         ):
        wallet_id = await service.check_and_reserve(req_valid, resource_type="image", required_credits=1)
        assert wallet_id == "app:cross-border"
        res = await service.deduct_credit(wallet_id, 1, "cb_ref_1", "Cross border image translate")
        assert res is not None and res["success"] is True

    # 2. Valid appId + secretKey but 0 coins -> blocked with 402 (anti-spam protection)
    with patch.object(service, "get_balance", new=AsyncMock(return_value=0)), \
         patch.object(service, "grant_trial", new=AsyncMock(return_value=0)):
        with pytest.raises(HTTPException) as exc_402:
            await service.check_and_reserve(req_valid, resource_type="image", required_credits=1)
        assert exc_402.value.status_code == 402

    # 3. Wrong secretKey -> blocked with 401 Unauthorized
    req_bad_secret = _mock_request({
        "X-App-Id": "cross-border",
        "X-Secret-Key": "wrong_secret_hacker",
    })
    with pytest.raises(HTTPException) as exc_401:
        await service.check_and_reserve(req_bad_secret, resource_type="image", required_credits=1)
    assert exc_401.value.status_code == 401


def test_app_login_route():
    app = FastAPI()
    app.include_router(auth_router)
    settings = SimpleNamespace(
        billing_enabled=True,
        pricing_gateway_url="http://pricing-gateway:8088",
        pricing_gateway_timeout=5.0,
        free_trial_credits=20,
        jwt_secret="route-test-secret",
        jwt_expire_hours=24,
        app_credentials=(
            ("cross-border", "cb_secret_chinhthien_2026"),
            ("dopa-english", "dopa_secret_chinhthien_2026"),
        ),
    )
    service = BillingService(settings)
    app.state.settings = settings
    app.state.billing_service = service
    client = TestClient(app)

    with patch.object(service, "get_balance", new=AsyncMock(return_value=500)):
        resp = client.post(
            "/v1/auth/app",
            json={"appId": "dopa-english", "secretKey": "dopa_secret_chinhthien_2026"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["user"]["userId"] == "app:dopa-english"
        assert body["user"]["balance"] == 500
        assert "accessToken" in body

    # Invalid secretKey returns 401
    bad_resp = client.post(
        "/v1/auth/app",
        json={"appId": "dopa-english", "secretKey": "bad_key"},
    )
    assert bad_resp.status_code == 401


def test_topup_route():
    app = FastAPI()
    app.include_router(auth_router)
    settings = SimpleNamespace(
        billing_enabled=True,
        jwt_secret="route-test-secret",
    )
    service = BillingService(settings)
    app.state.settings = settings
    app.state.billing_service = service
    client = TestClient(app)

    with patch.object(service, "topup", new=AsyncMock(return_value=1500)):
        resp = client.post(
            "/v1/auth/topup",
            json={"userId": "app:cross-border", "amount": 500, "referenceId": "order_123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["newBalance"] == 1500
        assert body["amountAdded"] == 500
