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
        self._local_wallets: dict[str, int] = {}

    def _get_db_conn(self):
        host = getattr(self.settings, "database_host", None)
        if not host:
            return None
        import pymysql

        return pymysql.connect(
            host=host,
            port=int(getattr(self.settings, "database_port", 3306)),
            database=getattr(self.settings, "database_name", "capyy"),
            user=getattr(self.settings, "database_user", "root"),
            password=getattr(self.settings, "database_password", ""),
            charset="utf8mb4",
            connect_timeout=2,
            read_timeout=3,
            autocommit=True,
        )

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

    @property
    def app_credentials_map(self) -> dict[str, str]:
        raw = getattr(
            self.settings,
            "app_credentials",
            (
                ("cross-border", "cb_secret_chinhthien_2026"),
                ("dopa-english", "dopa_secret_chinhthien_2026"),
            ),
        )
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        return {str(k): str(v) for k, v in raw}

    def get_app_secret(self, app_id: str) -> str | None:
        """Query secretKey for appId from MariaDB capyy.app_credentials, falling back to static config."""
        clean_id = (app_id or "").strip()
        if not clean_id:
            return None

        # 1. Try DB lookup first
        try:
            conn = self._get_db_conn()
            if conn:
                with conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "SELECT secret_key FROM app_credentials WHERE app_id = %s AND is_active = 1",
                            (clean_id,),
                        )
                        row = cursor.fetchone()
                        if row and row[0]:
                            return str(row[0])
        except Exception as e:
            logger.debug("DB lookup for app_credentials failed: %s", e)

        # 2. Fallback to static app_credentials map in config
        return self.app_credentials_map.get(clean_id)

    def verify_app_credentials(self, app_id: str, secret_key: str) -> str:
        """Verify appId + secretKey and return the wallet userId ('app:<appId>')."""
        clean_id = (app_id or "").strip()
        clean_secret = (secret_key or "").strip()
        expected = self.get_app_secret(clean_id)
        if not clean_id or not clean_secret or not expected:
            raise HTTPException(status_code=401, detail="Invalid appId or secretKey")
        if not hmac.compare_digest(expected.encode("utf-8"), clean_secret.encode("utf-8")):
            raise HTTPException(status_code=401, detail="Invalid appId or secretKey")
        return f"app:{clean_id}"

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

        # 1. Check App credentials (X-App-Id + X-Secret-Key / X-App-Secret)
        app_id = headers.get("x-app-id") or headers.get("X-App-Id")
        secret_key = (
            headers.get("x-secret-key")
            or headers.get("X-Secret-Key")
            or headers.get("x-app-secret")
            or headers.get("X-App-Secret")
        )
        if app_id or secret_key:
            return self.verify_app_credentials(app_id or "", secret_key or "")

        # 2. Check Authorization: Bearer <accessToken>
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

        # 3. Check X-User-Id / X-Client-Id headers
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
        """Fetch current credit balance for user directly from capyy.wallets."""
        try:
            conn = self._get_db_conn()
            if conn:
                with conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "SELECT balance FROM wallets WHERE user_id = %s",
                            (user_id,),
                        )
                        row = cursor.fetchone()
                        if row is not None:
                            return int(row[0])
                        # If wallet does not exist yet: initialize with trial credits (or 10000 for apps/local)
                        init_bal = (
                            10000
                            if (user_id.startswith("app:") or user_id in ("local-key", "anonymous_local"))
                            else self.free_trial_credits
                        )
                        cursor.execute(
                            "INSERT INTO wallets (user_id, balance) VALUES (%s, %s) "
                            "ON DUPLICATE KEY UPDATE balance = balance",
                            (user_id, init_bal),
                        )
                        cursor.execute(
                            "INSERT INTO coin_transactions (user_id, amount, balance_after, reference_id, description) "
                            "VALUES (%s, %s, %s, %s, %s)",
                            (user_id, init_bal, init_bal, "init", "Khởi tạo ví mới"),
                        )
                        return init_bal
        except Exception as e:
            logger.debug("Local DB get_balance failed, using fallback: %s", e)

        return self._local_wallets.get(user_id, self.free_trial_credits)

    async def topup(
        self,
        user_id: str,
        amount: int,
        reference_id: str = "",
        description: str = "",
    ) -> int:
        """Add credits to a wallet in capyy.wallets (called by Pricing Gateway webhook or admin)."""
        try:
            conn = self._get_db_conn()
            if conn:
                with conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "INSERT INTO wallets (user_id, balance) VALUES (%s, %s) "
                            "ON DUPLICATE KEY UPDATE balance = balance + %s",
                            (user_id, amount, amount),
                        )
                        cursor.execute("SELECT balance FROM wallets WHERE user_id = %s", (user_id,))
                        new_bal = cursor.fetchone()[0]
                        cursor.execute(
                            "INSERT INTO coin_transactions (user_id, amount, balance_after, reference_id, description) "
                            "VALUES (%s, %s, %s, %s, %s)",
                            (user_id, amount, new_bal, reference_id, description),
                        )
                        return int(new_bal)
        except Exception as e:
            logger.debug("Local DB topup failed, using fallback: %s", e)

        cur = self._local_wallets.get(user_id, 0)
        self._local_wallets[user_id] = cur + amount
        return self._local_wallets[user_id]

    async def grant_trial(self, user_id: str, amount: int | None = None) -> int:
        """Grant free trial credits to new user."""
        amt = amount if amount is not None else self.free_trial_credits
        return await self.topup(user_id, amt, reference_id="grant_trial", description="Cộng credit dùng thử miễn phí")

    async def deduct_credit(
        self,
        user_id: str,
        amount: int,
        reference_id: str,
        description: str,
    ) -> dict[str, Any] | None:
        """Deduct credits from user wallet directly in capyy.wallets."""
        if not self.billing_enabled:
            return None
        if user_id == "anonymous_local":
            return None

        try:
            conn = self._get_db_conn()
            if conn:
                with conn:
                    with conn.cursor() as cursor:
                        affected = cursor.execute(
                            "UPDATE wallets SET balance = balance - %s WHERE user_id = %s AND balance >= %s",
                            (amount, user_id, amount),
                        )
                        if affected > 0:
                            cursor.execute("SELECT balance FROM wallets WHERE user_id = %s", (user_id,))
                            new_bal = cursor.fetchone()[0]
                            cursor.execute(
                                "INSERT INTO coin_transactions (user_id, amount, balance_after, reference_id, description) "
                                "VALUES (%s, %s, %s, %s, %s)",
                                (user_id, -amount, new_bal, reference_id, description),
                            )
                            return {"success": True, "balance": new_bal, "deducted": amount}
                        else:
                            logger.warning("Insufficient balance to deduct %d credits for %s", amount, user_id)
                            return None
        except Exception as e:
            logger.debug("Local DB deduct_credit failed, using fallback: %s", e)

        cur = self._local_wallets.get(user_id, self.free_trial_credits)
        if cur >= amount:
            self._local_wallets[user_id] = cur - amount
            return {"success": True, "balance": cur - amount, "deducted": amount}
        return None

    async def check_and_reserve(
        self,
        request: Request,
        resource_type: str = "image",
        required_credits: int | None = None,
    ) -> str:
        """Pre-check credit balance before starting heavy AI processing."""
        user_id = self.extract_user_id(request)
        if not self.billing_enabled:
            return "anonymous_local"

        required = required_credits or (5 if resource_type == "video" else 1)

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
