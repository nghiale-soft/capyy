# ĐẶC TẢ KỸ THUẬT & HƯỚNG DẪN TRIỂN KHAI THƯƠNG MẠI HÓA CAPPY AI
## CAPPY AI LOCALIZATION SUITE: VISION & VIDEO INPAINTING ENGINE + PRICING GATEWAY INTEGRATION

> **Mục tiêu tài liệu**: Hướng dẫn chi tiết cho Kỹ sư / AI Agent triển khai cơ chế kiểm soát số dư tín dụng (Credit Metering), tích hợp Cổng thanh toán [`Pricing Gateway`](file:///Users/nghiale98/Documents/1.My%20Companies/ChinhThien/Source%20Code/Pricing%20Gateway), và tách rời Cappy thành nền tảng độc lập phục vụ cả **Web App** và **Chrome Extension**.

---

## 1. TỔNG QUAN KIẾN TRÚC TÁCH RỜI (DECOUPLED ARCHITECTURE)

Cappy không còn là một plugin phụ trợ của riêng 1688, mà là **Cốt lõi Xử lý Thị giác & Video Độc lập (Standalone AI Localization Engine)**:

```mermaid
flowchart TD
    subgraph Clients (Đầu cuối người dùng)
        C1["Cappy Web App<br/>(Kéo thả ảnh hàng loạt, Dán link Video Douyin/1688)"]
        C2["Universal Chrome Extension<br/>(Duyệt web 1-click trên 1688, Taobao, Douyin)"]
        C3["Internal 1688-to-Shopee ERP<br/>(Client nội bộ cắm API)"]
    end

    subgraph Cappy Core Engine (Port 1221)
        GW["FastAPI Gateway (gateway/app.py)"]
        AUTH["Auth & Credit Interceptor (gateway/deps.py)"]
        IMG["Image Service (gateway/routes/images.py)"]
        VID["Video Service (gateway/routes/videos.py)"]
    end

    subgraph Central Billing (Port 8088)
        PG["Pricing Gateway (Spring Boot / MariaDB)"]
        WALLET["Ví Credits (wallets & transactions)"]
        PAYOS["PayOS VietQR Napas 24/7"]
    end

    C1 & C2 & C3 -->|Gửi Request kèm X-User-Id / Bearer Token| GW
    GW --> AUTH
    AUTH -->|1. Pre-check Balance| WALLET
    AUTH -->|2. Đủ điểm -> Chuyển tiếp| IMG & VID
    IMG & VID -->|3. Dịch xong thành công| WALLET
    WALLET -->|Trừ Credit: POST /api/v1/credits/deduct| PG
    WALLET -.->|Hết tiền -> Trả 402 + QR VietQR| C1 & C2
```

---

## 2. QUY CHUẨN TIÊU THỤ TÍN DỤNG (CREDIT CONSUMPTION RULES)

Hệ thống hoạt động theo mô hình nạp trước (Prepaid Credits). Quy tắc quy đổi:

| Định dạng tài nguyên | Đơn vị tính | Số Credit tiêu thụ | Ghi chú kỹ thuật |
| :--- | :--- | :---: | :--- |
| **Hình ảnh (Image)** | Mỗi ảnh dịch hoàn tất | **1 Credit** | Áp dụng cho mọi định dạng JPG/PNG/WEBP. |
| **Video (Video)** | Mỗi block **10 giây** | **5 Credits** | Làm tròn lên theo block 10 giây (`ceil(duration_seconds / 10) * 5`). |

### Ví dụ quy đổi Video:
* Video $\le 10$ giây: $1 \times 5 = \mathbf{5\text{ Credits}}$.
* Video $11 - 20$ giây: $2 \times 5 = \mathbf{10\text{ Credits}}$.
* Video $21 - 30$ giây (chuẩn review Douyin/TikTok): $3 \times 5 = \mathbf{15\text{ Credits}}$.
* Video $51 - 60$ giây: $6 \times 5 = \mathbf{30\text{ Credits}}$.

### Bảng giá bán gói nạp tiền (VietQR PayOS):
* 🎁 **Tân thủ (Free Trial)**: Tặng ngay **20 Credits** khi người dùng tạo tài khoản/cài extension lần đầu.
* 🥉 **Gói Khởi Điểm**: `50.000 VNĐ` $\rightarrow$ Nhận **150 Credits** (~333đ/credit).
* 🥈 **Gói Phổ Thông (Khuyến nghị)**: `100.000 VNĐ` $\rightarrow$ Nhận **400 Credits** (~250đ/credit).
* 🥇 **Gói Pro / Studio**: `200.000 VNĐ` $\rightarrow$ Nhận **1.000 Credits** (~200đ/credit).

---

## 3. THÔNG SỐ GIAO TIẾP VỚI PRICING GATEWAY

Pricing Gateway chạy tại mạng Docker (`common-service`) hoặc Localhost:
* **Host nội bộ Docker**: `http://pricing-gateway:8088`
* **Host máy tính (Local)**: `http://localhost:8088`

### 3.1. Kiểm tra số dư (Pre-check Balance)
* **Endpoint**: `GET /api/v1/credits/{userId}/balance`
* **Headers**: `Accept: application/json`
* **Phản hồi mẫu**:
```json
{
  "code": 200,
  "message": "Success",
  "data": {
    "userId": "user_a1b2c3d4",
    "balance": 150,
    "currency": "CREDIT"
  }
}
```

### 3.2. Trừ tín dụng sau khi xử lý thành công (Deduct Credit)
* **Endpoint**: `POST /api/v1/credits/deduct`
* **Headers**: `Content-Type: application/json`
* **Body Request**:
```json
{
  "userId": "user_a1b2c3d4",
  "amount": 1,
  "referenceId": "img_tx_20261029_091522_891",
  "description": "Cappy AI dịch ảnh 1688"
}
```
* Đối với Video:
```json
{
  "userId": "user_a1b2c3d4",
  "amount": 15,
  "referenceId": "vid_tx_20261029_091801_102",
  "description": "Cappy AI dịch và lồng tiếng video Douyin (28s)"
}
```
* **Xử lý mã lỗi**:
  * `200 OK`: Trừ thành công, trả về số dư mới.
  * `400 BAD_REQUEST / INSUFFICIENT_CREDITS`: Tài khoản không đủ số dư.

---

## 4. CHI TIẾT SỬA ĐỔI CODE TRONG CAPPY (FASTAPI)

### 4.1. Cấu hình biến môi trường (`.env` và `config/settings.py`)
Thêm các biến cấu hình:
```bash
# Pricing Gateway Integration
PRICING_GATEWAY_URL="http://pricing-gateway:8088"
PRICING_GATEWAY_TIMEOUT=5.0
BILLING_ENABLED=true
FREE_TRIAL_CREDITS=20
```

### 4.2. Middleware / Dependency Kiểm Tra & Khấu Trừ (`gateway/deps.py`)
Tạo hàm dependency `verify_and_meter_credit(resource_type: str)`:

```python
import httpx
import math
from fastapi import Header, HTTPException, Request

async def check_and_reserve_credit(
    request: Request,
    resource_type: str, # "image" hoặc "video"
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    authorization: str | None = Header(None)
):
    """
    1. Trích xuất userId từ header (hoặc decode từ token).
    2. Gọi Pricing Gateway kiểm tra số dư tối thiểu (1 credit cho ảnh, 5 credit cho video).
    3. Trả về userId để endpoint thực thi và trừ điểm khi kết thúc.
    """
    if not settings.BILLING_ENABLED:
        return "anonymous_local"

    user_id = x_user_id or "anonymous_local"
    required_min = 1 if resource_type == "image" else 5

    async with httpx.AsyncClient(base_url=settings.PRICING_GATEWAY_URL, timeout=5.0) as client:
        try:
            resp = await client.get(f"/api/v1/credits/{user_id}/balance")
            if resp.status_code == 404:
                # User mới -> Tự động kích hoạt ví với Free Trial 20 Credits
                await client.post(f"/api/v1/credits/{user_id}/grant-trial", json={"amount": settings.FREE_TRIAL_CREDITS})
                current_balance = settings.FREE_TRIAL_CREDITS
            else:
                resp.raise_for_status()
                current_balance = resp.json()["data"]["balance"]
        except Exception as e:
            # Nếu Pricing Gateway offline hoặc lỗi, ghi log cảnh báo
            logger.error(f"Cannot verify credit from Pricing Gateway: {e}")
            raise HTTPException(status_code=503, detail="Billing service temporarily unavailable")

    if current_balance < required_min:
        raise HTTPException(
            status_code=402,
            detail={
                "code": "PAYMENT_REQUIRED",
                "message": f"Số dư của bạn ({current_balance} credits) không đủ thực hiện yêu cầu ({required_min} credits). Vui lòng nạp thêm.",
                "checkout_url": f"/billing/checkout?userId={user_id}"
            }
        )

    return user_id
```

### 4.3. Cập nhật Endpoint Ảnh (`gateway/routes/images.py`)
Sau khi pipeline dịch ảnh hoàn thành và trả về kết quả thành công:
```python
# Trừ 1 credit
async with httpx.AsyncClient(base_url=settings.PRICING_GATEWAY_URL, timeout=5.0) as client:
    await client.post(
        "/api/v1/credits/deduct",
        json={
            "userId": user_id,
            "amount": 1,
            "referenceId": f"img_{uuid.uuid4().hex[:12]}",
            "description": "Cappy dịch 1 ảnh sản phẩm"
        }
    )
```

### 4.4. Cập nhật Endpoint Video (`gateway/routes/videos.py`)
Tính thời lượng video thông qua metadata (`ffprobe` hoặc `moviepy`):
```python
duration_seconds = get_video_duration(video_bytes) # e.g., 28.5 seconds
blocks_10s = math.ceil(duration_seconds / 10.0)    # e.g., 3 blocks
credits_to_deduct = blocks_10s * 5                 # e.g., 15 credits

# Sau khi render video thành công:
async with httpx.AsyncClient(base_url=settings.PRICING_GATEWAY_URL, timeout=5.0) as client:
    await client.post(
        "/api/v1/credits/deduct",
        json={
            "userId": user_id,
            "amount": credits_to_deduct,
            "referenceId": f"vid_{uuid.uuid4().hex[:12]}",
            "description": f"Cappy dịch video ({duration_seconds:.1f}s - {credits_to_deduct} credits)"
        }
    )
```

---

## 5. ĐĂNG NHẬP GOOGLE (GMAIL) & CƠ CHẾ JWT `accessToken`

Hệ thống sử dụng **1 luồng tính Coin duy nhất** cho mọi tài khoản (không tách luồng Subscription hay Admin phức tạp). Khi cần cấp nhiều coin cho tài khoản Admin/nội bộ, chỉ cần chỉnh cột `balance` của email đó trong bảng `wallets` (database `pricing_gateway`).

### 5.1. Đăng nhập Gmail nhận `accessToken`
* **Endpoint**: `POST /v1/auth/google`
* **Body Request**:
```json
{
  "token": "<Google id_token từ Web hoặc OAuth2 access_token từ chrome.identity>"
}
```
* **Luồng xử lý Backend**:
  1. Xác thực token trực tiếp với Google (`oauth2.googleapis.com/tokeninfo` hoặc `googleapis.com/oauth2/v3/userinfo`).
  2. Lấy địa chỉ Gmail (ví dụ `user@gmail.com`) làm `userId` định danh ví trong bảng `wallets`.
  3. Nếu là tài khoản đăng nhập lần đầu (`balance == 0` và chưa từng nhận `TRIAL`), tự động cộng **20 Credits tân thủ**.
  4. Ký và trả về `accessToken` (JWT HS256, thời hạn mặc định 30 ngày).
* **Phản hồi mẫu**:
```json
{
  "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "tokenType": "Bearer",
  "user": {
    "userId": "user@gmail.com",
    "email": "user@gmail.com",
    "name": "Nguyen Van A",
    "picture": "https://lh3.googleusercontent.com/...",
    "balance": 20
  }
}
```

### 5.2. Xem thông tin tài khoản & số dư Coin hiện tại
* **Endpoint**: `GET /v1/auth/me`
* **Headers**: `Authorization: Bearer <accessToken>`
* **Phản hồi**: Thông tin `user` kèm `balance` thời gian thực từ `Pricing Gateway`.

### 5.3. Gọi API Dịch Ảnh / Video với `accessToken`
* Client gửi header `Authorization: Bearer <accessToken>` khi gọi `POST /v1/images/translate` hoặc `POST /v1/videos/translate`.
* Cappy tự giải mã JWT lấy `email` (`userId`), kiểm tra số dư và trừ coin sau khi dịch hoàn tất.
* Khi Backend Cappy trả về `HTTP 402`, Client tự động hiển thị modal quét mã VietQR để nạp thêm Credits.

---

## 6. KẾ HOẠCH TRIỂN KHAI VÀ KIỂM THỬ (VERIFICATION CHECKLIST)

- [x] **Bước 1**: Đảm bảo container `pricing-gateway` (port 8088) chạy ổn định trong `common-docker.yml`.
- [x] **Bước 2**: Tích hợp `BillingService` và `auth` routes (`/v1/auth/google`, `/v1/auth/me`) vào repo `capyy`.
- [x] **Bước 3**: Gọi thử test endpoint ảnh bằng `curl` kèm JWT `accessToken` hoặc `X-User-Id`:
  * Lần 1-20: Trừ điểm bình thường.
  * Lần 21: Nhận HTTP 402 kèm thông tin thanh toán.
- [x] **Bước 4**: Thử nghiệm gọi endpoint video và xác minh số điểm bị trừ tương ứng thời lượng video.
