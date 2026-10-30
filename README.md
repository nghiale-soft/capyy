# Capyy AI Gateway

<img src="tool/web/static/capyy.png" alt="Capyy" width="120">

> Cổng AI trung tâm hợp nhất (Unified AI Gateway) cho toàn bộ ứng dụng trên máy Mac và các dịch vụ microservice trong Docker (`common-service`).

---

## 🚀 1. Thông số kết nối nhanh (Quick Connection)

Mọi ứng dụng (App dịch thuật, Chatbot, Backend microservice, CLI tool...) chỉ cần cấu hình theo chuẩn **OpenAI**:

| Thông số | Dành cho App trên máy Host (Mac) | Dành cho Container trong Docker |
| :--- | :--- | :--- |
| **Base URL** | `http://localhost:2221/v1` | `http://capyy:2221/v1` |
| **API Key** | `local-key` (hoặc chuỗi bất kỳ) | `local-key` (hoặc chuỗi bất kỳ) |
| **Model** | `gemini-3.5-flash-lite` (hoặc để trống) | `gemini-3.5-flash-lite` (hoặc để trống) |
| **Dashboard** | `http://localhost:2222` | `http://capyy:2222` |

> 💡 **Tự động gán Model:** Nếu request không truyền `model` hoặc truyền `"default"`, Capyy sẽ tự động định tuyến đến model tối ưu nhất hiện tại (`gemini-3.5-flash-lite`).

---

## 📡 2. Danh sách API Endpoints

Capyy hỗ trợ đầy đủ cả 2 chuẩn giao tiếp phổ biến nhất thế giới hiện nay:

### 2.1. `POST /v1/chat/completions` (Chuẩn OpenAI)
Dành cho các ứng dụng dịch thuật, web chat, SDK OpenAI.

- **Non-streaming (JSON):**
```bash
curl -X POST http://localhost:2221/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{
    "model": "gemini-3.5-flash-lite",
    "messages": [
      {"role": "user", "content": "Dịch sang tiếng Việt: Hello world!"}
    ]
  }'
```

- **Streaming SSE (`"stream": true`):**
```bash
curl -N -X POST http://localhost:2221/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{
    "model": "gemini-3.5-flash-lite",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Kể 1 câu chuyện vui ngắn"}
    ]
  }'
```

### 2.2. `GET /v1/models` (Danh sách Model)
- **Lấy danh sách model đã cấu hình (siêu nhanh):**
  ```bash
  curl http://localhost:2221/v1/models
  ```
- **Live Fetch trực tiếp từ Provider (`?refresh=true`):**
  Gọi thời gian thực sang nhà cung cấp để quét toàn bộ model mới nhất:
  ```bash
  curl "http://localhost:2221/v1/models?refresh=true"
  ```

### 2.3. `POST /v1/images/translate` (Dịch và Thay thế Chữ trên Ảnh Sản Phẩm)
Tự động OCR nhận diện tọa độ chữ, dịch sang Tiếng Việt (hoặc ngôn ngữ tùy chọn), xóa nền chữ gốc và chèn chữ tiếng Việt bằng font chuẩn (Roboto) vào thẳng ảnh.
* **Chi phí:** Gần như **0 đồng** (tận dụng Gemini Vision Free Tier 1.500 req/ngày, chỉ ~400 text tokens thay vì gen ảnh đắt đỏ).
* **Độ chính xác:** Giữ nguyên 100% hình ảnh sản phẩm thực tế, không bị méo mó hay vẽ sai chi tiết.

#### A. Gửi file ảnh (Multipart Form-Data - Nhận về trực tiếp file ảnh JPEG):
```bash
curl -X POST http://localhost:2221/v1/images/translate \
  -H "Authorization: Bearer local-key" \
  -F "file=@san-pham-1688.jpg" \
  -F "target_language=Vietnamese" \
  -o translated_product.jpg
```

#### B. Gửi JSON với Base64 / URL ảnh (Nhận JSON chứa Base64 + Bounding Boxes):
```bash
curl -X POST http://localhost:2221/v1/images/translate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{
    "image_url": "https://example.com/product.jpg",
    "target_language": "Vietnamese",
    "response_format": "json"
  }'
```

Response JSON:
```json
{
  "data": {
    "id": "img_0123456789abcdef0123456789abcdef",
    "version": 1,
    "expires_at": 1790000000.0,
    "image_base64": "data:image/jpeg;base64,...",
    "format": "jpeg",
    "regions_count": 8,
    "regions": [
      {
        "region_id": 0,
        "box_2d": [75, 43, 140, 295],
        "original_text": "咖啡色",
        "translated_text": "Màu cà phê",
        "bg_color_hex": "#b56943",
        "text_color_hex": "#ffffff",
        "align_h": "center",
        "align_v": "center",
        "group_id": "color_options",
        "rotation_deg": 0,
        "bold": true
      }
    ],
    "qc_report": {
      "approved": true,
      "qc_score": 100,
      "deterministic": true,
      "issues": []
    }
  }
}
```

Khi trả về file JPEG trực tiếp, `id`/`version` nằm trong các header
`X-Capyy-Image-Id`, `X-Capyy-Image-Version`, `X-Capyy-Image-Expires-At`.

#### C. Sửa một vùng đã dịch (user-guided repair):
Mỗi ảnh dịch được lưu thành một session có TTL (mặc định 7 ngày). Khi phát hiện lỗi, gửi `region_id` (hoặc `bbox_2d`) và mô tả lỗi; gateway chỉ gọi lại model cho vùng đó và render lại từ ảnh gốc, không chồng chữ lên bản cũ.

```bash
curl -X POST http://localhost:2221/v1/images/img_0123456789abcdef0123456789abcdef/repair \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-key" \
  -d '{
    "instruction": "chữ 蛋 còn sót ở góc dưới trái, xoá giúp",
    "region_ids": [2],
    "idempotency_key": "fix-leftover-egg-1"
  }'
```

* `region_ids`: danh sách `region_id` lấy từ response translate.
* `bbox_2d`: `[ymin, xmin, ymax, xmax]` (0–1000) nếu muốn chọn vùng theo tọa độ.
* `base`: chỉ hỗ trợ `original` ở v1 (tránh double text).
* `idempotency_key`: gửi lại cùng key sẽ trả version đã lưu, không gọi model lần nữa.
* `enable_qc`: bật thêm LLM inspection (tùy chọn); mặc định chỉ chạy **deterministic QA miễn phí** để bắt chữ nguồn còn sót.
* Ảnh vượt quá `CAPYY_IMAGE_STORE_MAX_BYTES` (mặc định 25 MiB) bị từ chối với HTTP 413 trước khi gọi model.
* Các endpoint phụ: `GET /v1/images/{id}` (metadata + versions), `GET /v1/images/{id}/content?version=N` (JPEG), `DELETE /v1/images/{id}`.

* 🚀 **Bộ Thuật Toán & Tính Năng Đồ Họa Thông Minh:**
  * **Đo độ nghiêng tự động bằng CV thuần (`cv2.fitLine` & PCA):** Tự động phát hiện các nét ký tự gốc, lọc bỏ nhiễu biên và dùng thuật toán khớp đường thẳng L2 distance để tính góc nghiêng vật lý thực tế ($\theta = -\operatorname{degrees}(\arctan2(vy, vx))$). Đồng bộ góc nghiêng trung vị cho toàn bộ Speech Bubbles, loại bỏ hoàn toàn việc đoán mò góc nghiêng của AI.
  * **Căn chỉnh tỷ lệ font chữ Bao bì sản phẩm (`is_packaging`):** Khóa cứng định dạng 1 dòng (single-line) và tự động đo chiều cao ký tự gốc (`orig_glyph_h`) để co giãn font chữ La-tinh theo tỷ lệ vàng ($0.60 \times \text{glyph\_h}$). Chữ nằm gọn gàng, thanh mảnh trong dải bao bì, không làm vỡ bố cục, không đè lên nơ hay chi tiết trang trí.
  * **Tẩy xóa nét chữ tự nhiên (Context-Aware Inpainting) & Phủ hạt noise (Film Grain Injection):** Tự động phát hiện mask nét chữ gốc bằng Otsu thresholding và dùng thuật toán OpenCV Telea bù đắp màu nền xung quanh. Tự động đo độ lệch chuẩn texture để tiêm hạt noise Gaussian, loại bỏ hoàn toàn hiện tượng bệt màu mờ nhựa (plastic/waxy blur), bảo toàn 100% texture gỗ, tường và dải gradient.
  * **Bảo vệ Icon & Tránh tràn viền thông minh (Anti-Overflow & Icon Protection):** Tự động nhận diện huy hiệu chứa icon bên trái để giữ nguyên 100% icon. Áp dụng lề an toàn động ($\ge 12\%$ cho bong bóng thoại, bảo vệ mép biên màn hình), đảm bảo text không bao giờ chạm viền.
  * **Hiệu ứng đồ họa CSS & Typography cao cấp:** Hỗ trợ gradient đa điểm (vàng-cam, ánh kim bronze metallic), viền chữ tương phản (stroke outline), bóng đổ 3D (drop-shadow), và tự động chuyển đổi giữa Sans-serif (Roboto) hiện đại và Serif (Georgia) sang trọng.
  * **Đồng bộ Font Size theo nhóm (`group_id`):** Các label có cùng `group_id` (ví dụ 4 nút tính năng) sẽ tự động dùng chung cỡ font nhỏ nhất trong nhóm để giao diện đồng bộ 100%.
  * **Căn chỉnh 2 chiều (`align_h`, `align_v`):** Tự động căn giữa, trái, phải, trên, dưới tương ứng theo vị trí nút hoặc banner.

### 2.4. `POST /v1/videos/translate` (Dịch, Xóa Sub & Lồng Tiếng Video Sản Phẩm)
Tự động nhận diện subtitle/chữ tiếng Trung trong video, inpainting xóa sub cũ, dịch chuẩn ngữ cảnh TMĐT/review, tạo giọng đọc lồng tiếng AI (TTS tiếng Việt Hoài My / Nam Minh), và chèn sub Việt chuẩn timing.

* **Định dạng hỗ trợ:** MP4, MOV, WebM.
* **Thời gian xử lý:** Tối ưu hóa đa luồng với ffmpeg và inpainting keyframes.

#### Gửi file video (Multipart Form-Data - Nhận về trực tiếp video MP4):
```bash
curl -X POST http://localhost:2221/v1/videos/translate \
  -H "Authorization: Bearer local-key" \
  -H "X-User-Id: user_12345" \
  -F "file=@video_douyin_1688.mp4" \
  -F "target_language=Vietnamese" \
  -F "voice=female" \
  -F "cover_old_sub=true" \
  -o translated_dubbed_video.mp4
```

---

### 2.5. Hệ Thống Quản Lý Tín Dụng & Thanh Toán (Credit & Pricing Gateway Integration)

Cappy tích hợp trực tiếp với microservice [Pricing Gateway](file:///Users/nghiale98/Documents/1.My%20Companies/ChinhThien/Source%20Code/Pricing%20Gateway) (cổng 2088) để quản lý số dư tín dụng cho Web App và Extension:

#### Quy tắc tiêu thụ Tín dụng (Credit Rules):
* 📸 **Dịch 1 ảnh hoàn chỉnh:** Tiêu thụ **1 Credit** (~200đ - 250đ).
* 🎥 **Dịch 10 giây video:** Tiêu thụ **5 Credits** (quy đổi theo công thức `ceil(duration_seconds / 10) * 5`).
  * Video 30s review Douyin = **15 Credits**.
  * Video 60s giới thiệu sản phẩm = **30 Credits**.

#### Các gói nạp tiền tự động (Napas 24/7 VietQR PayOS):
1. 🎁 **Free Trial**: Tặng **20 Credits** cho tài khoản mới / extension mới.
2. 🥉 **Gói 50.000 VNĐ**: **150 Credits** (~333đ/credit).
3. 🥈 **Gói 100.000 VNĐ (Phổ thông)**: **400 Credits** (~250đ/credit).
4. 🥇 **Gói 200.000 VNĐ (Pro/Studio)**: **1.000 Credits** (~200đ/credit).

> ⚠️ **Khi số dư < mức yêu cầu:** Gateway trả về HTTP `402 Payment Required` kèm URL/QR nạp tiền để Client hiển thị modal thanh toán.

---

### 2.6. `GET /healthz` (Health Check)
```bash
curl http://localhost:2221/healthz
# Trả về: {"status": "ok"}
```

---

## 🛠 3. Hướng dẫn tích hợp cho từng App

### 3.1. Ứng dụng Dịch thuật (Bob, Immersive Translate, PopClip)
- **Loại dịch vụ / Dịch vụ tùy chỉnh:** Chọn `OpenAI` hoặc `OpenAI-compatible`.
- **API URL / Host:** `http://localhost:2221/v1` (hoặc `http://localhost:2221/v1/chat/completions` tùy app).
- **API Key:** `local-key`.
- **Model:** `gemini-3.5-flash-lite`.

### 3.2. Trong Docker Microservice (Node.js / Python / Go)
Ví dụ gọi từ service `iot-apartment` hoặc `common-zalo-api`:

#### Python (sử dụng thư viện `openai` hoặc `httpx`):
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://capyy:2221/v1",
    api_key="local-key",
)

response = client.chat.completions.create(
    model="gemini-3.5-flash-lite",
    messages=[{"role": "user", "content": "Tóm tắt tin nhắn này..."}],
)
print(response.choices[0].message.content)
```

#### Node.js (OpenAI SDK):
```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://capyy:2221/v1",
  apiKey: "local-key",
});

const completion = await client.chat.completions.create({
  model: "gemini-3.5-flash-lite",
  messages: [{ role: "user", content: "Xin chào!" }],
});
console.log(completion.choices[0].message.content);
```

### 3.3. Coding Agents (Claude Code, Codex, Cline, Aider)
- **Claude Code CLI:**
  ```bash
  export ANTHROPIC_BASE_URL="http://localhost:2221"
  claude
  ```
- **OpenAI-compatible tools:**
  ```bash
  export OPENAI_BASE_URL="http://localhost:2221/v1"
  export OPENAI_API_KEY="local-key"
  ```

### 3.4. Dịch Ảnh Sản Phẩm Hàng Loạt (Python / Node.js)

#### Python:
```python
import httpx

# 1. Gửi file ảnh trực tiếp và nhận lại ảnh đã dịch
with open("san_pham_1688.jpg", "rb") as f:
    resp = httpx.post(
        "http://capyy:2221/v1/images/translate",
        headers={"Authorization": "Bearer local-key"},
        files={"file": ("san_pham.jpg", f, "image/jpeg")},
        data={"target_language": "Vietnamese", "response_format": "image"},
        timeout=60.0
    )

if resp.status_code == 200:
    with open("san_pham_vietnam.jpg", "wb") as f:
        f.write(resp.content)
    print("Dịch ảnh thành công!")
```

#### Node.js / TypeScript:
```typescript
import fs from "fs";
import FormData from "form-data";
import axios from "axios";

async function translateProductImage(imagePath: string, outputPath: string) {
  const form = new FormData();
  form.append("file", fs.createReadStream(imagePath));
  form.append("target_language", "Vietnamese");
  form.append("response_format", "image");

  const response = await axios.post("http://capyy:2221/v1/images/translate", form, {
    headers: {
      ...form.getHeaders(),
      Authorization: "Bearer local-key",
    },
    responseType: "arraybuffer",
  });

  fs.writeFileSync(outputPath, response.data);
  console.log("Ảnh đã dịch lưu tại:", outputPath);
}
```

---

## 🐳 4. Vận hành trong `common-docker.yml`

Capyy được quản lý tập trung trong file `common-docker.yml`:

```bash
# Khởi động Capyy cùng cụm dịch vụ chung
docker compose -f common-docker.yml up -d capyy

# Xem logs thời gian thực
docker compose -f common-docker.yml logs -f capyy

# Khởi động lại khi đổi cấu hình
docker compose -f common-docker.yml restart capyy
```

- **Database:** MariaDB schema `capyy` trên container `common-mariadb`.
- **Dữ liệu cấu hình:** Lưu tại `./AI-Gateway-Provider/capyy/data/config`.
- **Lịch sử chat & sessions:** Lưu tại `./AI-Gateway-Provider/capyy/data/app`.
