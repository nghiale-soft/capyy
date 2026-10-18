# BÁO CÁO LỖI (BUG LOG): DỊCH & INPAINT ẢNH SẢN PHẨM 1688 (CAPYY IMAGE TRANSLATION)

- **Mã sự vụ**: `BUG-CAPYY-IMG-20261019`
- **Dịch vụ**: Capyy AI Gateway (`POST /v1/images/translate`)
- **Tập tin source liên quan**: `gateway/services/image_service.py`, `gateway/routes/images.py`
- **Nguồn kiểm thử**: Ảnh sản phẩm thực tế từ sàn 1688 (Máy uốn tóc đa năng & Máy massage cầm tay)
- **Hệ thống gọi API**: Cross-Border OMS ERP (Đã test trực tiếp trên DOM và lưu trữ qua Unlimited Drive Telegram)
- **Ngày ghi nhận**: 19/10/2026

---

## 1. TÀI NGUYÊN KIỂM THỬ ĐÍNH KÈM (TEST FIXTURES)

Tất cả ảnh mẫu gốc và ảnh kết quả sau dịch đã được lưu trữ trực tiếp trong thư mục này để developer Capyy tiện mở kiểm tra:

| File | Mô tả | Đường dẫn tương đối |
| :--- | :--- | :--- |
| **Ảnh 1688 Gốc (Máy uốn tóc)** | Ảnh sản phẩm chứa tiêu đề lớn, badge tính năng bo tròn, và 4 đốm tròn cam | [`./curler_original.jpg`](./curler_original.jpg) |
| **Ảnh dịch AI (Capyy Gen)** | Kết quả dịch thực tế do Capyy trả về (đang dính các lỗi bên dưới) | [`./curler_translated.jpg`](./curler_translated.jpg) |
| **Ảnh DOM Test ERP** | Ảnh chụp màn hình popup Edit Product Variant trên ERP lúc tích hợp gọi Capyy | [`./dom_test_preview.png`](./dom_test_preview.png) |
| **Ảnh 1688 Gốc (Máy massage)** | Ảnh sản phẩm test phụ chứa text tiếng Trung phức tạp | [`./massage_original.jpg`](./massage_original.jpg) |
| **Ảnh dịch AI (Máy massage)** | Kết quả dịch thực tế của máy massage | [`./massage_translated.jpg`](./massage_translated.jpg) |

---

## 2. CÁC LỖI CHI TIẾT (DEFECTS) & PHÂN TÍCH NGUYÊN NHÂN (ROOT CAUSE)

### LỖI 1: Bị sót ký tự đầu bên trái trên các Badge hình viên thuốc trắng (Pill Badges)
- **Hiện tượng**:
  - Trên các badge trắng bo tròn góc trái ảnh (*"亿级负离子"*, *"滴滴提示音"*), chữ Hán đầu tiên (*"亿"* và *"滴"*) bị cắt mất một nửa nét bên trái và không bị xoá, đè chữ tiếng Việt lên trên tạo thành vết lem nhem.
- **Vị trí code nguyên nhân**:
  - File: [`gateway/services/image_service.py`](../../gateway/services/image_service.py) khoảng dòng 472–476:
    ```python
    # Protect left icon in wide pill badges if box was overly inclusive
    badge_aspect = (xmax - xmin) / max(1, ymax - ymin)
    if is_badge and badge_aspect > 2.0 and xmin < 100:
        xmin = xmin + int((xmax - xmin) * 0.35)
    ```
- **Phân tích kỹ thuật**:
  - Logic cứng `xmin = xmin + int((xmax - xmin) * 0.35)` đang giả định rằng các badge nằm sát lề trái (`xmin < 100`) và có tỷ lệ ngang > 2.0 thì 35% diện tích bên trái chắc chắn là icon hình ảnh cần bảo vệ.
  - Tuy nhiên, trong thực tế ảnh 1688, rất nhiều pill badge không có icon hoặc icon chỉ chiếm lề rất nhỏ (< 10%). Việc tự động dịch `xmin` sang phải 35% đã cắt ngang chữ Hán đầu tiên, làm thuật toán inpainting bỏ sót ký tự này.
- **Đề xuất khắc phục**:
  1. Không tự ý dịch `xmin` cứng 35%. Thay vào đó, kiểm tra contour connected components hoặc dùng trực tiếp `char_boxes_2d` của từng ký tự do Gemini trả về.
  2. Bổ sung prompt yêu cầu Gemini phân tách rõ `icon_box_2d` và `text_box_2d` thay vì gộp chung một bounding box.

---

### LỖI 2: Xoá mất nền đốm tròn màu cam (Color Bubble Containers) & Chữ dịch bị mờ, mất tương phản
- **Hiện tượng**:
  - Ở 4 đốm tròn màu cam nổi bật phía dưới ảnh (*"玉米夹"*, *"直发梳"*, *"蛋卷"*, *"32mm大波浪"*):
    - Sau khi dịch, toàn bộ hình tròn màu cam bị thuật toán Inpainting xoá sạch thành nền trời/nền sản phẩm.
    - Chữ tiếng Việt dịch ra (*"Kẹp ngô"*, *"Lược chải"*, *"Bánh cuộn"*, *"Sóng lớn 32mm"*) màu trắng nhạt render thẳng lên nền ảnh không có tương phản, làm mất hẳn điểm nhấn thiết kế của bao bì sản phẩm.
- **Vị trí code nguyên nhân**:
  - File: [`gateway/services/image_service.py`](../../gateway/services/image_service.py) khoảng dòng 525–541:
    ```python
    diff = np.linalg.norm(roi.astype(np.float32) - bg_est.astype(np.float32), axis=2)
    raw_mask = cv2.bitwise_or((diff > 8).astype(np.uint8) * 255, m_c)
    ...
    dilated = cv2.dilate(clean_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)))
    ```
- **Phân tích kỹ thuật**:
  - Khi một vùng text nằm trong một container nhỏ (như hình tròn màu cam trên nền xanh), `bg_est` lấy mẫu từ viền ngoài ROI (nền xanh của ảnh lớn).
  - Do màu cam của hình tròn lệch rất lớn so với màu xanh viền ngoài (`diff > 8`), thuật toán nhận nhầm TOÀN BỘ hình tròn màu cam là "nét chữ cần xoá" thay vì chỉ xoá nét chữ bên trong.
  - Sau đó phép `dilate` với kernel 6x6 xoá hoàn toàn đốm cam và hoà tan vào nền trời xung quanh.
- **Đề xuất khắc phục**:
  1. Với các vùng được nhận diện là `bubble` / `badge` / `tag`, chỉ tính ngưỡng nhị phân chữ cục bộ (Local Otsu / Adaptive Thresholding) dựa trên màu nền nội tại của container chứ không lấy viền ngoài.
  2. Giảm kernel dilation xuống 2x2 hoặc 3x3 khi kích thước ROI nhỏ (< 80px) để không ăn lấn ra ngoài viền container.
  3. Khi render text thay thế: nếu phát hiện màu nền sáng hoặc màu chữ trùng màu nền (độ tương phản thấp), tự động thêm viền stroke mỏng (1px - 2px) hoặc drop shadow để đảm bảo khả năng đọc.

---

### LỖI 3: Dải ruy băng chân trang (Ribbon Banner) bị bẻ dòng làm vỡ bố cục
- **Hiện tượng**:
  - Dải ruy băng trắng ở đáy ảnh (*"三年质保 免费试用 赠运费险"* - Bảo hành 3 năm, dùng thử miễn phí, tặng bảo hiểm vận chuyển):
    - Tiếng Trung có 12 ký tự liền mạch trên 1 hàng. Khi dịch sang tiếng Việt: *"Bảo hành 3 năm Dùng thử miễn phí Tặng bảo hiểm vận chuyển"*, chữ dài hơn nhiều và bị wrap thành 2 dòng, đè lấn ra ngoài dải ruy băng trắng và đè lên viền hoa phía dưới.
- **Vị trí code nguyên nhân**:
  - File: [`gateway/services/image_service.py`](../../gateway/services/image_service.py) - hàm `_wrap_text` và `_render_translations`:
- **Phân tích kỹ thuật**:
  - Thuật toán wrap text tự động xuống dòng khi chiều dài vượt quá width của box, nhưng lại không kiểm tra giới hạn chiều cao khả dụng (`height`) của ruy băng.
- **Đề xuất khắc phục**:
  1. Thêm cơ chế tự động giảm cỡ chữ (Auto Font Downscale): nếu chiều cao sau khi wrap vượt quá 85% chiều cao của banner, giảm cỡ font xuống cho tới khi vừa vặn trên 1 dòng duy nhất.
  2. Hoặc nếu cụm từ có dấu cách phân tách các cam kết, tách thành 3 badge nhỏ độc lập hoặc dùng dấu gạch đứng `|` / dấu chấm `•` phân cách ngắn gọn: *"BH 3 năm • Dùng thử 0đ • Tặng BHVC"*.

---

## 3. LỆNH CURL ĐỂ BÊN CAPYY TÁI HIỆN LỖI (REPRODUCE COMMAND)

Developer Capyy có thể chạy lệnh cURL trực tiếp với file ảnh đính kèm ngay trong repo:

```bash
# Đứng tại thư mục gốc của Capyy repo:
curl -X POST http://localhost:1221/v1/images/translate \
  -H "Content-Type: multipart/form-data" \
  -F "image=@docs/bugs/image_translation/curler_original.jpg" \
  -F "target_lang=vi" \
  -F "instruction=Giữ nguyên khung hình tròn màu cam và viền badge, chỉ xoá chữ Trung Quốc bên trong" \
  --output docs/bugs/image_translation/reproduced_output.jpg
```

---

## 4. TÓM TẮT CHECKLIST CẦN FIX CHO BẢN PHÁT HÀNH TIẾP THEO

- [x] Gỡ bỏ việc cắt cứng `xmin += 0.35 * width` tại `gateway/services/image_service.py` -> Đã xoá bỏ hoàn toàn, hỗ trợ phân loại chính xác badge có icon vs không có icon qua prompt và prompt schema. Ký tự đầu tiên không còn bị cắt.
- [x] Tách biệt thuật toán bóc tách nét chữ (Text Stroke Segmentation) bên trong container màu sắc (giữ lại đốm tròn cam, tag xanh/đỏ) -> Đã tách biệt phân vùng container (badge, bubble, callout, tag, chip, sticker), dùng khoảng cách màu và Connected Components nội tại để chỉ xoá nét chữ, giữ nguyên 100% hình khối và màu nền container.
- [x] Thêm tính năng Auto-scale font size theo chiều cao cho các banner dạng dài nằm ngang (Horizontal Ribbon Banners) -> Đã bổ sung nhận diện dải banner/ribbon dài nằm ngang, ép buộc layout 1 dòng (single-line only) và tự động giảm cỡ font (Auto Font Downscale) vừa vặn bên trong ribbon.
- [x] Bổ sung stroke outline tối màu dự phòng khi render chữ trắng trên nền sáng -> Đã bổ sung cơ chế Contrast Safety: tự động kích hoạt outline 1px khi phát hiện chữ sáng trên nền sáng.
- [x] Bổ sung hỗ trợ form field `image` và alias `target_lang` tại endpoint `POST /v1/images/translate` để tương thích hoàn toàn với lệnh cURL của ERP.

> **Trạng thái**: Đã fix hoàn tất, đã test verify trực tiếp trên Docker `capyy` và xuất kết quả tại `docs/bugs/image_translation/reproduced_output.jpg`. Cả 3 lỗi đã được triệt tiêu 100%.
