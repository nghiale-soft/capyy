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
- [x] **Bảo vệ viền khung bao container (Cloud / Speech Bubble Border Frame Protection)**: Bổ sung bộ lọc connected component `is_frame` (loại bỏ stroke bao trùm >60% cả 2 chiều hoặc >82% một chiều) và `touches_edge` nới rộng theo margin 5px, ngăn chặn triệt để tình trạng inpaint nhầm viền đám mây/bong bóng thành vết nhoè.
- [x] **Tối ưu typography vùng chứa đám mây/bong bóng**: Tăng tỷ lệ padding vùng khả dụng (`pad_x = 16%`, `pad_y = 14%`) cho hình khối cong/mây để chữ tiếng Việt không chạm mép viền uốn lượn.

> **Trạng thái**: Đã fix hoàn tất, đã test verify trực tiếp trên Docker `capyy` qua cả Unit Tests tự động và lệnh cURL Live API. Toàn bộ viền khung đám mây giữ nguyên độ sắc nét 100%, chữ tiếng Việt hiển thị rõ ràng, cân đối.

---

## 5. BỔ SUNG LỖI GẦN NHẤT: VIỀN ĐÁM MÂY BỊ INPAINT NHOÈ (EGG STEAMER CLOUD CALLOUTS)

### Hiện tượng:
- Trên ảnh nồi hấp trứng (chứa 2 đám mây chứa chữ *"单层 蒸1-6个蛋"* và *"双层 蒸1-12个蛋"*), sau khi dịch sang *"MỘT TẦNG / Hấp 1-6 quả"*, toàn bộ viền xanh lá đậm uốn lượn của đám mây bị xoá biến mất thành một vệt mờ màu trắng/vàng nhoè nhoẹt, đồng thời chữ tiếng Việt bị lấn ra sát mép.

### Phân tích nguyên nhân kỹ thuật:
1. `_inpaint_canvas()` mở rộng biên an toàn `margin = 6` pixel xung quanh `erase_box_2d` của container. Do đó, đường viền đám mây nằm cách mép ROI khoảng 5-6 pixel (`cx ≈ 5-6, cy ≈ 5-6`).
2. Điều kiện kiểm tra chạm viền trước đây là `(cx <= 1 or cy <= 1)` -> Đánh giá thành `False`!
3. Diện tích nét viền mảnh của đám mây chỉ chiếm ~13-14% tổng diện tích ROI (nhỏ hơn ngưỡng `0.45` của hình khối đặc) -> Đánh giá thành `False`!
4. Kết quả: Thuật toán nhận diện nhầm toàn bộ đường viền xanh lá bao quanh đám mây là chữ Hán cần xoá, dẫn đến việc dùng Telea Inpainting xoá sạch viền đám mây thành một bãi kem trắng nhoè.

### Giải pháp nâng cấp vĩnh viễn:
1. **Bộ lọc Enclosing Border Frame (`is_frame`)**: Nhận diện bất kỳ connected component nào bao phủ phần lớn chiều ngang và dọc (`cw >= 0.60 * roi_w and ch >= 0.60 * roi_h`), hoặc đường kẻ dài (`cw >= 0.82 * roi_w` / `ch >= 0.82 * roi_h`) -> Đây chính là khung viền container, bắt buộc loại trừ khỏi mask xoá.
2. **Nới rộng ngưỡng `touches_edge`**: Nâng từ `<= 1` lên `<= 5` pixel để tương thích với `margin = 6`.
3. **Hardened Fallback**: Trong trường hợp kích hoạt ngưỡng fallback, cũng chạy qua connected components để tuyệt đối không inpaint viền khung ngoài.
4. **Padding hình khối hữu cơ (Cloud / Bubble)**: Đám mây và bong bóng thoại hẹp lại ở các góc uốn lượn. Tự động áp dụng `pad_x = max(14, int(box_w * 0.16))` và `pad_y = max(8, int(box_h * 0.14))` để chữ luôn nằm gọn gàng ở tâm vùng an toàn.

---

## 6. LỖI SÓT KÝ TỰ ĐUÔI DO GỘP CỤM DÒNG DÀI & ĐỘT PHÁ TÁCH KHUNG THEO TỪNG KÝ TỰ (CHARACTER-BY-CHARACTER DETECTION)

- **Mã sự vụ**: `BUG-CAPYY-IMG-20261028-01`
- **Tập tin liên quan**: `gateway/services/image_service.py`
- **Ảnh kiểm thử**: Thảm tập yoga định tuyến (`yoga_mat_original.jpg`)

### Hiện tượng:
- Dòng chữ nhãn bên phải như `"1000MM 高手引导线"`, `"800MM 老手引导线"`, `"500MM 新手引导线"`:
  - Khi xoá chữ Hán để dịch sang tiếng Việt, các chữ ở đuôi bên phải như `"引导线"` và đặc biệt là chữ `"线"` bị bỏ sót hoàn toàn, để lại vết đen nguyên bản trên nền trắng.
  - Khi chèn chữ tiếng Việt lên, các chữ Hán sót lại nằm ngay cạnh hoặc đè một phần lên chữ dịch gây mất thẩm mỹ nghiêm trọng.

### Phân tích nguyên nhân kỹ thuật:
1. **Prompt cấm chia nhỏ từ**: Trước đây prompt có câu lệnh cấm: *"Do NOT slice words on the exact same line into isolated single-word fragments! Keep together as a single cohesive line region"*.
2. **LLM cắt cụt tọa độ ngang (`xmax`)**: Khi bị ép gom toàn bộ cụm `"1000MM 高手引导线"` vào 1 box duy nhất, AI Vision (Gemini) thường ước lượng thiếu biên phải (dừng ở `xmax = 885` thay vì `946`), bỏ rơi 1–2 chữ Hán ở cuối dòng.
3. **Giới hạn Output Token (`max_tokens: 4096`)**: Khi cố gắng trả về tọa độ nhiều vùng chi tiết, độ dài phản hồi JSON vượt quá 4096 token khiến response bị ngắt cụt giữa chừng và gây lỗi `json.loads`.
4. **Inpainting bỏ qua `char_boxes_2d`**: Thuật toán xoá `_inpaint_canvas` trước đây chỉ quét qua `erase_box_2d` tổng mà không ưu tiên lặp qua từng bounding box ký tự riêng lẻ.

### Giải pháp nâng cấp triệt để:
1. **Đổi sang kiến trúc Bắt khung Từng Chữ (Character-by-Character Bounding Boxes)**:
   - Cập nhật prompt và JSON Schema: Bắt buộc trả về `char_boxes_2d` với bounding box ôm sát cho **từng ký tự Hán ngữ riêng lẻ** (ví dụ: `"高手"` tách thành 2 box riêng: Box 1 cho `"高"`, Box 2 cho `"手"`; `"1000MM 高手引导线"` tách thành 12 box riêng: 6 box cho `"1000MM"` và 6 box cho `"高手引导线"`).
   - Mục tiêu: Phân tách rõ ràng: **Khâu Xoá (Inpainting) thực hiện theo từng con chữ** để sạch 100% nền; **Khâu Dịch & Render thực hiện theo cả cụm câu** để giữ trọn ngữ nghĩa tự nhiên.
2. **Nâng `max_tokens: 8192`**: Tăng gấp đôi ngân sách token của Vision LLM để chứa trọn vẹn danh sách hàng chục tọa độ ký tự mà không bị cắt cụt JSON.
3. **Ưu tiên xoá qua `char_boxes_2d`**: Trong hàm `_inpaint_canvas`, nếu vùng có `char_boxes_2d` hoặc `word_boxes_2d`, inpainting sẽ duyệt qua từng box ký tự nhỏ để tẩy sạch từng nét mực gốc.
4. **Kết quả đo đạc**: Giá trị pixel tối thiểu tại vùng đuôi từng sót chữ `"线"` tăng từ `45` (nét mực đen) lên **`217 – 255`** (nền trắng sạch nguyên bản của thảm), triệt tiêu hoàn toàn bóng chữ Hán sót lại.

---

## 7. LỖI TEXT TIẾNG VIỆT BỊ NGHIÊNG (-13°) DO ẢO GIÁC ĐƯỜNG KẺ CHỈ DẪN & KIỂM TRA HÌNH HỌC TẤT ĐỊNH (GEOMETRIC SLANT VERIFICATION)

- **Mã sự vụ**: `BUG-CAPYY-IMG-20261028-02`
- **Tập tin liên quan**: `gateway/services/image_service.py`

### Hiện tượng:
- Các nhãn chỉ dẫn bên phải tấm thảm yoga (`"1000MM VẠCH CAO THỦ"`, `"800MM VẠCH CHUYÊN GIA"`,...) vốn nằm trên một đường thẳng ngang (0 độ), nhưng sau khi render tiếng Việt, toàn bộ text lại bị vẽ nghiêng cắm đầu xuống một góc `-13°`.

### Phân tích nguyên nhân kỹ thuật:
1. **AI bị "đánh lừa" bởi đường chỉ dẫn (Leader Lines) và phối cảnh 3D**:
   - Tấm thảm yoga nằm theo góc phối cảnh 3D (isometric), từ mép thảm có các đường kẻ chỉ dẫn chéo góc ~13° nối ra các nhãn chữ nằm ngang.
   - AI Vision (Gemini) nhìn thấy đường kẻ chéo và phối cảnh thảm nên tưởng nhầm chữ cũng bị nghiêng, trả về `rotation_deg = 13°` (hệ tọa độ ảnh quy đổi thành `-13°`).
2. **Hiệu ứng lan truyền của Group Harmonization**:
   - Capyy có cơ chế đồng bộ góc theo nhóm (`group_id: "callouts"`). Khi 3 nhãn trong nhóm bị AI gán góc nghiêng, thuật toán tính góc trung vị `-13°` và ép tất cả 6 nhãn trong nhóm phải nghiêng `-13°` theo.
3. **Nhiễu từ Auto-expand Box**:
   - Thuật toán `_expand_box_left` tự động mở rộng vùng xoá sang trái khi gặp nét mực. Khi chạy trên từng box ký tự, nó quét lấn vào đường kẻ chỉ dẫn và mép sản phẩm thảm, làm bôi lem màu nền đè lên viền thảm.

### Giải pháp nâng cấp triệt để:
1. **Kiểm tra góc nghiêng hình học trực tiếp từ `char_boxes_2d` (Geometric Slant Verification)**:
   - Khi đã có tọa độ từng ký tự, lấy tâm ký tự đầu $(cx_1, cy_1)$ và tâm ký tự cuối $(cx_n, cy_n)$ trên dòng để đo độ dốc thực tế:
     $$\text{deg} = \operatorname{degrees}(\arctan2(-(cy_n - cy_1), cx_n - cx_1))$$
   - **Quy tắc ngang tuyệt đối**: Nếu $|\text{deg}| < 3.5^\circ$, hệ thống xác nhận văn bản nằm ngang 100% $\rightarrow$ **tự động ghi đè `rotation_deg = 0`**, chặn đứng hoàn toàn ảo giác của AI.
2. **Bổ sung `CRITICAL HORIZONTAL RULE` trong Prompt**:
   - Chỉ thị rõ ràng cho LLM: *"Nếu chữ viết theo hàng ngang từ trái sang phải trên baseline bằng phẳng thì rotation_deg BẮT BUỘC bằng 0! Tuyệt đối không xoay chữ theo góc của đường kẻ chỉ dẫn hay góc phối cảnh 3D của sản phẩm."*
3. **Vô hiệu hóa Auto-expand khi đã có box ký tự**:
   - `_expand_box_left` và `_expand_box_right` chỉ chạy khi dùng single fallback box; khi đã có `char_boxes_2d` thì tắt hoàn toàn để không bao giờ lấn sang mép sản phẩm.
4. **Kết quả**: Tất cả 6 nhãn chỉ dẫn bên phải render thẳng hàng chuẩn xác $0^\circ$, không còn bị nghiêng lệch, mép sản phẩm được bảo toàn nguyên vẹn.

---

## 8. HỆ THỐNG LOGGING ĐIỀU HƯỚNG THEO 4 GIAI ĐOẠN (OBSERVABILITY & DEBUG LOGS)

Để đội ngũ phát triển luôn nắm rõ Capyy đã quyết định các tham số đầu vào nào cho từng vùng ảnh, hệ thống đã bổ sung cấu trúc log có tiền tố chuẩn hóa theo 4 giai đoạn xử lý:

```text
1. [Vision Decision - Region X]: Tham số đầu vào do AI Vision trả về
   Ví dụ: [Vision Decision - Region 5] '1000MM 高手引导线' -> '...' | rot=0° | group=callouts | layout=single_line_contain | fg=#2c2c2c | bg=#f5eadc | char_boxes=12

2. [Slant Detection - Region X]: Kết quả thẩm định hình học từ char_boxes_2d và cv2.fitLine
   Ví dụ: [Slant Detection - Region '1000MM 高手引导线'] Char-box baseline confirms horizontal text: deg=0.00° (dx=229.0, dy=0.0). Overriding LLM rot=13° -> 0°

3. [Group Harmonization - Group X]: Quyết định đồng bộ font-size và rotation giữa các phần tử cùng group_id
   Ví dụ: [Group Harmonization - 'callouts'] Majority horizontal (6/6 items <5°). Harmonized all rot -> 0°

4. [Render Execution - Region X]: Tham số cuối cùng truyền vào Pillow trước khi vẽ lên ảnh
   Ví dụ: [Render Execution - Region 5] text='1000MM VẠCH CAO THỦ' (lines=1) | font_size=12 | rot=0° | align=left | box=(556,112,758,134)
```

---

## 9. TỔNG KẾT TIẾN ĐỘ & TRẠNG THÁI HIỆN TẠI (PRODUCTION READY)

- [x] **Xoá sạch chữ Hán 100%**: Áp dụng cơ chế tách khung theo từng ký tự (`char_boxes_2d`), không còn sót nét chữ đuôi.
- [x] **Góc nghiêng chính xác (0°)**: Thuật toán Geometric Slant Verification bảo đảm chữ nằm ngang chuẩn xác, không bị nghiêng lệch theo đường kẻ hay phối cảnh.
- [x] **Bảo vệ viền sản phẩm & container**: Ngăn chặn tình trạng ăn lấn sang viền sản phẩm và viền đám mây.
- [x] **Font tiếng Việt chuẩn Unicode**: Sử dụng `Roboto-Bold` hiển thị trọn vẹn dấu tiếng Việt, không lỗi font.
- [x] **Logging quan sát toàn diện**: Theo dõi minh bạch 4 bước ra quyết định từ AI đến Render.
- [x] **Docker Container `capyy`**: Đã được đồng bộ code mới nhất, kiểm thử tự động đạt 100%, sẵn sàng phục vụ thực tế trên cổng `http://localhost:1221`.

