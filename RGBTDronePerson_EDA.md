# Báo cáo Exploratory Data Analysis — RGBTDronePerson

**Ngày phân tích:** 16/09/2026  
**Nguồn dữ liệu:** `RGBTDronePerson/`  
**Notebook tái lập:** [`RGBTDronePerson_EDA.ipynb`](./RGBTDronePerson_EDA.ipynb)

## Tóm tắt kết quả

RGBTDronePerson có **6.125 mẫu ghép cặp RGB–thermal**, tương ứng **12.250 file ảnh**, và **70.880 bounding box** thuộc 4 lớp. Tập dữ liệu có cấu trúc train/validation theo tỷ lệ 80/20, ảnh giữa hai modality được ghép đầy đủ và không phát hiện file thiếu hoặc corrupt.

Các điểm cần lưu ý nhất:

- Có **3 mẫu train** mà metadata COCO ghi kích thước `630×500`, trong khi cả ảnh visible và thermal thực tế đều là `640×512`.
- Có **1 bounding box không hợp lệ** do width bằng 0.
- Có **7 ảnh train không chứa object**; đây có thể là negative samples hợp lệ và cần được xác nhận trước khi loại bỏ.
- Dataset mất cân bằng mạnh: `person` chiếm **76,4%**, trong khi `uncertain` chỉ chiếm **2,0%**.
- **99,87%** bounding box thuộc nhóm small theo chuẩn COCO. Đây là bài toán small-object detection rõ rệt.
- Phân bố vị trí có center bias: **19,7%** tâm object nằm trong ô trung tâm chỉ chiếm 11,1% diện tích ảnh.
- Thermal là ảnh grayscale ba kênh, tối hơn nhưng có contrast trung bình cao hơn visible. Hai modality cần được normalize riêng.

---

## 1. Dataset overview

### 1.1. Quy mô dữ liệu

| Split | Số cặp RGB–thermal | Visible | Thermal | XML label | Object | Object/mẫu |
|---|---:|---:|---:|---:|---:|---:|
| Train | 4.900 | 4.900 | 4.900 | 4.900 | 56.314 | 11,493 |
| Validation | 1.225 | 1.225 | 1.225 | 1.225 | 14.566 | 11,891 |
| **Tổng** | **6.125** | **6.125** | **6.125** | **6.125** | **70.880** | **11,572** |

- Tỷ lệ mẫu train/validation là **80%/20%**.
- Mỗi sample gồm một ảnh visible và một ảnh thermal cùng tên file.
- Cả hai split đều có đầy đủ số lượng ảnh visible, thermal và XML tương ứng.

### 1.2. Class và định dạng

Dataset có 4 class:

| Category ID | Class |
|---:|---|
| 0 | `person` |
| 1 | `rider` |
| 2 | `crowd` |
| 3 | `uncertain` |

Các định dạng được cung cấp:

- Visible: `<split>/visible/*.jpg`.
- Thermal: `<split>/thermal/*.jpg`.
- Annotation chính: COCO JSON với các trường `images`, `annotations`, `categories`; bbox có dạng `[x, y, width, height]`.
- Annotation bổ sung: Pascal VOC XML với bbox dạng `xmin, ymin, xmax, ymax`.

### Insight và đánh giá

Cấu trúc 80/20 phù hợp cho huấn luyện và validation. Dữ liệu ghép cặp đầy đủ giúp xây dựng mô hình fusion RGB–thermal mà không phải xử lý missing modality. COCO JSON nên được chọn làm nguồn annotation chính; XML phù hợp cho việc đối chiếu và kiểm tra dữ liệu.

---

## 2. Data quality check

### 2.1. Kết quả kiểm tra file

| Hạng mục | Số lượng |
|---|---:|
| File ảnh dự kiến | 12.250 |
| Thiếu file ảnh | 0 |
| Ảnh không đọc được/corrupt | 0 |
| File ảnh có kích thước khác metadata COCO | 6 |
| Cặp visible–thermal khác kích thước nhau | 0 |
| Thiếu XML label | 0 |
| File ảnh dư ngoài COCO | 0 |
| XML dư ngoài COCO | 0 |
| Sample COCO không có object | 7 |

Pillow đã được dùng để mở và `verify()` toàn bộ 12.250 ảnh. Không phát hiện ảnh corrupt hoặc không đọc được.

### 2.2. Sai kích thước metadata

Ba sample train dưới đây có metadata COCO là `630×500`, trong khi kích thước file visible và thermal thực tế đều là `640×512`:

| Image ID | File | Metadata COCO | Kích thước thật | Số file bị ảnh hưởng |
|---:|---|---|---|---:|
| 65 | `00065.jpg` | 630×500 | 640×512 | 2 |
| 67 | `00067.jpg` | 630×500 | 640×512 | 2 |
| 76 | `00076.jpg` | 630×500 | 640×512 | 2 |

### 2.3. Bounding box không hợp lệ

| Split | Annotation ID | Image ID | Class | Bbox `[x,y,w,h]` | Lỗi |
|---|---:|---:|---|---|---|
| Train | 562 | 76 | `person` | `[629, 187, 0, 9]` | Width bằng 0 |

Các kiểm tra bbox còn lại:

| Kiểm tra | Số lỗi |
|---|---:|
| Giá trị không hữu hạn | 0 |
| Width/height không dương | 1 |
| Bbox vượt biên ảnh | 0 |
| Category không tồn tại | 0 |
| Trường `area` khác `width × height` | 0 |

### 2.4. Sample không có object

Cả 7 sample không có object đều thuộc tập train:

`01953.jpg`, `02002.jpg`, `03285.jpg`, `03714.jpg`, `03718.jpg`, `03721.jpg`, `03722.jpg`.

### Insight và đánh giá

Chất lượng file ảnh và tính đầy đủ của dataset nhìn chung tốt. Tuy nhiên, dataset **chưa nên đưa thẳng vào training** trước khi:

1. sửa metadata kích thước của ba sample thành `640×512`;
2. sửa hoặc loại annotation ID 562 có width bằng 0;
3. xác nhận 7 sample không object là negative samples có chủ đích. Nếu đúng, nên giữ lại vì ảnh âm giúp giảm false positive.

---

## 3. Image Resolution Analysis

Kích thước thực tế của tất cả ảnh:

| Split | Modality | Số ảnh | Width min/median/max | Height min/median/max | Aspect ratio mean ± std | Số resolution |
|---|---|---:|---|---|---|---:|
| Train | Visible | 4.900 | 640 / 640 / 640 | 512 / 512 / 512 | 1,250 ± 0,000 | 1 |
| Train | Thermal | 4.900 | 640 / 640 / 640 | 512 / 512 / 512 | 1,250 ± 0,000 | 1 |
| Validation | Visible | 1.225 | 640 / 640 / 640 | 512 / 512 / 512 | 1,250 ± 0,000 | 1 |
| Validation | Thermal | 1.225 | 640 / 640 / 640 | 512 / 512 / 512 | 1,250 ± 0,000 | 1 |

Histogram width, height và aspect ratio đều chỉ có một đỉnh tại:

- Width: **640 px**.
- Height: **512 px**.
- Aspect ratio: **1,25**.

### Insight và đánh giá

Toàn bộ file ảnh thực tế có cùng resolution và aspect ratio. Điều này đơn giản hóa batching và preprocessing. Nếu kiến trúc yêu cầu kích thước khác, nên resize đồng nhất hoặc dùng letterbox; cần áp dụng chính xác cùng một phép biến đổi hình học cho visible, thermal và bbox. Metadata lỗi của ba sample không phản ánh kích thước file thật.

---

## 4. Class distribution

| Class | Train | Validation | Tổng | Tỷ lệ |
|---|---:|---:|---:|---:|
| `person` | 43.169 | 11.009 | 54.178 | 76,436% |
| `crowd` | 7.172 | 2.004 | 9.176 | 12,946% |
| `rider` | 4.633 | 1.447 | 6.080 | 8,578% |
| `uncertain` | 1.340 | 106 | 1.446 | 2,040% |
| **Tổng** | **56.314** | **14.566** | **70.880** | **100%** |

Tỷ lệ giữa class nhiều nhất và ít nhất là **37,5×**.

### Insight và đánh giá

Dataset mất cân bằng mạnh về số object. Mô hình có nguy cơ thiên về `person` và đạt recall thấp trên `rider`, `crowd`, đặc biệt là `uncertain`. Nên cân nhắc:

- class-weight hoặc focal loss;
- sampler/crop có cân bằng class;
- báo cáo AP, precision và recall riêng cho từng class thay vì chỉ dùng mAP tổng;
- xem xét vai trò của `uncertain`: class huấn luyện độc lập, ignore region hay annotation cần loại tùy mục tiêu ứng dụng.

---

## 5. Bounding-box analysis

### 5.1. Small/medium/large theo chuẩn COCO

Ngưỡng sử dụng: small `<32²`, medium từ `32²` đến `<96²`, large `≥96²` pixel².

| Nhóm | Train | Validation | Tổng | Tỷ lệ |
|---|---:|---:|---:|---:|
| Small | 56.257 | 14.532 | 70.789 | 99,872% |
| Medium | 57 | 34 | 91 | 0,128% |
| Large | 0 | 0 | 0 | 0% |

### 5.2. Kích thước và hình dạng bbox

| Chỉ số | Mean | P25 | Median | P75 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Width (px) | 9,886 | 7 | 9 | 11 | 17 | 22 | 84 |
| Height (px) | 14,228 | 12 | 14 | 16 | 21 | 26 | 71 |
| Area (px²) | 147,433 | 91 | 126 | 176 | 300 | 483 | 4.347 |
| Aspect ratio bbox (w/h) | 0,717 | 0,538 | 0,667 | 0,824 | 1,214 | 1,667 | 6,444 |

- Bbox trung vị chỉ chiếm **0,038%** diện tích ảnh.
- Aspect ratio bbox trung vị là **0,67**, cho thấy phần lớn object cao hơn chiều rộng.

### 5.3. Số object trên mỗi ảnh

| Split | Mean | Median | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 11,493 | 7 | 16 | 28 | 38 | 57 | 80 |
| Validation | 11,891 | 7 | 13 | 24 | 42 | 87 | 98 |

### Insight và đánh giá

RGBTDronePerson là bài toán **small-object detection ở mức rất cao**. Downsample mạnh có thể làm đối tượng biến mất khỏi feature map. Các lựa chọn nên thử gồm:

- input resolution đủ lớn, FPN/PAN hoặc feature map stride nhỏ;
- multi-scale training và crop giữ object;
- augmentation không làm co hoặc méo người quá mạnh;
- kiểm tra anchor/assignment phù hợp với bbox khoảng 9×14 px;
- cấu hình `max_detections` đủ lớn và điều chỉnh NMS cho các ảnh đông người.

Ngưỡng COCO là ngưỡng tuyệt đối theo pixel; do đó nên báo cáo thêm AP theo `area_ratio` để đánh giá ổn định khi thay đổi resolution đầu vào.

---

## 6. Spatial distribution

Tâm bbox được chuẩn hóa về `[0,1]` và chia ảnh thành lưới 3×3. Center là ô giữa; tám ô còn lại được xem là edge.

### 6.1. Center và edge

| Vùng | Train | Validation | Tổng | Tỷ lệ |
|---|---:|---:|---:|---:|
| Center | 10.452 | 3.502 | 13.954 | 19,687% |
| Edge | 45.862 | 11.064 | 56.926 | 80,313% |

Mặc dù số object ở edge lớn hơn, vùng edge chiếm 88,9% diện tích ảnh. Ô center chỉ chiếm 11,1% diện tích nhưng chứa 19,7% số object, cao hơn khoảng **1,77×** so với kỳ vọng của phân bố đều theo diện tích.

### 6.2. Phân bố theo lưới 3×3

|  | Left | Center | Right |
|---|---:|---:|---:|
| Top | 8,033% | 12,734% | 8,492% |
| Middle | 9,747% | **19,687%** | 11,161% |
| Bottom | 8,015% | 13,974% | 8,156% |

### Insight và đánh giá

Object có xu hướng tập trung trên trục dọc giữa ảnh, đặc biệt tại ô `middle–center`. Đây là dấu hiệu center bias có thể khiến mô hình kém ổn định khi người xuất hiện sát mép. Nên dùng random translation/crop có kiểm soát, đồng thời đánh giá AP hoặc recall riêng cho center và edge. Khi crop cần bảo toàn bbox nhỏ, tránh tạo quá nhiều box bị cắt cụt.

---

## 7. Pixel/intensity analysis

Phân tích sử dụng **1.000 ảnh**: lấy ngẫu nhiên 250 ảnh cho mỗi nhóm train-visible, train-thermal, validation-visible và validation-thermal, với seed cố định. Ảnh được resize về `128×128` chỉ cho mục đích thống kê pixel.

- Brightness là mean luminance `0.2126R + 0.7152G + 0.0722B`.
- Contrast là độ lệch chuẩn luminance trong từng ảnh.

### 7.1. Thống kê theo split và modality

| Split | Modality | Mean R | Mean G | Mean B | Std R | Std G | Std B | Brightness | Contrast |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | Visible | 119,539 | 119,086 | 112,828 | 36,783 | 36,352 | 37,374 | 118,730 | 36,158 |
| Train | Thermal | 95,460 | 95,460 | 95,460 | 42,627 | 42,627 | 42,627 | 95,460 | 42,627 |
| Validation | Visible | 128,936 | 128,735 | 118,512 | 35,034 | 31,797 | 35,084 | 128,039 | 32,245 |
| Validation | Thermal | 91,760 | 91,760 | 91,760 | 48,213 | 48,213 | 48,213 | 91,760 | 48,213 |

### 7.2. Giá trị tổng hợp theo modality

| Modality | RGB mean | Brightness trung bình | Contrast trung bình | Nhận xét |
|---|---|---:|---:|---|
| Visible | (124,2; 123,9; 115,7) | 123,4 | 34,2 | Có khác biệt giữa các kênh màu |
| Thermal | (93,6; 93,6; 93,6) | 93,6 | 45,4 | Ba kênh giống nhau, thực chất grayscale |

Histogram thermal của ba kênh R/G/B trùng nhau. Histogram visible có chênh lệch kênh, trong đó B thấp hơn R và G. Thermal tối hơn visible theo brightness trung bình nhưng có contrast nội ảnh cao hơn.

Ngoài ra có khác biệt giữa train và validation:

- Visible validation sáng hơn train khoảng **9,3 intensity level** và contrast thấp hơn khoảng **3,9**.
- Thermal validation tối hơn train khoảng **3,7 intensity level** và contrast cao hơn khoảng **5,6**.

### Insight và đánh giá

Visible và thermal không nên dùng chung thống kê normalization. Khuyến nghị:

- tính mean/std **chỉ từ train** và riêng cho từng modality để tránh data leakage;
- không áp trực tiếp ImageNet normalization cho thermal nếu chưa kiểm chứng;
- coi thermal là một kênh hoặc giữ ba kênh lặp tùy yêu cầu backbone pretrained;
- cấu hình brightness/contrast augmentation riêng cho từng modality;
- theo dõi domain shift brightness/contrast giữa train và validation.

---

## Kết luận và hành động đề xuất

### Ưu tiên xử lý dữ liệu

1. **Bắt buộc:** sửa hoặc loại bbox annotation ID 562 có width bằng 0.
2. **Bắt buộc:** sửa metadata kích thước cho image ID 65, 67 và 76 từ `630×500` thành `640×512`.
3. **Xác minh:** quyết định giữ 7 ảnh không object như negative samples.
4. **Kiểm thử lại:** chạy data quality check sau khi sửa và bảo đảm visible–thermal–bbox vẫn đồng bộ.

### Khuyến nghị modeling và evaluation

- Thiết kế pipeline cho small object: resolution cao, multi-scale feature và hạn chế downsampling quá sớm.
- Dùng phép biến đổi hình học đồng bộ cho cả RGB, thermal và bbox.
- Xử lý class imbalance bằng loss/sampling phù hợp.
- Normalize riêng visible và thermal dựa trên train.
- Báo cáo metric theo class, kích thước bbox và vùng center/edge.
- Kiểm tra `max_detections` và NMS trên các ảnh có tới 80–98 object.

Nhìn chung, dataset có tính đầy đủ và tính nhất quán file tốt, nhưng bài toán khó do đối tượng cực nhỏ, mất cân bằng class, mật độ object cao và khác biệt phân bố intensity giữa hai modality.
