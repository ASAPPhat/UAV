# Báo cáo Exploratory Data Analysis — VTUAV-det

**Ngày phân tích:** 16/09/2026  
**Nguồn dữ liệu:** `VTUAV-det/`  
**Notebook tái lập:** [`VTUAV-det_EDA.ipynb`](./VTUAV-det_EDA.ipynb)

## Tóm tắt kết quả

VTUAV-det có **16,770 sample ghép cặp RGB–IR**, **33,540 ảnh kỳ vọng**, **124,869 object** và **4 class**. Train/validation chiếm 67.9%/32.1%. Annotation chính là COCO JSON; `val_ir.json` ánh xạ tới thư mục vật lý `test/`.

Các điểm đáng chú ý:

- Phát hiện **0 ảnh thiếu**, **0 ảnh corrupt**, **0 ảnh lệch metadata**, **0 cặp lệch kích thước**.
- COCO có **20 bbox sai hình học** và **40 ảnh không có object**.
- XML phụ thiếu **33 file**; **31** trường hợp là ảnh COCO không có object và **2** trường hợp có object. Có **3 sidecar không phải XML**.
- `person` chiếm **97.0%** object; mất cân bằng lớn nhất/nhỏ nhất là **179.2:1**.
- **14.0%** bbox là small; median bbox chiếm **0.170%** diện tích ảnh.
- **25.7%** tâm object ở center, bằng **2.32×** mức kỳ vọng đồng đều.
- RGB có brightness/contrast **97.6/34.5**, IR **134.2/54.5** (ước lượng lấy mẫu, thang 0–255).

---

## 1. Dataset overview

| split   | physical_folder   |   paired_samples_COCO |   rgb_files |   ir_files |   xml_files |   objects_COCO |
|:--------|:------------------|----------------------:|------------:|-----------:|------------:|---------------:|
| train   | train             |                 11392 |       11392 |      11392 |       11392 |          76314 |
| val     | test              |                  5378 |        5378 |       5378 |        5345 |          48555 |

### Class và format

|   id | name      | supercategory   |
|-----:|:----------|:----------------|
|    0 | person    | none            |
|    1 | rider     | none            |
|    2 | crowd     | none            |
|    3 | uncertain | none            |

- Ảnh: JPEG trong `<split>/rgb/` và `<split>/ir/`.
- Annotation chính: COCO JSON, bbox `[x, y, width, height]`.
- Annotation phụ: Pascal VOC XML, bbox `xmin, ymin, xmax, ymax`.
- Split logic `val` nằm trong thư mục vật lý `test`.

### Insight và đánh giá

Dataset có 11,392 train và 5,378 val, cùng dùng 4 class. Cấu trúc cặp RGB–IR phù hợp cho mô hình fusion; cần giữ phép biến đổi hình học đồng bộ. Nên dùng COCO JSON làm nguồn nhãn chuẩn để tránh phụ thuộc độ đầy đủ của XML.

---

## 2. Data quality check

| Hạng mục                          |   Số lượng |
|:----------------------------------|-----------:|
| Ảnh kỳ vọng                       |      33540 |
| Ảnh thiếu                         |          0 |
| Ảnh corrupt/unreadable            |          0 |
| Ảnh lệch metadata                 |          0 |
| Cặp RGB–IR lệch kích thước        |          0 |
| XML thiếu                         |         33 |
| COCO bbox sai hình học            |         20 |
| Ảnh COCO không có object          |         40 |
| Sidecar annotation không phải XML |          3 |

### Chi tiết annotation COCO

|                                      |   count |
|:-------------------------------------|--------:|
| duplicate_image_id_within_split      |       0 |
| duplicate_annotation_id_within_split |       0 |
| missing_image_reference              |       0 |
| unknown_category                     |       0 |
| non_finite_bbox                      |       0 |
| non_positive_bbox                    |      20 |
| bbox_outside_image                   |       0 |
| wrong_bbox_total                     |      20 |
| declared_area_mismatch               |       0 |
| images_without_objects               |      40 |

### Insight và đánh giá

Trên 33,540 ảnh kỳ vọng có 0 ảnh thiếu và 0 ảnh corrupt. COCO có 20 bbox sai hình học; ảnh không có object (40) cần được xem là negative sample cho đến khi có xác nhận khác. XML thiếu 33 file; 31 ảnh tương ứng là negative theo COCO và 2 ảnh có object. Vì vậy, trình đọc XML cần quy ước rõ file vắng mặt có phải nhãn rỗng hay lỗi dữ liệu. Các sidecar không phải XML: `03726.txt, 05850.json, classes.txt`.

---

## 3. Image Resolution Analysis

Top cấu hình độ phân giải:

| modality   | resolution   |   count |   percent_within_modality |
|:-----------|:-------------|--------:|--------------------------:|
| ir         | 1920×1080    |   16770 |                    100.00 |
| rgb        | 1920×1080    |   16770 |                    100.00 |

### Insight và đánh giá

Độ phân giải phổ biến nhất là **1920×1080** (100.0%); có 1 cấu hình, aspect ratio 1.778–1.778. Dù cặp modality có thể đồng kích thước, pipeline vẫn nên resize/letterbox nhất quán và biến đổi đồng bộ với bbox.

---

## 4. Class distribution

| class     |     train |       val |      total |   percent |
|:----------|----------:|----------:|-----------:|----------:|
| person    | 74,010.00 | 47,114.00 | 121,124.00 |     97.00 |
| rider     |  1,316.00 |    898.00 |   2,214.00 |      1.77 |
| crowd     |    426.00 |    429.00 |     855.00 |      0.68 |
| uncertain |    562.00 |    114.00 |     676.00 |      0.54 |

### Insight và đánh giá

`person` là class lớn nhất (97.0%), `uncertain` nhỏ nhất (0.5%), tỷ lệ 179.2:1. Chênh tỷ trọng train–val lớn nhất là `uncertain` (0.50 điểm %). Nên theo dõi AP từng class và cân nhắc weighting/sampling.

---

## 5. Bounding box

| size   |     train |       val |     total |   percent |
|:-------|----------:|----------:|----------:|----------:|
| small  |  7,650.00 |  9,862.00 | 17,512.00 |     14.03 |
| medium | 50,795.00 | 32,733.00 | 83,528.00 |     66.90 |
| large  | 17,867.00 |  5,942.00 | 23,809.00 |     19.07 |

Thống kê object/ảnh:

| split   |     count |   mean |   std |   min |   25% |   50% |   75% |   90% |   95% |   99% |    max |
|:--------|----------:|-------:|------:|------:|------:|------:|------:|------:|------:|------:|-------:|
| train   | 11,392.00 |   6.70 | 12.40 |  0.00 |  1.00 |  3.00 |  7.00 | 15.00 | 24.00 | 81.00 | 103.00 |
| val     |  5,378.00 |   9.03 | 10.83 |  0.00 |  2.00 |  4.00 | 14.00 | 23.00 | 30.00 | 52.00 |  74.00 |

### Insight và đánh giá

Sau khi loại 20 bbox sai hình học, small chiếm **14.0%** bbox hợp lệ; median diện tích tương đối là **0.170%**. Median aspect ratio là 0.44; bbox cực đoan chiếm 1.26%. Median object/ảnh là 3, p95 27. Nên ưu tiên feature độ phân giải cao/multi-scale và kiểm tra giới hạn max detections.

---

## 6. Spatial distribution

| zone   |     count |   percent |
|:-------|----------:|----------:|
| edge   | 92,711.00 |     74.26 |
| center | 32,138.00 |     25.74 |

Lưới 3×3 (%):

| cy_norm   |   left |   center |   right |
|:----------|-------:|---------:|--------:|
| top       |   7.04 |    17.66 |    8.81 |
| middle    |  10.15 |    25.74 |   10.07 |
| bottom    |   5.07 |     9.41 |    6.04 |

### Insight và đánh giá

Center chứa **25.7%** object, bằng 2.32× mức kỳ vọng đều. Ô dày nhất là middle–center (25.7%). Nên dùng crop/translation hợp lệ và đánh giá riêng center/edge để kiểm tra độ bền vị trí.

---

## 7. Pixel / intensity

Thống kê lấy mẫu (tối đa 200 ảnh / split / modality, resize 128×128):

|                  |   sampled_images |   brightness_mean |   brightness_std_between_images |   contrast_mean |   mean_R |   mean_G |   mean_B |   std_R |   std_G |   std_B |   channel_delta |
|:-----------------|-----------------:|------------------:|--------------------------------:|----------------:|---------:|---------:|---------:|--------:|--------:|--------:|----------------:|
| ('train', 'ir')  |           200.00 |            142.26 |                           58.84 |           50.67 |   142.26 |   142.26 |   142.26 |   50.67 |   50.67 |   50.67 |            0.00 |
| ('train', 'rgb') |           200.00 |             92.69 |                           40.29 |           32.63 |    97.00 |    92.36 |    83.27 |   34.95 |   32.76 |   30.35 |           23.73 |
| ('val', 'ir')    |           200.00 |            126.12 |                           53.92 |           58.25 |   126.12 |   126.12 |   126.12 |   58.25 |   58.25 |   58.25 |            0.00 |
| ('val', 'rgb')   |           200.00 |            102.60 |                           35.22 |           36.35 |   107.81 |   101.80 |    95.21 |   38.62 |   36.49 |   35.11 |           22.01 |

### Insight và đánh giá

RGB có brightness/contrast trung bình **97.6/34.5**, IR **134.2/54.5**. IR gần như grayscale lưu 3 kênh. Nên normalize từng modality riêng; color augmentation cho RGB và IR không nên dùng chung một cách máy móc.

---

## Khuyến nghị cho modeling

1. Dùng COCO JSON làm annotation chuẩn; nếu framework yêu cầu XML, quy ước rõ ảnh negative có XML rỗng hay không có file và chuyển đổi nhất quán từ JSON.
2. Xử lý/loại bbox sai sau khi kiểm tra trực quan; giữ ảnh không object như negative sample nếu đúng chủ đích.
3. Áp dụng augmentation hình học đồng bộ cho RGB–IR–bbox; normalize hai modality riêng.
4. Ưu tiên kiến trúc và kích thước input phù hợp small object; cân nhắc multi-scale training/tiling.
5. Báo cáo AP theo class, APs/APm/APl và metric center/edge; giám sát class imbalance và positional bias.

## Tính tái lập

- Kiểm tra corrupt/kích thước: toàn bộ ảnh bằng Pillow `verify()`.
- Pixel statistics: seed `42`, tối đa `200` ảnh mỗi split × modality, resize `128×128`.
- Toàn bộ bảng và biểu đồ chi tiết nằm trong notebook cùng tên.
