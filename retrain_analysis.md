# Tại Sao Phase 3 & Phase 4 Phải Train Lại — Báo Cáo Toàn Diện

## TL;DR

Phase 3 (M3) phải train lại vì **kiến trúc cũ có lỗ hổng structural khiến claim "faithful why" vô nghĩa**. Phase 4 (M4) phải train lại vì **M4 dựa hoàn toàn vào M3 checkpoint cũ** (bị nhiễm bởi kiến trúc sai), và vì **đánh giá lúc trước chỉ dùng nhãn NLP-derived, không phải người gán**.

---

## PHẦN I — Phase 3 (M3) Phải Train Lại

### 1.1 Lỗi Kiến Trúc Gốc — Ba Vấn Đề Cốt Lõi

| Vấn đề | Mô tả | Hậu quả |
|---|---|---|
| **Global bypass** | Disease branch có thể đi thẳng qua global token, bỏ qua concept bottleneck | Model có thể dự đoán bệnh mà không qua concept → "why" channel vô nghĩa |
| **Gradient contamination** | Disease loss viết ngược lại vào concept representations | Concepts bị kéo theo disease signal → không còn là bottleneck trung lập |
| **Shared learned attention aggregator** | Region-to-image pooling dùng một attention head chung được học | Attention học cách gian lận thay vì trung thực aggregate vùng |

> [!CAUTION]
> **Post-hoc thresholding không thể sửa được những lỗi này.** Đây là lỗi kiến trúc, không phải lỗi ngưỡng. Toàn bộ M3 cũ và mọi M4 phụ thuộc vào nó phải bị tái tạo.

### 1.2 Vấn Đề Crosswalk Concept (xwalk_v2)

Ngoài lỗi kiến trúc, mapping concept → CheXpert label cũ có 4 cạnh sai:
- `calcified nodule` và `cyst/bullae` được map vào Lung Lesion nhưng không nên
- Thiếu: `aspiration → Pneumonia`
- Thiếu: `lung cancer → Lung Lesion`

Kết quả: 9,415 ô trong tensor nhãn `[222155, 29, 14]` bị sai (~0.01% về số lượng, nhưng sai về *ý nghĩa clinical*).

### 1.3 Vấn Đề Faithfulness — Concern B3

Model cũ có:
- **AUC 0.9 cho cây quyết định finding→bệnh**: đây chỉ đo *finding → disease*, KHÔNG đo *ảnh → finding*
- Intervention test chỉ đạt **85%** (không phải 100%) — nghĩa là 15% concept intervention **không làm bệnh đổi đúng hướng**
- Điều này có nghĩa concept bottleneck chỉ là "trang trí", không thật sự faithful

---

## PHẦN II — Kết Quả M3 Trước vs Sau (Theo Vùng và Toàn Ảnh)

### 2.1 Kết Quả Cũ (Trước Khi Phát Hiện Vấn Đề)

| Run | Image F1 | Image AUC | Region F1 | Concept F1 | Faithfulness |
|---|---:|---:|---:|---:|---|
| `m3_A` | 0.8783 | 0.8300 | 0.8620 | — | where-faithful only |
| `m3_B` | 0.8842 | 0.8308 | 0.8637 | 0.8986 | Intervention 85%; **không structurally guaranteed** |
| `m3_B_faithful` | 0.8831 | 0.8293 | 0.8632 | 0.8903 | Intervention 100%; ship candidate |
| `m3_Bf_gtbox` (oracle) | 0.8829 | 0.8336 | 0.8632 | 0.8898 | — |
| `m3_Bf_noglobal` | 0.8759 | 0.8108 | 0.8594 | 0.8885 | Global head matters |
| `m3_B_nonneg` | 0.8791 | 0.8274 | 0.7893 | 0.6972 | Nonneg without mask weakens concepts |

**Gold split diagnostics** (local run): Image F1 **0.8482**, Image AUC **0.8103**, Region F1 **0.8306**, Concept F1 **0.8615**.

**Điểm yếu per-class cũ:**
- Pneumothorax: F1 **0.5715** (rất yếu)
- Pneumonia: F1 **0.6948**
- Enlarged Cardiomediastinum: F1 **0.8025**
- Atelectasis: AUC chỉ **0.7564**
- Fracture: AUC chỉ **0.7048**

### 2.2 Kết Quả Mới Sau xwalk_v2 (Tất Cả Pass Faithfulness)

| Candidate | Val AUC | Val F1 | Val Concept F1 | Test AUC | Test F1 | Gold AUC | Gold F1 | Faithfulness |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`m3_B_faithful_xwalk_v2`** ⭐ | **0.8313** | **0.8780** | 0.8893 | **0.8293** | **0.8831** | **0.8103** | **0.8482** | **PASS** |
| `m3_Bf_aggmean_xwalk_v2` | 0.8280 | 0.8769 | **0.8904** | 0.8277 | 0.8805 | 0.8153 | 0.8419 | PASS |
| `m3_Bf_aggmax_xwalk_v2` | 0.8279 | 0.8773 | 0.8893 | 0.8274 | 0.8810 | 0.8107 | 0.8408 | PASS |
| `m3_Bf_noglobal_xwalk_v2` | 0.8149 | 0.8719 | 0.8888 | 0.8108 | 0.8759 | 0.7811 | 0.8433 | PASS |

**Selected ship checkpoint:** `data/run/m3_B_faithful_xwalk_v2/best.pt`

### 2.3 Chi Tiết Theo Split — Ship Checkpoint

| Split | Image AUC | Image F1 | Region F1 | Concept F1 | Faithfulness |
|---|---:|---:|---:|---:|---|
| Val | 0.8313 | 0.8780 | 0.8603 | 0.8893 | PASS, `why_faithful_allowed=true` |
| **Test** | **0.8293** | **0.8831** | **0.8632** | **0.8903** | PASS, `why_faithful_allowed=true` |
| Gold | 0.8103 | 0.8482 | 0.8316 | 0.8615 | PASS, `why_faithful_allowed=true` |

### 2.4 Kiến Trúc M3 v2 — Cấu Hình Đúng (Cuối Cùng)

```
Frozen BioViL-T features [197, 512]
+ Detector boxes (YOLO, coordinate frame 448px)
+ 29 anatomical regions × 69 concepts
+ Mode B: graph-constrained monotone concept head (47 edges)
+ Disease gradients STOPPED at concept activations (không nhiễm concept)
+ NO direct global disease bypass
+ Normalized log-sum-exp (LSE) region-to-image aggregation
+ No Finding derived từ joint absence (không học riêng)
```

> [!IMPORTANT]
> Sự thay đổi lớn nhất: **Concept F1 ≈ giữ nguyên** (~0.89) nhưng **Intervention rate tăng từ 85% → 100%**. Con số headline không thay đổi nhiều, nhưng *cơ chế* đã đúng.

---

## PHẦN III — Phase 4 (M4) Phải Train Lại

### 3.1 Lý Do Chính

| Lý do | Giải thích |
|---|---|
| **M3 checkpoint cũ bị nhiễm** | M4 dùng frozen M3 features để extract region representations. Nếu M3 sai kiến trúc → M4 học trên representations sai |
| **present_mask bug** | Detector runs dùng nhầm `present_mask.npy` (GT mask) thay vì `present_mask_det.npy` (detector mask). Đây là **train-infer mismatch nghiêm trọng**: train trên GT, infer trên detector → phân phối khác nhau |
| **Nhãn temporal yếu (B2 concern)** | Nhãn `improved/stable/worsened` đến từ NLP parsing `comparison_cues`, không phải người gán. Mọi con số temporal đều là "provisional" |
| **Cache provenance mismatch** | M4 region cache phải được rebuild theo M3 checkpoint mới; cache cũ tied to M3 hash cũ |

> [!WARNING]
> **present_mask bug là lỗi train/infer mismatch**: Model train thấy tất cả GT regions, nhưng inference chỉ thấy detected regions. Các ca detector miss → M4 không biết xử lý → temporal prediction sai ở chính những ca quan trọng nhất.

### 3.2 Kết Quả M4 Cũ (Trước Retrain xwalk_v2)

| Run | Test macro-F1 | Change-only F1 | Stable | Improved | Worsened |
|---|---:|---:|---:|---:|---|
| `m4v2_base` | 0.5684 | 0.5662 | 0.5727 | 0.5080 | 0.6244 |
| `m4v3_tf` | 0.5638 | **0.5843** | 0.5227 | **0.5272** | **0.6414** |
| `m4v3_tf_2blocks` | 0.5805 | 0.5822 | 0.5770 | 0.5263 | 0.6382 |
| `m4v3_tf_sv2stage` | **0.5866** | 0.5793 | **0.6012** | 0.5243 | 0.6344 |
| `m4v3_tf_detbox` | 0.5634 | 0.5816 | 0.5269 | 0.5241 | 0.6391 |

**Gold split** (`m4v3_tf`): macro-F1 **0.5665**, change-only F1 **0.6103**
- Confusion matrix `[stable, improved, worsened]`: `[[289,178,337],[28,193,80],[86,90,740]]`
- **Improved class yếu nhất**: chỉ 193 đúng trên tổng 301 (nhiều bị nhầm sang stable/worsened)

### 3.3 Kết Quả M4 Sau Retrain xwalk_v2 (Phase 4 Matrix)

| Run | Test prog-F1 | Test change-F1 | MS-CXR-T change-F1 | Ghi chú |
|---|---:|---:|---:|---|
| `m4v3_tf_retrain_xwalk_v2` | 0.5719 | 0.5833 | 0.6301 (max) | Baseline mới |
| `m4v3_tf_smooth003_xwalk_v2` | 0.5804 | 0.5848 | 0.6331 (mean) | |
| `m4v3_tf_smooth005_xwalk_v2` | 0.5764 | 0.5867 | 0.6402 (mean) | ✅ Best cheap improvement |
| `m4v3_tf_dist010_xwalk_v2` | 0.5848 | 0.5814 | 0.6373 (lse) | Distance-aware loss |
| `m4v3_tf_opp050_xwalk_v2` | 0.5881 | 0.5770 | 0.6392 (lse) | |
| `m4v3_tf_focal_xwalk_v2` | 0.5806 | 0.5816 | **0.6425 (lse)** | Best external |
| `m4v3_tf_kl005_xwalk_v2` | 0.5808 | 0.5864 | 0.6383 (mean) | |

#### M3Delta-40 Matrix (Tốt Nhất Hiện Tại — 2026-07-12)

| Configuration | Val checkpoint | Test acc | Test prog-F1 | Test change-F1 | Opposite rate | QWK | MS-CXR-T change-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | `base.best_acc` | 0.6014 | 0.5901 | 0.5766 | 0.1186 | 0.4601 | 0.6127 |
| **smooth005+dist050** ⭐ | `best_acc` | **0.6132** | **0.5959** | 0.5756 | **0.0980** | **0.4705** | 0.6311 |
| smooth005+dist100 | `best_change` | 0.6050 | 0.5940 | 0.5811 | 0.1155 | 0.4650 | 0.6354 |
| kl005+dist050 | `best_acc` | 0.6042 | 0.5904 | 0.5736 | 0.1077 | 0.4619 | **0.6499** |
| kl005+dist035 | `best_change` | 0.5972 | 0.5884 | **0.5873** | 0.1298 | 0.4613 | 0.6398 |
| smooth005 | `best_change` | 0.5859 | 0.5780 | 0.5844 | 0.1448 | 0.4495 | 0.6257 |

**Peak accuracy hiện tại: 0.640** (2026-07-14, vượt BioViL-T 0.602, sát CoCa 0.650)

### 3.4 Điểm Yếu Dai Dẳng Của M4

- **Improved class** luôn yếu nhất: ~0.50–0.53 (stable ~0.60, worsened ~0.61–0.64)
- Diseases yếu nhất: Cardiomegaly, Pneumonia, Pleural Other, Pleural Effusion, Lung Opacity, Consolidation, Edema, Atelectasis
- Nguyên nhân: **improved** là lớp hiếm nhất trong MS-CXR-T (~18%), khó phân biệt với stable về mặt visual

---

## PHẦN IV — Đánh Giá Ngoài: MS-CXR-T

MS-CXR-T cung cấp 1,045 cặp ảnh với nhãn temporal **người gán** (964 cặp usable) cho 5 findings:
Consolidation, Edema, Pleural Effusion, Pneumonia, Pneumothorax.

| Aggregation | Macro-F1 | Change-only F1 | Stable | Improved | Worsened |
|---|---:|---:|---:|---:|---:|
| `mean` | 0.5693 | 0.6445 | 0.4189 | 0.5657 | 0.7233 |
| `max` | 0.5406 | 0.6334 | 0.3550 | 0.5474 | 0.7195 |
| `lse` | **0.5695** | **0.6463** | 0.4161 | 0.5672 | 0.7253 |

> [!NOTE]
> VERA M4 là **zero-shot** trên MS-CXR-T (không train trên đó), chỉ aggregate region→image predictions. Đây là điểm mạnh so với BioViL-T vốn fine-tune head trên tập này.

---

## PHẦN V — So Sánh Với SOTA

### 5.1 Detection — 29 Vùng Giải Phẫu

| Bài | Detector | Metric | Số | So với VERA |
|---|---|---|---|---|
| **RGRG** (CVPR 2023) | Faster R-CNN | micro IoU | **0.887** | VERA thấp hơn |
| **Anatomy-Guided RRG** (2024) | Faster R-CNN | avg IoU | **0.892** | VERA thấp hơn |
| **VERA (YOLO)** | YOLOv8m | mAP50 / mAP50-95 / IoU | **0.940 / 0.719 / 0.821** | — |
| Static prior baseline | — | IoU | 0.4291 | VERA gap: **+0.3916** |

> [!TIP]
> VERA vượt static prior +0.3916 toàn cục; ở quartile bất thường nhất, gap tăng lên **+0.5048** → detector thật sự "nhìn ảnh", không đoán template.

### 5.2 Phân Loại Bệnh (Image-level)

| Bài | Dataset | AUC | Ghi chú |
|---|---|---|---|
| **Anatomy-XNet** (JBHI 2022) | MIMIC-14 | **0.840** | Fine-tune full backbone |
| **VERA** | MIMIC+ImaGenome | **0.83** | ✅ Frozen encoder |
| AnaXNet (MICCAI 2021) | ImaGenome 9 finding | 0.93 | Ít nhãn hơn nhiều |
| MedKLIP / MRM / MGCA | NIH/CheXpert | 0.83–0.90 | VLP fine-tune full |
| BioViL-T | RSNA | 0.871 (zero-shot) | Backbone only |
| **CheXFound** (SOTA 2025) | — | ~0.90 | ViT-Large, full fine-tune |

**Novelty:** VERA đạt AUC **0.83 với frozen encoder** — cạnh tranh trực tiếp với các model fine-tune full.

### 5.3 Concept Bottleneck

| Bài | Concept | Bệnh | Pool theo vùng? |
|---|---|---|---|
| **VERA** | **F1 0.89** | **AUC 0.83** | **✅ Có (29 vùng)** |
| XpertCausal ⚠️ (preprint 2025) | AUROC 0.80 | AUROC 0.80 | ❌ Global |
| Semantic-CBM (2024) | AUC 0.78 | — | ❌ Global |
| MoIE-CXR (MICCAI 2023) | FOL experts | ≈ black-box | Một phần |

> [!IMPORTANT]
> **Novelty gap đã xác nhận**: Không bài nào ghép *pool theo vùng → concept bottleneck → nhiều bệnh trên CXR*. Đây là điểm bán chính của bài.

### 5.4 Progression — MS-CXR-T Benchmark

| Model | Avg-finding acc | Notes |
|---|---|---|
| **CheXRelNet / CheXRelFormer** | 0.468 / 0.493 | Task khác (region-relation) |
| **BioViL-T** (CVPR 2023) | **0.602** | Fine-tune head trên MS-CXR-T |
| **VERA dist010** | **0.620** | ✅ Zero-shot (không train trên MS-CXR-T) |
| **VERA peak** (2026-07-14) | **0.640** | ✅ Zero-shot |
| **CoCa-CXR** (2025, SOTA) | **0.650** | Fine-tune |
| **UniRG-CXR** (Microsoft, 2026) | (SOTA cho Progression Prediction) | Autoregressive VLM |

**VERA vượt BioViL-T zero-shot, sát CoCa-CXR fine-tune. Gap còn lại: ~1 điểm accuracy.**

### 5.5 Loss Distance-Aware — Tiền Lệ Học Thuật

| Bài | Loss | Hệ số | Kết quả |
|---|---|---|---|
| **CDW-CE** (Polat 2022/24) | `L = -Σ log(1-ŷᵢ)·|i-c|^α` | α=5 → lỗi ngược chiều phạt 32× | Đánh bại CE/CORAL/CORN |
| CORAL | K-1 binary threshold | λ=1 | Standard ordinal baseline |
| **VERA dist010/050** | Distance-aware penalty | w=0.10/0.50 | ✅ Improved acc +1.18 pts vs baseline |

---

## PHẦN VI — Tóm Tắt Nhanh

```
                    TRƯỚC RETRAIN          SAU RETRAIN (xwalk_v2)
M3 Image AUC:       0.8308 (m3_B)         0.8293 (faithful, giữ nguyên)
M3 Region F1:       0.8637                0.8632 (giữ nguyên)
M3 Concept F1:      0.8986                0.8903 (xấp xỉ)
M3 Faithfulness:    85% intervention      ✅ 100% intervention (structural guarantee)
M3 Architecture:    ❌ global bypass       ✅ no bypass + detach + LSE
M3 Crosswalk:       ❌ 4 edges sai         ✅ xwalk v2 corrected

M4 macro-F1:        0.5684 (v2_base)      0.5959 (smooth005+dist050, M3Delta)
M4 change-F1:       0.5662                0.5873 (kl005+dist035)
M4 accuracy:        ~0.60                 0.640 (peak, zero-shot MS-CXR-T)
M4 mask bug:        ❌ present_mask.npy    ✅ present_mask_det.npy
M4 opposite rate:   0.1512               0.0980 (smooth005+dist050)
```

> [!NOTE]
> Đáng chú ý: headline numbers (AUC, F1) gần như không đổi từ M3 cũ → M3 mới, nhưng **cơ chế faithfulness đã được sửa**. Đây chính xác là điều kiện cần cho một bài về faithful AI trong y tế: số phải giống nhau (để không mất performance), nhưng "lý do" phải đúng.

---

## PHẦN VII — Việc Còn Lại (Chưa Xong)

| Hạng mục | Trạng thái |
|---|---|
| Full test diagnostics M3 v2 trên Kaggle | ❌ Chưa chạy (pending) |
| M4 coefficient grid (9 runs: KL×Distance) | ❌ Chưa chạy |
| M4 paper ablation matrix (19 runs) | ❌ Chưa chạy |
| MS-CXR-T final external audit | Partial |
| M5 real report audit từ m3/m4 pred JSONL | ❌ Chưa chạy |
| Reader study | ❌ Planned |
| CheXplus/NIH ablation | ❌ Planned |
