# VERA Project — Handoff Document
**Last updated:** 2026-08-18  
**Mục đích:** Tóm tắt đầy đủ mọi thứ đã biết về dự án VERA để phiên chat tiếp theo phân tích kết quả server mà không cần đọc lại từ đầu.

---

## 1. VERA Là Gì — Một Đoạn

VERA (**Verifiable Evidence-grounded Regional Assembly**) là pipeline AI đọc ảnh X-quang ngực (CXR) với điểm bán chính: **báo cáo trung thực bằng thiết kế** (by construction), không phải hậu kiểm. Thay vì để LLM tự do viết báo cáo (dễ hallucinate), VERA lắp ráp báo cáo từ bảng dự đoán có cấu trúc:

- **Where-faithful:** mọi tuyên bố bệnh gắn với 29 vùng giải phẫu cụ thể
- **Why-faithful (M3):** bệnh được suy ra *qua* concept bottleneck → có thể giải thích "vì sao"
- **When-faithful (M4):** tiến triển chỉ được tuyên bố khi có ảnh prior thật của chính bệnh nhân

Pipeline: `M0 (tiền xử lý) → M1 (BioViL-T encoder, frozen) → M2 (YOLO 29 vùng) → M3 (bệnh + concept theo vùng) → M4 (tiến triển temporal) → M5 (lắp ráp báo cáo)`

Dataset chính: MIMIC-CXR + Chest ImaGenome (222,168 ảnh, 63,334 bệnh nhân).

---

## 2. Tại Sao M3 Và M4 Phải Train Lại

### 2.1 Ba Lỗi Kiến Trúc M3 Cũ

| Lỗi | Cơ chế | Hậu quả |
|---|---|---|
| **Global bypass** | Disease branch có đường thẳng qua global token, bỏ qua concept bottleneck | "Why" channel vô nghĩa — bệnh predict không cần concept |
| **Gradient contamination** | Disease loss lan ngược vào concept representations | Concept bị kéo theo disease signal → không phải bottleneck trung lập |
| **Shared learned attention aggregator** | Region→image pooling dùng attention head chung được học | Attention học cách "cheat" thay vì faithful aggregate |

Post-hoc thresholding **không thể sửa** các lỗi này — chúng là lỗi kiến trúc.

### 2.2 Lỗi Crosswalk (xwalk_v2)
4 cạnh trong graph concept→disease sai: `calcified nodule` và `cyst/bullae` không nên map vào Lung Lesion; thiếu `aspiration→Pneumonia` và `lung cancer→Lung Lesion`.

### 2.3 Tại Sao M4 Cũng Phải Train Lại
1. M4 dùng frozen M3 features → nếu M3 sai → M4 học trên representations sai
2. **`present_mask` bug**: train dùng `present_mask.npy` (GT boxes), inference dùng `present_mask_det.npy` (detector boxes) → train/infer mismatch nghiêm trọng

---

## 3. Kiến Trúc M3 v2 — Cấu Hình VERA Main

```
m3v2_vera_graph_lse_det:
  --mode B                   : concept bottleneck (region_feat → 69 concepts → 14 diseases)
  --head-type mlp            : MLP cho concept head
  --disease-head faithful    : masked non-negative (47 cạnh graph), softplus weights
                               → intervention PASS by construction
  --detach-concept           : disease loss không chảy ngược vào concept extractor
  --no-global-head           : bỏ global bypass hoàn toàn
  --region-agg lse           : log-sum-exp aggregation (không phải learned attention)
  --derive-no-finding        : No Finding = prod(1 - p_disease_i) cho các class khác
  --box-source detector      : dùng boxes_det.npy cho cả train và infer
```

**Code quan trọng:**
- `phase_3/src/model.py` — class `CKAN`, forward pass, `_derive_no_finding`, `_aggregate`
- `phase_3/src/heads.py` — `ConceptDiseaseHead` (faithful masked non-neg head)
- `phase_3/src/config.py` — tất cả toggles và defaults
- `phase_3/run_paper_m3_v2.sh` — định nghĩa chính xác flag cho từng run

---

## 4. Kết Quả — Trước vs Hiện Tại

### 4.1 Reference Cũ (m3_B_faithful_xwalk_v2) — Có Global Head, Attention Agg, No Derive
| Split | Image AUC | Image F1 | Region F1 | Concept F1 |
|---|---:|---:|---:|---:|
| Val | **0.8313** | 0.8780 | 0.8603 | 0.8893 |
| Test | **0.8293** | 0.8831 | 0.8632 | 0.8903 |

### 4.2 Server Run Hiện Tại (2026-08-18 08:25 UTC)
> Primary = image macro-AUC | Auxiliary = image macro-F1

| Run | Val AUC | Test AUC | Val F1 | Test F1 | Status |
|---|---:|---:|---:|---:|---|
| `m3v2_vera_graph_lse_det` ⭐ VERA main | 0.8039 | **0.7986** | 0.7812 | 0.7872 | complete |
| `m3v2_no_concept_det` | 0.8119 | **0.8136** | 0.7546 | 0.7647 | complete |
| `m3v2_concept_mlp_det` | 0.8071 | 0.8061 | 0.6969 | 0.7111 | complete |
| `m3v2_graph_global_fusion_det` | 0.8040 | 0.8075 | 0.7402 | 0.7495 | complete |
| `m3v2_global_only_det` | 0.8038 | 0.8070 | 0.7397 | 0.7525 | complete |
| `m3v2_graph_attention_det` | **0.8137** | 0.8046 | **0.6387** | **0.6444** | complete |
| `m3v2_graph_mean_det` | ~0.803 | - | ~0.784 | - | running 31/40 |
| `m3v2_graph_max_det` | - | - | - | - | pending |
| `m3v2_vera_graph_lse_gt` | - | - | - | - | pending (GT oracle) |

---

## 5. Phân Tích Chi Tiết Kết Quả Hiện Tại

### 5.1 Red Flag Chính
`m3v2_no_concept_det` (AUC **0.8136**) > `m3v2_vera_graph_lse_det` (AUC **0.7986**).

Mode A (no concept, direct feature→disease) beat VERA main. Đây là **expected** — mode A là accuracy ceiling. Điều đáng lo là gap lớn hơn dự kiến. Trong v1, gap gần như bằng 0 vì global head bù đắp.

### 5.2 Phát Hiện Mới Từ Attention Run
`m3v2_graph_attention_det`: Val AUC **0.8137** (cao nhất tất cả!) nhưng aux F1 chỉ **0.638/0.644**.

**Cách đọc aux metric:**
- Với mode B: aux = concept F1
- Với mode A / global-only: aux = disease F1 thông thường

Vậy attention run có AUC bệnh cao nhưng concept F1 cực thấp (0.64 vs 0.78 của LSE). Điều này gợi ý:
- Attention aggregation **học cách gian lận** qua shared weight → bệnh được predict tốt mà không cần concept tốt
- LSE (deterministic, không học) **buộc model phải học concept tốt** vì không có đường tắt
- Đây chính xác là concern ban đầu về "shared learned attention aggregator"
- **Hệ quả:** attention có AUC cao hơn nhưng về faithfulness, nó có thể fail intervention test

### 5.3 Phân Rã Gap AUC (0.829 → 0.799, -0.030)

Dựa trên kết quả đã có:

| Toggle | Delta AUC (ước tính từ data mới) |
|---|---|
| Bỏ global head | `global_fusion (0.8075)` vs `vera_main (0.7986)` = **-0.009** (nhỏ hơn -0.018 cũ) |
| LSE vs attention | `attention val (0.8137)` vs `vera_main val (0.8039)` = **-0.010** |
| derive-no-finding | **Chưa đo** (cần run_diag g2) |
| faithful head vs mlp | `concept_mlp (0.8061)` vs `vera_main (0.7986)` = **-0.008** |

Tổng cộng: -0.009 (global) + -0.010 (LSE) + -0.008 (faithful head) + ?? (derive) ≈ -0.027 + derive.

**Note về global head:** Gap nhỏ hơn v1 (0.009 vs 0.018 cũ) có thể vì derive-no-finding đang làm yếu global head effectiveness.

### 5.4 Bug Potential: `--derive-no-finding`

```python
# Từ model.py:
p_no_finding = (1 - sigmoid(logit_i)).prod(dim=1)  # product của 13 class
```

Vấn đề: khi model uncertain (logit ≈ 0, sigmoid ≈ 0.5), `p_no_finding ≈ 0.5^13 ≈ 0.0001`. No Finding bị underestimate cực kỳ khi model chưa confident. Điều này có thể:
1. Làm loss No Finding bị dominated bởi false negative (model thường đoán không có No Finding)
2. Ảnh hưởng gradient lên các disease class còn lại theo hướng ngược lại mong muốn

**Cần test:** `diag_g2_G0_D0_Alse` (VERA main không có derive) để đo delta.

---

## 6. Bộ Test run_diag.sh — 21 Runs

Script: `server_hoang/run_diag.sh`  
Sẽ tự chạy sau khi `run_all.sh` xong (via `bash server_hoang/run_diag.sh after`).

### Factorial 2^3 (Group g2)

| Cell | Global Head | Derive NF | Aggregation | Câu hỏi |
|---|---|---|---|---|
| `g2_G1_D0_Aat` | ON | OFF | attention | Old anchor — reproduce g1? |
| `g2_G1_D0_Alse` | ON | OFF | lse | Isolate agg only |
| `g2_G1_D1_Aat` | ON | ON | attention | Isolate derive only |
| `g2_G1_D1_Alse` | ON | ON | lse | All except no-global |
| `g2_G0_D0_Aat` | OFF | OFF | attention | Isolate global only |
| `g2_G0_D0_Alse` | OFF | OFF | lse | **KEY: VERA - derive** |
| `g2_G0_D1_Aat` | OFF | ON | attention | Global off + derive |
| `g2_G0_D1_Alse` | OFF | ON | lse | New v2 anchor (phải ≈0.799) |

### Cách Đọc Kết Quả

```
Step 1: diag_g1_old_repro
  ≈0.829  → env/data OK, gap hoàn toàn từ kiến trúc
  <0.810  → DATA BUG (boxes_det.npy hoặc env khác nhau) — debug ngay

Step 2: tính delta từng toggle từ g2 cells
  delta_global = G1D0Aat - G0D0Aat
  delta_derive = G1D0Aat - G1D1Aat
  delta_lse    = G1D0Aat - G1D0Alse

Step 3: faithfulness audit của attention runs
  Nếu m3v2_graph_attention_det FAIL faithfulness → LSE là bắt buộc
  Nếu PASS nhưng concept F1 thấp → viết vào paper như ablation

Step 4: quyết định config paper cuối
  Mục tiêu: maximize AUC trong khi PASS faithfulness 100%
```

---

## 7. SOTA Comparison

### Disease Classification (Image AUC)
| Model | AUC | Dataset | Encoder |
|---|---|---|---|
| CheXFound 2025 | ~0.90 | — | ViT-Large fine-tune |
| BioViL-T | 0.871 | RSNA | fine-tune |
| Anatomy-XNet | 0.840 | MIMIC-14 | fine-tune |
| **VERA v1** | **0.829** | MIMIC+ImaGenome | **frozen** |
| **VERA v2 main** | **0.799** | MIMIC+ImaGenome | **frozen** |
| **VERA v2 attention** | **0.804-0.814** | MIMIC+ImaGenome | **frozen** |

### Temporal Progression (MS-CXR-T avg accuracy)
| Model | Accuracy | Zero-shot? |
|---|---|---|
| BioViL-T | 0.602 | ❌ fine-tune trên MS-CXR-T |
| **VERA M4** | **0.640** | ✅ zero-shot |
| CoCa-CXR 2025 | 0.650 | ❌ fine-tune |

---

## 8. Files Quan Trọng

| File | Nội dung |
|---|---|
| `docs/now/flow.md` | Kiến trúc đầy đủ + rationale từng quyết định |
| `docs/now/VERA_methodology_concerns.md` | 6 concerns B1-B6, lý do chi tiết phải retrain |
| `docs/now/m3_retrain_server_runbook.md` | Protocol chính thức cho server run |
| `docs/now/VERA_sota_comparison.md` | So sánh SOTA chi tiết |
| `docs/now/more_papers.md` | Survey sâu về các bài SOTA |
| `phase_3/src/model.py` | M3 model (`CKAN` class) |
| `phase_3/src/config.py` | Tất cả config flags và defaults |
| `phase_3/src/heads.py` | `ConceptDiseaseHead` |
| `phase_3/run_paper_m3_v2.sh` | Exact flags cho paper matrix |
| `server_hoang/run_all.sh` | Main pipeline supervisor |
| `server_hoang/run_diag.sh` | Diagnostic battery (21 runs, 2^3 factorial) |

---

## 9. Tóm Tắt Cho Phiên Chat Tiếp

**Tình hình ngay bây giờ:**
- `run_all.sh` đang chạy M3 paper matrix (6/9 xong, tiếp theo là M4)
- `run_diag.sh` sẽ tự chạy sau khi `run_all.sh` xong

**Kết quả đã có (M3):**
- VERA main (LSE, no global): AUC **0.799**
- Attention thay LSE: AUC **0.804-0.814** nhưng concept F1 chỉ 0.64 (nghi cheat)
- Global head back: AUC **0.808**
- No concept (mode A): AUC **0.814** (accuracy ceiling)

**Khi run_diag.sh xong, việc cần làm:**
1. Check `diag_g1_old_repro` để biết env/data có gây ra drop không
2. Tính delta từng toggle từ factorial g2
3. Xem faithfulness audit của attention runs
4. Quyết định config paper final
5. Nếu config M3 thay đổi → rebuild M4 cache và retrain M4

**Câu hỏi cốt lõi còn mở:**
- `--derive-no-finding` có gây ra drop không? (cần g2_G0_D0_Alse vs g2_G0_D1_Alse)
- Attention aggregation có fail faithfulness không? (cần faithfulness audit)
- Config paper final là gì: LSE + no global (faithful, AUC 0.799) hay attention + no global (AUC 0.814, cần kiểm tra faithfulness)?
