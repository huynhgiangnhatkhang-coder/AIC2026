# ─────────────────────────────────────────────────────────────────
#  AIC 2026 Baseline — README
# ─────────────────────────────────────────────────────────────────

# AIC 2026 Vòng Sơ Tuyển — Video Retrieval Baseline

Pipeline hoàn chỉnh cho truy vấn của AIC 2026 với kiến trúc **Two-Stage Retrieval (SigLIP + Florence-2 + OCR)** dành riêng cho Textual KIS nhằm đạt độ chính xác tối đa.

---

## 📁 Cấu trúc thư mục

```text
baseline/
├── config.yaml                    # ← CẤU HÌNH CHÍNH (đường dẫn DB, Keyframes, NPY)
├── requirements.txt
├── run_search.py                  # CLI entry point chính
│
├── scripts/
│   ├── 01_extract_keyframes.py   # Bước 1: Trích xuất frame + build frame_map.json
│   ├── 02_build_clip_index.py    # Bước 2: Build FAISS index từ CLIP features (Legacy)
│   ├── 03_build_bm25_index.py    # Bước 3: Build BM25 corpus từ metadata (Legacy)
│   └── build_siglip_db.py        # NEW: Build Milvus DB với SigLIP Embeddings
│
├── src/
│   ├── retrieval/
│   │   ├── milvus_retriever.py   # Milvus vector search (SigLIP)
│   │   ├── clip_retriever.py     # FAISS search
│   │   └── bm25_retriever.py     # BM25 keyword search
│   ├── query/
│   │   ├── textual_kis.py        # FlorenceKISSearcher (Kiến trúc mới)
│   │   ├── qa_vqa.py             # Q&A VQA searcher
│   │   └── trake.py              # TRAKE temporal searcher
│   ├── scoring.py                # Tính điểm theo luật BTC
│   └── submission.py             # Xuất file nộp bài
│
├── app/
│   └── main.py                   # FastAPI REST API
│
└── data/
    └── sample_queries.json       # Ví dụ queries
```

---

## ⚡ Quickstart

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chỉnh config.yaml

Đảm bảo các đường dẫn trỏ đúng tới dữ liệu BTC cấp:

```yaml
data:
  root: "./DATASET"
  keyframes_root: "./DATASET"
index:
  npy_dirs: "./clip-features-32-aic25-b1/clip-features-32"
  keyframes_dirs_list:
    - "DATASET/Keyframes_L21/keyframes"
    - "DATASET/Keyframes_L22/keyframes"
    - "DATASET/Keyframes_L23/keyframes"
```

### 3. Khởi tạo Database (Chạy 1 lần)

Hệ thống KIS mới sử dụng Milvus Lite và mô hình SigLIP để đạt độ chính xác cao nhất.

```bash
# Build SigLIP Milvus DB từ thư mục Keyframes
python scripts/build_siglip_db.py --config config.yaml
```
*(Yêu cầu đã có sẵn file `ocr_database.json` trong thư mục gốc để hệ thống KIS quét chữ trên hình).*

### 4. Chạy Tìm Kiếm (Two-Stage OCR + SigLIP + Florence-2)

Hệ thống áp dụng cơ chế 2 kênh:
1. **Milvus HNSW Search:** Lấy 1000 ứng viên hình ảnh giống nhất bằng SigLIP.
2. **Independent OCR Search:** Quét toàn bộ `ocr_database.json` để vớt các frame chứa chữ trùng khớp (cực kỳ hiệu quả cho bản tin thời sự).
-> Hợp nhất ứng viên, lọc top 100 đưa cho Florence-2 phân tích chi tiết.

```bash
# Chạy KIS Search và BẬT tính năng hiển thị Top 10 ảnh kết quả
python run_search.py --query "bản tin về tai nạn giao thông tại dak lak" --type kis --show-images --show-k 10
```

*Ảnh kết quả sẽ được lưu thành `search_results_preview.jpg` và tự động mở lên màn hình.*

```bash
# Batch queries từ file JSON
python run_search.py \
  --query-file data/sample_queries.json \
  --output submissions/ \
  --evaluate
```

---

## 🔍 Kiến trúc thuật toán Textual KIS mới

1. **Dịch thuật Offline (Opus-MT):** Dịch câu truy vấn tiếng Việt sang tiếng Anh và trích xuất danh từ cốt lõi.
2. **Kênh 1 (Visual):** Nhúng câu truy vấn bằng SigLIP, tìm 1000 ứng viên gần nhất bằng Milvus HNSW.
3. **Kênh 2 (Text/OCR):** Chuẩn hóa Unicode, tìm độc lập các khung hình chứa từ khóa trực tiếp trong `ocr_database.json`.
4. **Pre-ranking:** Gộp kết quả 2 kênh, xếp hạng sơ bộ bằng `SigLIP Score + 0.5 * OCR Score`.
5. **Re-ranking (Florence-2):** Đưa Top 100 ảnh vào Florence-2 (Phrase Grounding) để đánh giá độ chính xác từng vật thể.
6. **OCR Override:** Tính tổng điểm. Nếu OCR score rất cao (>= 0.6), điểm số sẽ được gán mức trần (0.95+), đảm bảo các ảnh chứa text bản tin luôn nằm trên top 1 bất chấp các mô hình hình ảnh có nhận diện được hay không.

---

## 📝 Format Nộp Bài

| Loại | Format |
|------|--------|
| Textual KIS | `video_id, frame_id` |
| Q&A | `video_id, frame_id, answer` |
| TRAKE | `video_id, frame_id_1, ..., frame_id_N` |

- Mỗi query: tối đa **100 câu trả lời**
- Càng đúng ở Top 1, điểm R-Score tổng càng cao.
