# ─────────────────────────────────────────────────────────────────
#  AIC 2026 Baseline — README
# ─────────────────────────────────────────────────────────────────

# AIC 2026 Vòng Sơ Tuyển — Full-stack Video Retrieval Baseline

Pipeline hoàn chỉnh cho truy vấn của AIC 2026 với kiến trúc **Two-Stage Retrieval (SigLIP + Florence-2 + OCR)** dành riêng cho Textual KIS nhằm đạt độ chính xác tối đa. Hệ thống hiện tại đã bao gồm đầy đủ **Giao diện Web (Frontend)** và **API (Backend)**.

---

## 📁 Cấu trúc thư mục

```text
baseline/
├── AIChallenge2026-master/        # ← GIAO DIỆN WEB (React + Vite)
├── config.yaml                    # ← CẤU HÌNH CHÍNH (đường dẫn DB, Keyframes)
├── requirements.txt
├── run_search.py                  # CLI entry point chính
├── start.sh                       # Script chạy một chạm (cả Web + API)
│
├── src/
│   ├── retrieval/
│   │   ├── milvus_retriever.py   # Milvus vector search (SigLIP)
│   │   └── hybrid_retriever.py   # Hybrid (kết hợp vector + BM25 cũ)
│   ├── query/
│   │   ├── florence_kis.py       # FlorenceKISSearcher (Kiến trúc xịn nhất)
│   │   ├── qa_vqa.py             # Q&A VQA searcher (sử dụng Qwen2-VL)
│   │   └── trake.py              # TRAKE temporal searcher (Dynamic Programming)
│   ├── scoring.py                # Tính điểm theo luật BTC
│   └── submission.py             # Xuất file nộp bài
│
├── app/
│   └── main.py                   # FastAPI REST API Backend
│
└── data/
    └── sample_queries.json       # Ví dụ queries batch
```

---

## ⚡ Quickstart

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
pip install qwen-vl-utils accelerate # Cài thêm thư viện cho Q&A
```

Đảm bảo bạn đã cài đặt Node.js để chạy Frontend.

### 2. Chỉnh config.yaml

Đảm bảo các đường dẫn trỏ đúng tới dữ liệu BTC cấp:

```yaml
data:
  keyframes_root: "./DATASET"
index:
  milvus_db_path: "./aic_kis_database_siglip.db"
```

*(Yêu cầu đã có sẵn file `ocr_database.json` trong thư mục gốc để hệ thống KIS quét chữ trên hình).*

### 3. Khởi động Giao diện Web + API (Full-stack)

Hệ thống đã có sẵn script khởi động mọi thứ trong 1 lệnh:

```bash
bash start.sh
```

- **Frontend (UI)**: Truy cập `http://localhost:5173`
- **Backend (API)**: Chạy tại `http://localhost:8000`

---

## 🔍 Kiến trúc thuật toán

### 1. Textual KIS (Hai Kênh + Florence-2 Re-ranking)
Hệ thống áp dụng cơ chế 2 kênh:
1. **Dịch thuật Offline:** Dịch truy vấn tiếng Việt sang tiếng Anh, tách từ khóa/danh từ riêng.
2. **Kênh 1 (Milvus HNSW Search):** Lấy 1000 ứng viên hình ảnh giống nhất bằng SigLIP.
3. **Kênh 2 (Independent OCR Search):** Quét toàn bộ `ocr_database.json` để vớt các frame chứa chữ trùng khớp (cực kỳ hiệu quả cho bản tin thời sự).
4. **Hợp nhất (Pre-ranking):** Gộp kết quả 2 kênh.
5. **Re-ranking (Florence-2):** Đưa Top 100 ảnh vào Florence-2 (Phrase Grounding) để đánh giá độ chính xác từng vật thể.
6. **OCR Override:** Tính tổng điểm. Nếu điểm OCR rất cao, điểm số sẽ được gán mức trần (0.95+), đảm bảo các ảnh chứa text bản tin luôn nằm trên top 1.

### 2. VQA / Q&A (Qwen2-VL)
- Tìm frame phù hợp nhất bằng KIS Retrieval.
- Truyền frame đó vào VLM **Qwen2-VL-2B-Instruct** (mô hình thông minh xử lý tiếng Việt cực đỉnh) để trả lời câu hỏi trực quan (Ví dụ: "Xe ô tô màu gì?", "Biển số xe là bao nhiêu?").
- Có kết hợp confidence score để loại bỏ frame ảo.

### 3. TRAKE (Monotone Chain DP)
- Tìm kiếm từng chuỗi sự kiện bằng KIS Retrieval.
- Sử dụng **Quy hoạch động (Dynamic Programming)** O(N*K log K) để gióng hàng thời gian cực nhanh, chọn ra chuỗi sự kiện tăng dần chặt chẽ nhất.

---

## 📝 Format Nộp Bài

| Loại | Format |
|------|--------|
| Textual KIS | `video_id, frame_id` |
| Q&A | `video_id, frame_id, answer` |
| TRAKE | `video_id, frame_id_1, ..., frame_id_N` |

- Mỗi query: tối đa **100 câu trả lời**
- Chạy batch bằng lệnh: `python run_search.py --query-file data/sample_queries.json --output submissions/ --evaluate`

---

## 🌐 Chi tiết REST API

Chạy thủ công server API (nếu không dùng `start.sh`):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

| Endpoint | Mô tả |
|----------|-------|
| `GET /health` | Health check |
| `POST /search/kis` | Textual KIS — tìm theo mô tả văn bản |
| `POST /search/qa` | Q&A / VQA — tìm + trả lời câu hỏi |
| `POST /search/trake` | TRAKE — chuỗi sự kiện theo thời gian |
| `GET /frames/{video_id}/{frame_id}` | Trả về ảnh keyframe để hiển thị lên web |
| `POST /frames/batch` | Lấy nhiều ảnh cùng lúc dưới dạng base64 |
