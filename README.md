# ─────────────────────────────────────────────────────────────────
#  AIC 2026 Baseline — README
# ─────────────────────────────────────────────────────────────────

# AIC 2026 Vòng Sơ Tuyển — Video Retrieval Baseline

Pipeline hoàn chỉnh cho 3 loại truy vấn của AIC 2026:
**Textual KIS** · **Q&A (VQA)** · **TRAKE (Temporal)**

---

## 📁 Cấu trúc thư mục

```
baseline/
├── config.yaml                    # ← CẤU HÌNH CHÍNH (chỉnh đường dẫn ở đây)
├── requirements.txt
├── run_search.py                  # CLI entry point
│
├── scripts/
│   ├── 01_extract_keyframes.py   # Bước 1: Trích xuất frame + build frame_map.json
│   ├── 02_build_clip_index.py    # Bước 2: Build FAISS index từ CLIP features
│   └── 03_build_bm25_index.py    # Bước 3: Build BM25 corpus từ metadata
│
├── src/
│   ├── retrieval/
│   │   ├── clip_retriever.py     # CLIP text→image retrieval (FAISS)
│   │   ├── bm25_retriever.py     # BM25 keyword search trên metadata
│   │   └── hybrid_retriever.py   # Hybrid fusion (RRF) + object boost
│   ├── query/
│   │   ├── textual_kis.py        # Textual KIS searcher
│   │   ├── qa_vqa.py             # Q&A VQA searcher (BLIP-2)
│   │   └── trake.py              # TRAKE temporal searcher (DP)
│   ├── scoring.py                # R-Score + Final Score (theo đặc tả BTC)
│   └── submission.py             # Export JSON/CSV/TXT
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

# CLIP (chọn 1 trong 2):
pip install git+https://github.com/openai/CLIP.git
# hoặc:
pip install open_clip_torch
```

### 2. Chỉnh config.yaml

Mở `config.yaml` và chỉnh đường dẫn dataset:

```yaml
data:
  root: "f:/aic/Dataset"     # ← đường dẫn đến thư mục dataset
```

### 3. Build indexes (chạy 1 lần)

```bash
# Bước 1: Build frame_map.json từ thư mục Keyframes/
python scripts/01_extract_keyframes.py --config config.yaml

# Bước 2: Build FAISS index từ clip_features.npy (đã có sẵn)
python scripts/02_build_clip_index.py --config config.yaml

# Bước 3: Build BM25 corpus từ metadata
python scripts/03_build_bm25_index.py --config config.yaml
```

> **Lưu ý**: Nếu dataset chưa có `clip_features.npy`, thêm `--from-images`:
> ```bash
> python scripts/02_build_clip_index.py --config config.yaml --from-images
> ```

### 4. Chạy search

```bash
# Textual KIS (single query)
python run_search.py --query "một người đang mở laptop" --type kis

# Q&A
python run_search.py \
  --query "cảnh bữa tiệc" \
  --question "Người phụ nữ mặc váy màu gì?" \
  --type qa

# TRAKE
python run_search.py \
  --type trake \
  --events "vận động viên giậm nhảy" "bay qua xà" "tiếp đất"

# Batch queries từ file
python run_search.py \
  --query-file data/sample_queries.json \
  --output submissions/ \
  --evaluate
```

### 5. Khởi động REST API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/health` | Health check |
| GET | `/info` | Thông tin model/index |
| POST | `/search/kis` | Textual KIS search |
| POST | `/search/qa` | Q&A / VQA search |
| POST | `/search/trake` | TRAKE temporal search |
| POST | `/submit` | Batch submission |

### Ví dụ gọi API

```python
import requests

# Textual KIS
resp = requests.post("http://localhost:8000/search/kis", json={
    "query": "một người đang mở laptop trong phòng họp",
    "top_k": 100
})
print(resp.json()["answers"][:5])

# TRAKE
resp = requests.post("http://localhost:8000/search/trake", json={
    "events": ["giậm nhảy", "bay qua xà", "tiếp đất"],
    "top_k": 100
})
```

---

## 📊 Cách tính điểm (theo đặc tả BTC)

### R-Score

| Loại | Điều kiện R-Score = 1 |
|------|----------------------|
| **KIS** | `video_id` đúng + `frame_id ∈ [s, e]` |
| **Q&A** | `video_id` đúng + `frame_id ∈ [s, e]` + `answer` khớp |
| **TRAKE** | `video_id` đúng + tỉ lệ frame đúng / N |

### Final Score

```
Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5
```

**Chiến lược**: Xếp câu trả lời **tốt nhất lên đầu** để tối đa R@1!

---

## 🚀 Cải tiến so với baseline này

| Cải tiến | Mô tả |
|---------|-------|
| **Query Expansion** | Dùng LLM (Gemini/GPT) để paraphrase query |
| **CLIP Model mạnh hơn** | ViT-L/14, ViT-G/14 thay vì ViT-B/32 |
| **Re-ranking** | Cross-encoder (CLIP4Clip, BLIP-2 ITM) |
| **OCR Search** | Tìm kiếm text xuất hiện trong video |
| **ASR Search** | Tìm kiếm theo lời thoại (speech-to-text) |
| **Ensemble** | Kết hợp nhiều model CLIP khác nhau |

---

## 📝 Format Nộp Bài

| Loại | Format |
|------|--------|
| Textual KIS | `video_id, frame_id` |
| Q&A | `video_id, frame_id, answer` |
| TRAKE | `video_id, frame_id_1, ..., frame_id_N` |

- Mỗi query: tối đa **100 câu trả lời**
- Sắp xếp theo **độ tin cậy giảm dần**
