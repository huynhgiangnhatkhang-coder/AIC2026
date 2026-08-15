# 🎯 Hướng dẫn chạy AIC 2026 Baseline (CLIP + Grounding DINO)

> Pipeline tìm kiếm video bằng **CLIP** (lọc thô) + **Grounding DINO** (re-rank) + dịch tiếng Việt **offline** (Opus-MT).

---

## 📋 Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|------------|-------------------|
| Python | 3.9+ |
| GPU VRAM | ≥ 8GB (chạy cả CLIP + DINO cùng lúc) |
| RAM | ≥ 16GB |
| CUDA | 11.8+ (hoặc chạy CPU — rất chậm) |

---

## 📁 Cấu trúc thư mục cần có

Trước khi chạy, đảm bảo thư mục `f:/aic/AIC2026/` có đúng cấu trúc sau:

```
f:/aic/
├── AIC2026/
│   ├── aic_kis_database.db/         ← Database Milvus (đã build sẵn từ AIC2026)
│   ├── clip-features-32-aic25-b1/
│   │   └── clip-features-32/
│   │       ├── L21_V001.npy         ← CLIP embeddings per-video
│   │       ├── L21_V002.npy
│   │       └── ...
│   └── DATASET/                     ← Thư mục chứa ảnh Keyframes
│       ├── Keyframes_L21/
│       │   └── keyframes/
│       │       ├── L21_V001/
│       │       │   ├── 0001.jpg
│       │       │   └── ...
│       │       └── L21_V002/
│       ├── Keyframes_L22/
│       └── ...
└── baseline/                        ← Thư mục project này
    ├── config.yaml
    ├── run_search.py
    └── ...
```

---

## 🚀 Hướng dẫn từng bước

### Bước 1 — Cài đặt môi trường

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt CLIP (bắt buộc, chọn 1 trong 2)
pip install git+https://github.com/openai/CLIP.git
# HOẶC:
# pip install open_clip_torch

# Cài đặt Milvus Lite
pip install pymilvus
```

> ⚠️ **Lưu ý**: `sentencepiece` và `sacremoses` (dùng cho dịch thuật) đã có trong `requirements.txt`, không cần cài riêng.

---

### Bước 2 — Kiểm tra / Cấu hình `config.yaml`

Mở [`config.yaml`](./config.yaml) và kiểm tra các đường dẫn:

```yaml
data:
  keyframes_root: "f:/aic/AIC2026/DATASET"   # ← Trỏ đến thư mục DATASET của bạn
  npy_dirs:
    - "f:/aic/AIC2026/clip-features-32-aic25-b1/clip-features-32"

index:
  milvus_db_path: "f:/aic/AIC2026/aic_kis_database.db"   # ← Database đã build

dino:
  alpha_clip: 0.75       # Trọng số CLIP (75% CLIP, 25% DINO)
  top_k_coarse: 30       # Số frame CLIP lấy ra để DINO chấm
```

---

### Bước 3 — Build Milvus Database (chỉ cần làm 1 lần)

> **Bỏ qua bước này** nếu `f:/aic/AIC2026/aic_kis_database.db` đã tồn tại (đã build từ `AIC2026/build.py`).

Nếu chưa có database, chạy script build:

```bash
cd f:/aic/baseline

python scripts/02b_build_milvus_db.py --config config.yaml
```

Script sẽ:
1. Quét tất cả file `.npy` trong `npy_dirs`
2. Ghép nối với ảnh keyframe tương ứng
3. Nạp toàn bộ vector vào Milvus Lite (`aic_kis_database.db`)

---

### Bước 4 — Chạy tìm kiếm

#### 🔍 Tìm kiếm đơn lẻ (Interactive Mode)

```bash
cd f:/aic/baseline

# Chạy và nhập query trực tiếp
python run_search.py --type kis
```

Hệ thống sẽ hỏi:
```
Query> một người đang mở laptop trong phòng họp
```
Hệ thống sẽ:
1. Dịch → `"a person opening a laptop in a meeting room"`
2. CLIP tìm Top 30 frames
3. Grounding DINO chấm điểm lại
4. In ra Top 10 kết quả

---

#### 📌 Tìm kiếm single query qua CLI

```bash
# Textual KIS (tiếng Việt hoặc tiếng Anh đều được)
python run_search.py --query "một người đang mở laptop" --type kis

# Lưu kết quả ra file
python run_search.py --query "hai người ngồi uống cà phê" --type kis --output submissions/

# Q&A (VQA)
python run_search.py \
  --query "cảnh bữa tiệc ngoài trời" \
  --question "Người phụ nữ mặc váy màu gì?" \
  --type qa

# TRAKE (truy vấn chuỗi sự kiện)
python run_search.py \
  --type trake \
  --events "vận động viên giậm nhảy" "bay qua xà" "tiếp đất"
```

---

#### 📦 Chạy batch queries từ file JSON

Tạo file `queries.json`:

```json
[
  {
    "query_id": "q1",
    "query_type": "textual_kis",
    "query_text": "một người đang lái xe máy trên đường phố"
  },
  {
    "query_id": "q2",
    "query_type": "textual_kis",
    "query_text": "hai người ngồi nói chuyện trong quán cà phê"
  }
]
```

Chạy batch:
```bash
python run_search.py \
  --query-file queries.json \
  --output submissions/ \
  --evaluate
```

Kết quả lưu tại `submissions/result.txt` (và `eval_report.json` nếu có ground truth).

---

### Bước 5 — Khởi động REST API (tùy chọn)

```bash
cd f:/aic/baseline

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập Swagger UI: **http://localhost:8000/docs**

Ví dụ gọi API bằng Python:

```python
import requests

# Textual KIS
response = requests.post("http://localhost:8000/search/kis", json={
    "query": "một người đang mở laptop trong phòng họp",
    "top_k": 100
})
print(response.json()["answers"][:5])
```

#### 🖼️ Lấy ảnh keyframe về hiển thị trên website

Mỗi answer trong `/search/*` đều kèm field **`image_url`** để render ảnh trực tiếp:

```python
import requests

resp = requests.post("http://localhost:8000/search/kis", json={
    "query": "một người đang mở laptop trong phòng họp",
    "top_k": 5,
})
for ans in resp.json()["answers"]:
    print(ans["rank"], ans["video_id"], ans["frame_id"], "→", ans["image_url"])
```

Dùng `image_url` trực tiếp trong thẻ `<img>` trên website:

```html
<img src="/frames/L21_V001/1" alt="Kết quả #1">
```

- **`GET /frames/{video_id}/{frame_id}`** — trả về chính ảnh keyframe (`image/jpeg` hoặc `image/png`), có hỗ trợ cache (ETag + `Cache-Control`). Chấp nhận cả `L21_V001` lẫn `L21_V001.mp4`.
- **`POST /frames/batch`** — lấy nhiều ảnh cùng lúc dưới dạng base64 data URL:

```bash
curl -X POST http://localhost:8000/frames/batch \
  -H 'Content-Type: application/json' \
  -d '{"frames":[{"video_id":"L21_V001","frame_id":1},{"video_id":"L21_V001","frame_id":99999}]}'
# → {"num_requests":2,"num_found":1,"frames":[{"video_id":"L21_V001","frame_id":1,"found":true,"data_url":"data:image/jpeg;base64,..."},{"video_id":"L21_V001","frame_id":99999,"found":false}]}
```

> **Lưu ý TRAKE:** `frame_ids` của kết quả TRAKE có dạng `[before, current, after, ...]`. `image_url` luôn trỏ tới **frame hiện tại** (middle — luôn tồn tại), còn frame trước/sau có thể là `None`.
>
> Đường dẫn ảnh được resolve ưu tiên từ các **index đã build** (`indexes/frame_map.json`, `ocr_database.json`, Milvus) rồi mới fallback heuristic trên filesystem — xem `src/frame_paths.py`.

---

## 🔧 Tinh chỉnh thông số

### Điều chỉnh alpha (cân bằng CLIP vs DINO)

Chỉnh trong `config.yaml`:

```yaml
dino:
  alpha_clip: 0.75    # 0.75 = tin CLIP nhiều hơn DINO (khuyến nghị)
                      # 0.5  = cân bằng (khi DINO tin cậy hơn)
                      # 1.0  = chỉ dùng CLIP, bỏ DINO hoàn toàn
```

### Điều chỉnh số candidates CLIP

```yaml
dino:
  top_k_coarse: 30    # Tăng lên 50-100 để DINO xem xét thêm nhiều ảnh
                      # (chậm hơn nhưng kết quả tốt hơn)
```

---

## 🐛 Xử lý lỗi thường gặp

### ❌ `Index chưa tồn tại: .../aic_kis_database.db`
```bash
# Chạy lại script build Milvus
python scripts/02b_build_milvus_db.py --config config.yaml
```

### ❌ `CUDA out of memory`
- Giảm `dino.top_k_coarse` xuống `10` hoặc `15`
- Hoặc đổi `clip.device: "cpu"` trong `config.yaml` (chậm hơn)

### ❌ `ImportError: No module named 'clip'`
```bash
pip install git+https://github.com/openai/CLIP.git
```

### ❌ `FileNotFoundError: ... opus-mt-vi-en`
```bash
# Model sẽ tự download lần đầu chạy, cần kết nối internet
# Nếu offline hoàn toàn, download trước bằng:
python -c "from transformers import MarianMTModel, MarianTokenizer; MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-vi-en'); MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-vi-en')"
```

### ❌ Không tìm thấy ảnh keyframe (`❌ Không tìm thấy ảnh`)
Kiểm tra lại cấu trúc thư mục DATASET và đường dẫn `keyframes_root` trong `config.yaml`.
Cấu trúc bắt buộc: `DATASET/Keyframes_L21/keyframes/L21_V001/0001.jpg`

---

## 📊 Luồng hoạt động chi tiết

```
Input (tiếng Việt)
       │
       ▼
 [Translate] Helsinki-NLP/opus-mt-vi-en
 Khử nhiễu stop-phrases → Dịch sang tiếng Anh
       │
       ▼ english_query
 [CLIP / Milvus] MilvusRetriever
 Encode text → vector 512D → ANN Search
 → Top 30 candidates (video_id, frame_id, clip_score)
       │
       ▼
 [Grounding DINO] IDEA-Research/grounding-dino-base
 Load ảnh từ DATASET/ → Zero-shot object detection
 → dino_score = max confidence score
       │
       ▼
 [Ensemble Scoring]
 final_score = 0.75 × clip_score + 0.25 × dino_score
       │
       ▼
 Sort by final_score (giảm dần) → Top 100
       │
       ▼
 Output: video_id, frame_id
```

---

## 📝 Format kết quả nộp bài

| Loại query | Format mỗi dòng |
|------------|-----------------|
| Textual KIS | `L21_V001.mp4, 1234` |
| Q&A | `L21_V001.mp4, 1234, màu đỏ` |
| TRAKE | `L21_V001.mp4, 100, 250, 380` |

- Tối đa **100 câu trả lời** mỗi query
- Sắp xếp theo **độ tin cậy giảm dần** (quan trọng nhất lên đầu)
