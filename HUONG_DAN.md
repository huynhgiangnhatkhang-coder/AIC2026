# 🎯 Hướng dẫn chạy AIC 2026 Baseline (SigLIP + Florence-2 + OCR)

> Pipeline tìm kiếm video sử dụng **Hai Kênh Độc Lập**: Kênh Hình (SigLIP) + Kênh Chữ (OCR), sau đó lọc lại độ chính xác cực cao bằng **Florence-2**.

---

## 📋 Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|------------|-------------------|
| Python | 3.9+ |
| GPU VRAM | ≥ 8GB (chạy mượt SigLIP và Florence-2) |
| RAM | ≥ 16GB |
| CUDA | 11.8+ (Bắt buộc để chạy mô hình AI nhanh) |

---

## 📁 Cấu trúc dữ liệu yêu cầu

Đảm bảo thư mục dữ liệu (DATASET) của bạn có cấu trúc sau:

```
f:/aic/
├── AIC2026/
│   ├── aic_kis_database_siglip.db   ← Database Milvus (Build bằng SigLIP)
│   ├── ocr_database.json            ← Database OCR text
│   └── DATASET/                     ← Thư mục chứa ảnh Keyframes
│       ├── Keyframes_L21/
│       │   └── keyframes/
│       │       ├── L21_V001/
│       │       │   ├── 0001.jpg
│       │       │   └── ...
└── baseline/                        ← Thư mục project này
    ├── config.yaml
    └── run_search.py
```

---

## 🚀 Hướng dẫn từng bước

### Bước 1 — Cài đặt môi trường

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies cơ bản
pip install -r requirements.txt

# Cài đặt PyTorch với CUDA (nếu chưa có)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt thư viện cho Qwen2-VL (VQA)
pip install qwen-vl-utils accelerate
```

### Bước 2 — Kiểm tra / Cấu hình `config.yaml`

Mở [`config.yaml`](./config.yaml) và kiểm tra các đường dẫn:

```yaml
data:
  keyframes_root: "./DATASET"   # ← Trỏ đến thư mục DATASET của bạn

index:
  milvus_db_path: "./aic_kis_database_siglip.db"   # ← Database Milvus SigLIP
```

### Bước 3 — Chạy Tìm kiếm (CLI)

Hệ thống hỗ trợ 3 chế độ tìm kiếm KIS chính: `visual`, `text`, `hybrid`.

```bash
# 1. Chế độ Mặc định (Tự động chạy và hiển thị 3 Grid: Thuần Hình / Thuần Chữ / Hỗn hợp để so sánh)
python run_search.py --query "đoàn người đua xe đạp về đích ở Tam Kì, Quảng Nam"

# 2. Chế độ Thuần Chữ (Chỉ dùng OCR - Rất nhanh, lý tưởng để tìm số nhà, địa danh, chữ trên áo)
python run_search.py --query "Tam Kì" --search-mode text

# 3. Chế độ Thuần Hình Ảnh (SigLIP + Florence - Lờ đi chữ cái gây nhiễu, lý tưởng tìm sự kiện)
python run_search.py --query "đoàn người đua xe đạp về đích" --search-mode visual

# 4. Tìm kiếm Q&A (Video Question Answering - Dùng Qwen2-VL)
python run_search.py --query "cảnh bữa tiệc" --question "Váy của cô gái màu gì?" --type qa

# 5. Truy vấn chuỗi sự kiện theo thời gian (TRAKE)
python run_search.py --type trake --events "vận động viên giậm nhảy" "bay qua xà" "tiếp đất"
```

### Bước 4 — Khởi động Full-stack (API + Giao diện Web)

Project đã được trang bị **giao diện web (Frontend bằng React/Vite)** và **API (FastAPI)**.

Chạy script khởi động tự động:
```bash
bash start.sh
```
Lệnh này sẽ bật:
- **Backend API**: `http://localhost:8000`
- **Frontend UI**: `http://localhost:5173`

*(Truy cập `http://localhost:5173` trên trình duyệt để sử dụng giao diện đồ họa).*

---

## 📊 Luồng hoạt động Textual KIS mới

1. **Dịch thuật Offline (Opus-MT):** Dịch câu truy vấn tiếng Việt sang tiếng Anh và tách từ khóa.
2. **Kênh 1 (Hình Ảnh - SigLIP):** Tìm Top 1000 khung hình tương đồng nhất bằng Milvus.
3. **Kênh 2 (Chữ - OCR):** Quét độc lập `ocr_database.json` để tìm frame có chứa chữ trùng khớp với yêu cầu.
4. **Hợp nhất (Pre-ranking):** Ghép kết quả 2 kênh.
5. **Re-ranking bằng Florence-2:** Đưa Top 100 ảnh kết quả vào mô hình Florence-2 (Phrase Grounding) để dò tìm vật thể chi tiết.
6. **Lọc OCR Override:** Nếu từ khóa tìm kiếm có địa danh hoặc text cụ thể, điểm OCR sẽ được ưu tiên để đẩy frame đó lên Top 1.

---

## 📝 Format kết quả nộp bài (Batch)

Để nộp bài chấm điểm, tạo file `queries.json` chứa các câu hỏi, sau đó chạy:

```bash
python run_search.py \
  --query-file queries.json \
  --output submissions/ \
  --evaluate
```

Hệ thống sẽ tự lưu file `result.txt` chứa kết quả chuẩn format:

| Loại query | Format mỗi dòng |
|------------|-----------------|
| Textual KIS | `L21_V001.mp4, 1234` |
| Q&A | `L21_V001.mp4, 1234, màu đỏ` |
| TRAKE | `L21_V001.mp4, 100, 250, 380` |
