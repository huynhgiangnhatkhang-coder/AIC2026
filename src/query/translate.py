"""
AIC 2026 Baseline — Query Translation (Offline)
=================================================
Dịch tiếng Việt → tiếng Anh offline bằng Helsinki-NLP/opus-mt-vi-en.
Chuyển từ AIC2026/translate.py sang baseline.

Yêu cầu: pip install transformers sentencepiece sacremoses
"""
import re
import torch
from transformers import MarianMTModel, MarianTokenizer

# Lazy-load: model được tải lần đầu khi hàm analyze_query_offline_mt được gọi
_mt_model = None
_mt_tokenizer = None
_MT_MODEL_NAME = "Helsinki-NLP/opus-mt-vi-en"

# Các cụm từ vô nghĩa cần khử nhiễu trước khi dịch
STOP_PHRASES = [
    "tìm video về", "tìm video", "có cảnh", "cho tôi xem",
    "hãy tìm", "video quay cảnh", "xuất hiện", "hình ảnh", "phía sau có",
    "con", "đang", "cái",
]


def _get_model():
    """Lazy-load mô hình dịch thuật (chỉ load 1 lần)."""
    global _mt_model, _mt_tokenizer
    if _mt_model is None:
        print("[Translate] Đang tải mô hình dịch thuật Offline (opus-mt-vi-en)...")
        _mt_tokenizer = MarianTokenizer.from_pretrained(_MT_MODEL_NAME)
        _mt_model = MarianMTModel.from_pretrained(_MT_MODEL_NAME)
        print("[Translate] Mô hình dịch thuật đã tải xong.")
    return _mt_model, _mt_tokenizer


def analyze_query_offline_mt(vietnamese_query: str) -> dict:
    """
    Xử lý truy vấn tiếng Việt:
      A. Khử nhiễu (loại bỏ stop phrases)
      B. Dịch sang tiếng Anh bằng opus-mt-vi-en (100% Offline)
      C. Trích xuất danh sách object từ bản dịch

    Args:
        vietnamese_query: Câu mô tả tiếng Việt thô từ người dùng

    Returns:
        {
            "clip_query":       str,        # câu tiếng Anh để query CLIP
            "required_objects": List[str],  # danh sách vật thể trích xuất
        }
    """
    print("\n[Translate] Đang phân tích truy vấn và dịch thuật Offline...")
    mt_model, mt_tokenizer = _get_model()

    query_lower = vietnamese_query.lower()

    # PHẦN A: Khử nhiễu tiếng Việt
    cleaned_query = query_lower
    for phrase in STOP_PHRASES:
        cleaned_query = re.sub(fr'\b{phrase}\b', '', cleaned_query).strip()
    cleaned_query = re.sub(r' +', ' ', cleaned_query).strip()

    # PHẦN B: Dịch sang tiếng Anh (100% Offline)
    inputs = mt_tokenizer(cleaned_query, return_tensors="pt", padding=True)
    with torch.no_grad():
        translated_tokens = mt_model.generate(**inputs, max_new_tokens=50)
    clip_query = mt_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # PHẦN C: Trích xuất danh sách object từ bản dịch
    clean_eng = clip_query.replace(".", "").lower().strip()
    raw_objects = re.split(r',|\band\b', clean_eng)
    required_objects = []
    for obj in raw_objects:
        obj = obj.strip()
        if obj and obj not in required_objects:
            required_objects.append(obj)
    required_objects = required_objects[:10]  # Giới hạn tối đa 10 objects

    print(f"[Translate] -> Câu tiếng Việt đã lọc: '{cleaned_query}'")
    print(f"[Translate] -> Dịch sang tiếng Anh (Opus-MT): '{clip_query}'")
    print(f"[Translate] -> Vật thể bắt buộc trích xuất: {required_objects}")

    return {
        "clip_query": clip_query,
        "required_objects": required_objects,
    }
