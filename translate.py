import re
import torch
from transformers import MarianMTModel, MarianTokenizer

print("Đang tải mô hình dịch thuật Offline (opus-mt-vi-en)...")
# Tải mô hình dịch thuật cục bộ
MT_MODEL_NAME = "Helsinki-NLP/opus-mt-vi-en"
mt_tokenizer = MarianTokenizer.from_pretrained(MT_MODEL_NAME)
mt_model = MarianMTModel.from_pretrained(MT_MODEL_NAME)

# 1. BỘ TỪ ĐIỂN ÁNH XẠ OBJECTS (Giữ nguyên như cũ)
OPENIMAGES_CLASSES_MAP = {
    "Tree": ["cây xanh", "cây cối", "cái cây", "rừng"],
    "Person": ["người", "đàn ông", "phụ nữ", "nhân viên", "diễn giả", "cảnh sát"],
    "Car": ["xe hơi", "ô tô", "xe con"],
    # ... bổ sung thêm ...
}

# 2. CÁC TỪ VÔ NGHĨA CẦN KHỬ NHIỄU
STOP_PHRASES = [
    "tìm video về", "tìm video", "có cảnh", "cho tôi xem", 
    "hãy tìm", "video quay cảnh", "xuất hiện", "hình ảnh", "phía sau có"
]

def analyze_query_offline_mt(vietnamese_query):
    print("\nĐang phân tích truy vấn và dịch thuật Offline...")
    query_lower = vietnamese_query.lower()
    required_objects = []

    # PHẦN A: TRÍCH XUẤT OBJECT BẰNG TỪ ĐIỂN
    for eng_class, vi_synonyms in OPENIMAGES_CLASSES_MAP.items():
        for syn in vi_synonyms:
            if re.search(fr'\b{syn}\b', query_lower):
                required_objects.append(eng_class)
                break

    # PHẦN B: KHỬ NHIỄU
    cleaned_query = query_lower
    for phrase in STOP_PHRASES:
        cleaned_query = re.sub(fr'\b{phrase}\b', '', cleaned_query).strip()
    cleaned_query = re.sub(' +', ' ', cleaned_query) # Xóa khoảng trắng thừa

    # PHẦN C: DỊCH SANG TIẾNG ANH BẰNG OPUS-MT (100% OFFLINE)
    # Tokenize câu tiếng Việt
    inputs = mt_tokenizer(cleaned_query, return_tensors="pt", padding=True)
    
    # Sinh câu dịch
    with torch.no_grad():
        translated_tokens = mt_model.generate(**inputs)
        
    # Giải mã (Decode) ra văn bản tiếng Anh
    clip_query = mt_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    print(f"-> Câu tiếng Việt đã lọc: '{cleaned_query}'")
    print(f"-> Dịch sang tiếng Anh (Opus-MT): '{clip_query}'")
    print(f"-> Vật thể bắt buộc: {required_objects}")
    
    return {
        "clip_query": clip_query,
        "required_objects": required_objects
    }