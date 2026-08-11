import re
import torch
from transformers import MarianMTModel, MarianTokenizer

print("Đang tải mô hình dịch thuật Offline (opus-mt-vi-en)...")
# Tải mô hình dịch thuật cục bộ
MT_MODEL_NAME = "Helsinki-NLP/opus-mt-vi-en"
mt_tokenizer = MarianTokenizer.from_pretrained(MT_MODEL_NAME)
mt_model = MarianMTModel.from_pretrained(MT_MODEL_NAME)

# CÁC TỪ VÔ NGHĨA CẦN KHỬ NHIỄU TRƯỚC KHI DỊCH
STOP_PHRASES = [
    "tìm video về", "tìm video", "có cảnh", "cho tôi xem", 
    "hãy tìm", "video quay cảnh", "xuất hiện", "hình ảnh", "phía sau có"
]

def analyze_query_offline_mt(vietnamese_query):
    print("\nĐang phân tích truy vấn và dịch thuật Offline...")
    query_lower = vietnamese_query.lower()

    # PHẦN A: KHỬ NHIỄU TIẾNG VIỆT
    cleaned_query = query_lower
    for phrase in STOP_PHRASES:
        cleaned_query = re.sub(fr'\b{phrase}\b', '', cleaned_query).strip()
    
    # Xóa khoảng trắng thừa
    cleaned_query = re.sub(' +', ' ', cleaned_query).strip()

    # PHẦN B: DỊCH SANG TIẾNG ANH (100% OFFLINE)
    inputs = mt_tokenizer(cleaned_query, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        translated_tokens = mt_model.generate(**inputs)
        
    clip_query = mt_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # PHẦN C: LẤY OBJECT TRỰC TIẾP TỪ BẢN DỊCH
    # 1. Chuyển thành chữ thường và xóa dấu chấm ở cuối (thường do AI dịch tự động thêm vào)
    clean_eng = clip_query.replace(".", "").lower().strip()
    
    # 2. Tách câu thành danh sách vật thể dựa trên dấu phẩy hoặc chữ "and"
    # Ví dụ: "women, yellow shirts" -> ["women", "yellow shirts"]
    # Ví dụ: "a red car and a dog" -> ["a red car", "a dog"]
    raw_objects = re.split(r',|\band\b', clean_eng)
    
    # 3. Lọc bỏ các khoảng trắng thừa ở hai đầu mỗi object
    required_objects = [obj.strip() for obj in raw_objects if obj.strip()]

    print(f"-> Câu tiếng Việt đã lọc: '{cleaned_query}'")
    print(f"-> Dịch sang tiếng Anh (Opus-MT): '{clip_query}'")
    print(f"-> Vật thể bắt buộc trích xuất: {required_objects}")
    
    return {
        "clip_query": clip_query,
        "required_objects": required_objects
    }
    