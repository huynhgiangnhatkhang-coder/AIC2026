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
    "hãy tìm", "video quay cảnh", "xuất hiện", "hình ảnh", "phía sau có",
    "tìm cho tôi", "tìm giúp tôi", "cho xem", "tìm", "những", "các",
    "con", "đang", "cái", "một chiếc", "một", "chiếc",
]

def analyze_query_offline_mt(vietnamese_query):
    print("\nĐang phân tích truy vấn và dịch thuật Offline...")
    
    # PHẦN A: KHỬ NHIỄU TIẾNG VIỆT (Giữ nguyên hoa/thường để nhận diện danh từ riêng)
    cleaned_query = vietnamese_query
    for phrase in STOP_PHRASES:
        cleaned_query = re.sub(fr'(?i)\b{phrase}\b', '', cleaned_query).strip()
    
    cleaned_query = re.sub(' +', ' ', cleaned_query).strip()

    # Nhận diện danh từ riêng (viết hoa) và thay bằng placeholder LOC
    words = cleaned_query.split()
    proper_nouns = []
    new_words = []
    
    for i, w in enumerate(words):
        w_clean = re.sub(r'[^\w\s]', '', w)
        if not w_clean:
            new_words.append(w)
            continue
            
        # Nếu từ có chữ cái đầu viết hoa
        if w_clean[0].isupper():
            # Nếu là từ đầu câu, chỉ coi là danh từ riêng nếu từ tiếp theo cũng viết hoa
            if i == 0:
                is_proper = False
                if len(words) > 1:
                    w2_clean = re.sub(r'[^\w\s]', '', words[1])
                    if w2_clean and w2_clean[0].isupper():
                        is_proper = True
                if is_proper:
                    proper_nouns.append(w)
                    new_words.append(f"LOC{len(proper_nouns)-1}")
                else:
                    new_words.append(w.lower())
            else:
                proper_nouns.append(w)
                new_words.append(f"LOC{len(proper_nouns)-1}")
        else:
            new_words.append(w)

    query_with_placeholders = " ".join(new_words)

    # PHẦN B: DỊCH SANG TIẾNG ANH (100% OFFLINE)
    inputs = mt_tokenizer(query_with_placeholders, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        translated_tokens = mt_model.generate(**inputs)
        
    clip_query = mt_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # Khôi phục danh từ riêng
    for i, p_noun in enumerate(proper_nouns):
        # Opus-MT có thể gắn thêm dấu cách hoặc viết thường placeholder, ta replace linh hoạt
        clip_query = re.sub(fr'(?i)loc{i}', p_noun, clip_query)

    # PHẦN C: LẤY OBJECT TRỰC TIẾP TỪ BẢN DỊCH
    # 1. Chuyển thành chữ thường và xóa dấu chấm ở cuối (thường do AI dịch tự động thêm vào)
    clean_eng = clip_query.replace(".", "").lower().strip()
    
    # 2. Tách câu thành danh sách vật thể dựa trên dấu phẩy hoặc các từ nối thông dụng
    # Giúp Florence-2 dễ dàng tìm kiếm từng vật thể đơn lẻ thay vì nguyên một câu dài
    split_pattern = r',|\band\b|\bwith\b|\bin\b|\bon\b|\bat\b|\bholding\b|\bwearing\b|\bnext to\b|\bhas\b|\bhaving\b|\bcontaining\b|\bshows\b|\bshowing\b'
    raw_objects = re.split(split_pattern, clean_eng)
    
    # 3. Lọc bỏ các khoảng trắng thừa và các từ vô nghĩa tiếng Anh ở đầu/cuối mỗi object
    english_stop_phrases = ["find me a", "find me", "show me a", "show me", "a", "an", "the", "some"]
    required_objects = []
    for obj in raw_objects:
        obj = obj.strip()
        for sp in english_stop_phrases:
            if obj.startswith(sp + " "):
                obj = obj[len(sp)+1:].strip()
        if obj:
            required_objects.append(obj)

    print(f"-> Câu tiếng Việt đã lọc: '{cleaned_query}'")
    print(f"-> Dịch sang tiếng Anh (Opus-MT): '{clip_query}'")
    print(f"-> Vật thể bắt buộc trích xuất: {required_objects}")
    
    return {
        "clip_query": clip_query,
        "required_objects": required_objects
    }
