import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Đang tải mô hình dịch thuật Offline (VietAI/envit5-translation)...")
# Tải mô hình dịch thuật cục bộ T5
MT_MODEL_NAME = "VietAI/envit5-translation"
mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mt_model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL_NAME).to(device)

# CÁC TỪ VÔ NGHĨA CẦN KHỬ NHIỄU TRƯỚC KHI DỊCH
STOP_PHRASES = [
    "tìm video về", "tìm video", "có cảnh", "cho tôi xem", 
    "hãy tìm", "video quay cảnh", "xuất hiện", "hình ảnh", "phía sau có",
    "tìm cho tôi", "tìm giúp tôi", "cho xem", "tìm", "những", "các",
    "con", "đang", "cái", "một chiếc", "một", "chiếc",
]

def is_english(text):
    # Một hàm kiểm tra nhanh xem query có phải tiếng Anh không
    # Nếu không có dấu tiếng Việt nào, khả năng cao là tiếng Anh
    vi_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    return not any(c in text.lower() for c in vi_chars)

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

    # BYPASS DỊCH THUẬT: Nếu query nhập vào là tiếng Anh, không dịch nữa
    if is_english(cleaned_query):
        clip_query = cleaned_query
        print("-> Bypass dịch thuật (Query là tiếng Anh)")
    else:
        # PHẦN B: DỊCH SANG TIẾNG ANH (100% OFFLINE)
        # envit5-translation cần tiền tố "vi: " để dịch sang tiếng Anh
        input_text = "vi: " + query_with_placeholders
        inputs = mt_tokenizer(input_text, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            translated_tokens = mt_model.generate(**inputs, max_length=512)
            
        raw_translation = mt_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        # Bỏ tiền tố "en: " do model sinh ra
        clip_query = raw_translation.replace("en: ", "", 1).strip()

    # Khôi phục danh từ riêng
    for i, p_noun in enumerate(proper_nouns):
        # Model có thể gắn thêm dấu cách hoặc viết thường placeholder, ta replace linh hoạt
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
    if proper_nouns:
        print(f"-> Danh từ riêng phát hiện: {proper_nouns}")
    
    return {
        "clip_query": clip_query,
        "required_objects": required_objects,
        "proper_nouns": proper_nouns
    }
