import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm

# Cấu hình
KEYFRAMES_DIR = "DATASET"
OUTPUT_OCR_FILE = "ocr_database.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Đang tải Florence-2 để trích xuất OCR...")
florence_model_name = "microsoft/Florence-2-base"
florence_processor = AutoProcessor.from_pretrained(florence_model_name, trust_remote_code=True)
florence_model = AutoModelForCausalLM.from_pretrained(
    florence_model_name, 
    trust_remote_code=True,
    attn_implementation="sdpa"
).to(device)
florence_model.eval()

def run_ocr(image_path):
    """Sử dụng tác vụ <OCR> của Florence-2 để đọc toàn bộ chữ trong ảnh"""
    prompt = "<OCR>"
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=1
            )
            
        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = florence_processor.post_process_generation(
            generated_text, 
            task="<OCR>", 
            image_size=(image.width, image.height)
        )
        
        # Kết quả trả về là chuỗi văn bản nhận diện được
        text_found = parsed_answer.get('<OCR>', '').strip().lower()
        return text_found
    except Exception:
        return ""

# Lấy danh sách toàn bộ ảnh trong tập dataset
print("Đang quét danh sách ảnh...")
image_list = []
for root, dirs, files in os.walk(KEYFRAMES_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            full_path = os.path.join(root, file)
            # Tạo key định danh dạng: video_id/frame_id
            rel_path = os.path.relpath(full_path, KEYFRAMES_DIR)
            image_list.append((rel_path, full_path))

print(f"Tìm thấy {len(image_list)} ảnh. Bắt đầu đọc chữ OCR...")

ocr_db = {}
# Tải lại database cũ nếu đang chạy dở
if os.path.exists(OUTPUT_OCR_FILE):
    with open(OUTPUT_OCR_FILE, "r", encoding="utf-8") as f:
        ocr_db = json.load(f)

for rel_path, full_path in tqdm(image_list, desc="OCR Progress"):
    if rel_path in ocr_db:
        continue # Bỏ qua ảnh đã trích xuất rồi
        
    text = run_ocr(full_path)
    if text: # Chỉ lưu nếu có chữ
        ocr_db[rel_path] = text
        
    # Lưu định kỳ sau mỗi 500 ảnh để tránh mất dữ liệu khi cúp điện
    if len(ocr_db) % 500 == 0:
        with open(OUTPUT_OCR_FILE, "w", encoding="utf-8") as f:
            json.dump(ocr_db, f, ensure_ascii=False, indent=2)

# Lưu lần cuối
with open(OUTPUT_OCR_FILE, "w", encoding="utf-8") as f:
    json.dump(ocr_db, f, ensure_ascii=False, indent=2)

print(f"Đã trích xuất xong OCR cho {len(ocr_db)} ảnh và lưu vào {OUTPUT_OCR_FILE}")
