import os
# Giới hạn số luồng nội bộ để CPU không bị tranh chấp tài nguyên (chống giật máy)
os.environ["OMP_NUM_THREADS"] = "1" 

import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ
# ==========================================
KEYFRAMES_DIR = "DATASET"
OUTPUT_OCR_FILE = "ocr_database.json"
BATCH_SIZE = 8
NUM_WORKERS = 4  # Số luồng CPU dùng để đọc ảnh (Tăng lên 8 nếu CPU bạn xịn)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Đang khởi động trích xuất OCR trên thiết bị: {device.upper()}")

# ==========================================
# 2. KHAI BÁO CLASS ĐỌC ẢNH ĐA LUỒNG
# ==========================================
class KeyframeDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        rel_path, full_path = self.image_list[idx]
        try:
            img = Image.open(full_path).convert("RGB")
            return rel_path, img
        except Exception:
            return rel_path, None

def custom_collate(batch):
    """Hàm gom mẻ: Lọc bỏ những ảnh bị lỗi không đọc được"""
    valid_rels, valid_imgs = [], []
    for rel_path, img in batch:
        if img is not None:
            valid_rels.append(rel_path)
            valid_imgs.append(img)
    return valid_rels, valid_imgs

# ==========================================
# 3. CHẠY CHÍNH (MAIN TRÌNH)
# ==========================================
if __name__ == "__main__":
    print("Đang tải Florence-2...")
    florence_model_name = "microsoft/Florence-2-base"
    florence_processor = AutoProcessor.from_pretrained(florence_model_name, trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained(
        florence_model_name, 
        trust_remote_code=True,
        attn_implementation="sdpa"
    ).to(device)
    florence_model.eval()

    print("Đang quét danh sách ảnh trong DATASET...")
    image_list = []
    for root, dirs, files in os.walk(KEYFRAMES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, KEYFRAMES_DIR).replace("\\", "/")
                image_list.append((rel_path, full_path))

    ocr_db = {}
    if os.path.exists(OUTPUT_OCR_FILE):
        with open(OUTPUT_OCR_FILE, "r", encoding="utf-8") as f:
            ocr_db = json.load(f)
        print(f"Đã tải {len(ocr_db)} bản ghi cũ.")

    pending_images = [(rel, full) for rel, full in image_list if rel not in ocr_db]
    print(f"🔥 Còn {len(pending_images)} ảnh cần trích xuất OCR.")

    # Khởi tạo Dataloader đa luồng
    dataset = KeyframeDataset(pending_images)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        collate_fn=custom_collate,
        pin_memory=True if device == "cuda" else False # Tăng tốc đẩy dữ liệu từ RAM lên VRAM
    )

    print(f"Bắt đầu xử lý (Batch Size = {BATCH_SIZE}, Workers = {NUM_WORKERS})...")

    # Vòng lặp xử lý chính
    for i, (valid_rels, images) in enumerate(tqdm(dataloader, desc="Tiến độ OCR", unit="batch")):
        if not images:
            continue
            
        prompts = ["<OCR>"] * len(images)
        
        try:
            # Main thread đẩy lên GPU
            inputs = florence_processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                generated_ids = florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=1
                )
                
            generated_texts = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)
            
            for j, (gen_text, img) in enumerate(zip(generated_texts, images)):
                parsed_answer = florence_processor.post_process_generation(
                    gen_text, 
                    task="<OCR>", 
                    image_size=(img.width, img.height)
                )
                
                text_found = parsed_answer.get('<OCR>', '').strip().lower()
                if text_found: 
                    ocr_db[valid_rels[j]] = text_found
                    
        except Exception as e:
            print(f"\nLỗi xử lý GPU: {e}")
            
        # Lưu file định kỳ
        if i > 0 and i % 50 == 0:
            with open(OUTPUT_OCR_FILE, "w", encoding="utf-8") as f:
                json.dump(ocr_db, f, ensure_ascii=False, indent=2)

    # Lưu lần cuối
    with open(OUTPUT_OCR_FILE, "w", encoding="utf-8") as f:
        json.dump(ocr_db, f, ensure_ascii=False, indent=2)

    print(f"✅ Hoàn thành! Đã lưu tổng cộng {len(ocr_db)} bản ghi OCR vào {OUTPUT_OCR_FILE}")
