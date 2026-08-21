import os
# Giới hạn số luồng nội bộ để CPU không bị tranh chấp tài nguyên (chống giật máy)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Tắt cảnh báo fork tokenizer

import json
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ
# ==========================================
KEYFRAMES_DIR = "DATASET"
OUTPUT_OCR_FILE = "ocr_database.json"
BATCH_SIZE = 32
NUM_WORKERS = 8
EAST_CHUNK_SIZE = 100  # Chia nhỏ 100 ảnh mỗi mẻ để lọc rồi chạy Florence ngay

# --- EAST Text Detector ---
EAST_MODEL_PATH = "frozen_east_text_detection.pb"
EAST_INPUT_SIZE = 320
EAST_MIN_TEXT_RATIO = 0.005  # Tối thiểu 0.5% diện tích ảnh phải là chữ (lọc logo HTV)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Đang khởi động trích xuất OCR trên thiết bị: {device.upper()}")

# ==========================================
# 2. BỘ LỌC NHANH EAST TEXT DETECTOR
# ==========================================
print("Đang tải mô hình EAST Text Detector...")
east_net = cv2.dnn.readNet(EAST_MODEL_PATH)
east_output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
print("✅ EAST Text Detector đã sẵn sàng.")

def has_text_east(img_path):
    """
    Kiểm tra nhanh xem ảnh có chữ không (đã che đi góc trái trên cùng chứa logo HTV).
    """
    image = cv2.imread(img_path)
    if image is None:
        return False
    
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (EAST_INPUT_SIZE, EAST_INPUT_SIZE),
        (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    east_net.setInput(blob)
    scores, _ = east_net.forward(east_output_layers)
    
    scores_map = scores[0, 0, :, :]
    
    # XÓA LOGO HTV: Che đen (set score = 0) ở góc trên cùng bên trái
    # Ma trận scores_map có kích thước 80x80.
    # Logo HTV thường nằm ở 20% chiều cao trên cùng và 30% chiều rộng bên trái
    scores_map[0:16, 0:24] = 0.0
    
    # Chỉ cần CÓ 1 pixel nào đó (ngoài vùng logo) là chữ (score > 0.5) thì lấy ảnh đó
    return float(np.max(scores_map)) > 0.5


# ==========================================
# 3. KHAI BÁO CLASS ĐỌC ẢNH ĐA LUỒNG
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


def run_florence_on_batch(filtered_images, florence_processor, florence_model, ocr_db):
    """Chạy Florence-2 OCR trên danh sách ảnh đã lọc qua EAST."""
    if not filtered_images:
        return 0
    
    dataset = KeyframeDataset(filtered_images)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=min(NUM_WORKERS, len(filtered_images)),
        collate_fn=custom_collate,
        pin_memory=(device == "cuda")
    )
    
    new_count = 0
    for valid_rels, images in dataloader:
        if not images:
            continue
        prompts = ["<OCR>"] * len(images)
        try:
            inputs = florence_processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
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
                    gen_text, task="<OCR>", image_size=(img.width, img.height)
                )
                text_found = parsed_answer.get('<OCR>', '').strip().lower()
                if text_found:
                    ocr_db[valid_rels[j]] = text_found
                    new_count += 1
        except Exception as e:
            print(f"\n  ⚠️ Lỗi Florence-2: {e}")
    
    return new_count


# ==========================================
# 4. CHẠY CHÍNH
# ==========================================
if __name__ == "__main__":
    print("Đang tải Florence-2...")
    florence_model_name = "microsoft/Florence-2-base"
    florence_processor = AutoProcessor.from_pretrained(florence_model_name, trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained(
        florence_model_name, 
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.float16
    ).to(device)
    florence_model.eval()

    print("Đang quét danh sách ảnh trong DATASET (chỉ lấy L26)...")
    image_list = []
    for root, dirs, files in os.walk(KEYFRAMES_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, KEYFRAMES_DIR).replace("\\", "/")
                if any(rel_path.startswith(f"Keyframes_L{L}") for L in [26]):
                    image_list.append((rel_path, full_path))

    ocr_db = {}
    if os.path.exists(OUTPUT_OCR_FILE):
        with open(OUTPUT_OCR_FILE, "r", encoding="utf-8") as f:
            ocr_db = json.load(f)
        print(f"Đã tải {len(ocr_db)} bản ghi cũ.")

    pending_images = [(rel, full) for rel, full in image_list if rel not in ocr_db]
    total_pending = len(pending_images)
    print(f"📋 Tổng ảnh cần xử lý: {total_pending}")

    # --- XỬ LÝ THEO CHUNK: EAST lọc 100 → Florence → Lưu JSON → Lặp lại ---
    total_chunks = (total_pending + EAST_CHUNK_SIZE - 1) // EAST_CHUNK_SIZE
    total_east_passed = 0
    total_east_skipped = 0
    total_ocr_found = 0

    for chunk_idx in range(total_chunks):
        start = chunk_idx * EAST_CHUNK_SIZE
        end = min(start + EAST_CHUNK_SIZE, total_pending)
        chunk = pending_images[start:end]

        # Bước 1: Lọc EAST với tỷ lệ diện tích
        filtered = []
        skipped = 0
        for rel_path, full_path in chunk:
            if has_text_east(full_path):
                filtered.append((rel_path, full_path))
            else:
                skipped += 1
        
        total_east_passed += len(filtered)
        total_east_skipped += skipped

        # Bước 2: Chạy Florence-2 trên ảnh đã lọc
        new_ocr = run_florence_on_batch(filtered, florence_processor, florence_model, ocr_db)
        total_ocr_found += new_ocr

        # Bước 3: Lưu ngay vào JSON
        with open(OUTPUT_OCR_FILE, "w", encoding="utf-8") as f:
            json.dump(ocr_db, f, ensure_ascii=False, indent=2)

        # In tiến độ
        processed = end
        print(
            f"[Chunk {chunk_idx+1}/{total_chunks}] "
            f"Đã xử lý: {processed}/{total_pending} | "
            f"EAST lọc: {len(filtered)} có chữ / {skipped} skip | "
            f"Florence tìm thấy: +{new_ocr} OCR | "
            f"Tổng DB: {len(ocr_db)}"
        )

    print(f"\n{'='*60}")
    print(f"✅ HOÀN THÀNH!")
    print(f"   Tổng ảnh đã quét:     {total_pending}")
    print(f"   EAST phát hiện chữ:   {total_east_passed} ({total_east_passed*100//max(1,total_pending)}%)")
    print(f"   EAST bỏ qua (logo HTV): {total_east_skipped} ({total_east_skipped*100//max(1,total_pending)}%)")
    print(f"   Florence OCR mới:     {total_ocr_found}")
    print(f"   Tổng bản ghi OCR DB: {len(ocr_db)}")
    print(f"   Đã lưu vào: {OUTPUT_OCR_FILE}")
