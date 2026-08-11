import os
import json
import torch
from PIL import Image
from pymilvus import MilvusClient
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel
from translate import analyze_query_offline_mt
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ & DATA OCR
# ==========================================    
KEYFRAMES_DIR = "DATASET"
MILVUS_DB_PATH = "aic_kis_database_siglip.db"
COLLECTION_NAME = "kis_keyframes_siglip"
OCR_DB_PATH = "ocr_database.json"

# Tải dữ liệu OCR vào bộ nhớ
OCR_DATA = {}
if os.path.exists(OCR_DB_PATH):
    with open(OCR_DB_PATH, "r", encoding="utf-8") as f:
        OCR_DATA = json.load(f)
    print(f"✅ Đã tải {len(OCR_DATA)} bản ghi OCR vào bộ nhớ!")
else:
    print(f"⚠️ Không tìm thấy file {OCR_DB_PATH}. Tính năng OCR sẽ bị bỏ qua (0 điểm).")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH (SIGLIP & FLORENCE-2)
# ==========================================
print("Đang tải các mô hình AI lên bộ nhớ (SigLIP & Florence-2)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tải Google SigLIP (Lọc thô)
siglip_model_name = "google/siglip-base-patch16-224"
siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)
siglip_model = AutoModel.from_pretrained(siglip_model_name).to(device)
siglip_model.eval()

# Tải Florence-2 (Trọng tài Re-ranking)
florence_model_name = "microsoft/Florence-2-base"
florence_processor = AutoProcessor.from_pretrained(florence_model_name, trust_remote_code=True)
florence_model = AutoModelForCausalLM.from_pretrained(
    florence_model_name, 
    trust_remote_code=True,
    attn_implementation="sdpa" # Dùng cơ chế tăng tốc tích hợp sẵn của PyTorch
).to(device)
florence_model.eval()

# ==========================================
# 3. CÁC HÀM CHẤM ĐIỂM (FLORENCE & OCR)
# ==========================================
def get_florence_scores_batch(image_paths, required_objects, batch_size=8):
    """
    Xử lý song song nhiều ảnh cùng lúc bằng Florence-2 để tăng tốc độ.
    """
    if not required_objects or not image_paths:
        return [1.0] * len(image_paths) if not required_objects else [0.0] * len(image_paths)
        
    text_input = " and ".join(required_objects)
    prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {text_input}"
    
    all_scores = []
    
    # Chia nhỏ danh sách ảnh thành các batch
    for i in tqdm(range(0, len(image_paths), batch_size), desc="[Florence-2] Batch Re-ranking", unit="batch"):
        batch_paths = image_paths[i:i+batch_size]
        
        images = []
        valid_indices = []
        
        # Load các ảnh trong batch
        for idx, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Lỗi đọc ảnh tại {path}: {e}")
                
        if not images:
            all_scores.extend([0.0] * len(batch_paths))
            continue
            
        prompts = [prompt] * len(images)
        
        try:
            # Đẩy nguyên 1 batch vào GPU
            inputs = florence_processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                generated_ids = florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
                
            generated_texts = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)
            
            batch_scores = [0.0] * len(batch_paths)
            
            # Tính điểm cho từng ảnh trong batch
            for j, (gen_text, img) in enumerate(zip(generated_texts, images)):
                parsed_answer = florence_processor.post_process_generation(
                    gen_text, 
                    task="<CAPTION_TO_PHRASE_GROUNDING>", 
                    image_size=(img.width, img.height)
                )
                
                results = parsed_answer.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
                labels_found = results.get('labels', [])
                
                if labels_found:
                    unique_labels_found = set(labels_found)
                    score = len(unique_labels_found) / len(required_objects)
                    batch_scores[valid_indices[j]] = min(score, 1.0)
                    
            all_scores.extend(batch_scores)
            
        except Exception as e:
            print(f"Lỗi Batch inference Florence-2: {e}")
            all_scores.extend([0.0] * len(batch_paths))
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return all_scores

def get_ocr_score(image_rel_path, search_text):
    # (GIỮ NGUYÊN CODE CỦA HÀM NÀY NHƯ CŨ)
    if not search_text or not OCR_DATA:
        return 0.0
    image_rel_path = image_rel_path.replace("\\", "/") 
    ocr_text = ""
    for key, val in OCR_DATA.items():
        if image_rel_path.endswith(key.replace("\\", "/")):
            ocr_text = val
            break
    if not ocr_text:
        return 0.0
    search_keywords = search_text.lower().split()
    if not search_keywords:
        return 0.0
    matched = sum(1 for kw in search_keywords if kw in ocr_text)
    return matched / len(search_keywords)

# ==========================================
# 4. HÀM TÌM KIẾM VÀ RE-RANK KẾT HỢP
# ==========================================
def search_kis_with_florence(client, parsed_query_data, top_k=5):
    clip_query = parsed_query_data["clip_query"]
    required_objects = parsed_query_data["required_objects"]
    
    print(f"\n[SigLIP] Đang mã hóa câu truy vấn: '{clip_query}'")
    
    text_inputs = siglip_processor(
        text=[clip_query], 
        padding="max_length", 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        text_features = siglip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_vector = text_features[0].cpu().tolist()

    print("[Milvus] Đang truy xuất Top 50 khung hình gần giống nhất...") 
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[text_vector],
        limit=50, 
        output_fields=["video_id", "frame_id"], 
        search_params={"metric_type": "IP"}
    )

    # 1. Tìm toàn bộ đường dẫn ảnh hợp lệ trước
    valid_hits = []
    valid_image_paths = []
    
    for hit in search_results[0]:
        v_id = hit["entity"]["video_id"]
        f_id = hit["entity"]["frame_id"]
        
        video_folder = v_id.replace(".mp4", "")
        batch_prefix = video_folder.split("_")[0]  
        batch_folder_name = f"Keyframes_{batch_prefix}" 
        
        possible_formats = [
            f"{f_id}.jpg", f"{f_id:03d}.jpg", f"{f_id:04d}.jpg", 
            f"{f_id:05d}.jpg", f"{f_id:06d}.jpg",
            f"{f_id}.png", f"{f_id:04d}.png" 
        ]
        
        image_path = None
        for fmt in possible_formats:
            temp_path = os.path.join(KEYFRAMES_DIR, batch_folder_name, "keyframes", video_folder, fmt)
            if os.path.exists(temp_path):
                image_path = temp_path
                break
                
        if image_path:
            valid_hits.append(hit)
            valid_image_paths.append(image_path)

    # 2. Re-rank đồng loạt bằng Florence-2 (Batch Size = 8)
    print(f"[Tiến trình] Đang chấm điểm {len(valid_image_paths)} ảnh. Từ khóa: {required_objects}...")
    
    # BẠN CÓ THỂ CHỈNH BATCH_SIZE TẠI ĐÂY (4, 8, 16) tùy vào VRAM GPU
    florence_scores = get_florence_scores_batch(valid_image_paths, required_objects, batch_size=8)
    
    # 3. Tổng hợp điểm (SigLIP + Florence + OCR)
    scored_results = []
    for hit, image_path, florence_score in zip(valid_hits, valid_image_paths, florence_scores):
        clip_score = hit["distance"]
        
        rel_image_path = os.path.relpath(image_path, KEYFRAMES_DIR)
        ocr_score = get_ocr_score(rel_image_path, clip_query)

        final_score = (0.6 * clip_score) + (0.3 * florence_score) + (0.1 * ocr_score)
        if ocr_score > 0.5:
            final_score += 0.2
            
        scored_results.append({
            "hit": hit,
            "clip_score": clip_score,
            "florence_score": florence_score,
            "ocr_score": ocr_score,
            "final_score": final_score
        })

    # Sắp xếp lại theo điểm tổng hợp từ cao xuống thấp
    scored_results = sorted(scored_results, key=lambda x: x["final_score"], reverse=True)

    print("\n=== KẾT QUẢ TÌM KIẾM CHÍNH THỨC ===")
    if not scored_results:
        print("Không tìm thấy khung hình nào khớp!")
    else:
        for item in scored_results[:top_k]:
            hit = item["hit"]
            v_id = hit["entity"]["video_id"]
            f_id = hit["entity"]["frame_id"]
            print(f"▶ video_id = {v_id}, frame_id = {f_id} | Tổng điểm: {item['final_score']:.4f} "
                  f"(SigLIP: {item['clip_score']:.4f} | Florence: {item['florence_score']:.4f} | OCR: {item['ocr_score']:.4f})")

# ==========================================
# 5. CHẠY CHÍNH
# ==========================================
if __name__ == "__main__":
    print("Đang kết nối vào Database...")
    try:
        milvus_client = MilvusClient(MILVUS_DB_PATH)
        milvus_client.load_collection(COLLECTION_NAME) 
        
        raw_query = input("\nM kiếm gì?\nT kiếm: ")
        parsed_query_data = analyze_query_offline_mt(raw_query)
        
        # Chạy tìm kiếm, in ra top 5
        search_kis_with_florence(milvus_client, parsed_query_data, top_k=5)
        
    except Exception as e:
        print(f"Lỗi! Chi tiết: {e}")
