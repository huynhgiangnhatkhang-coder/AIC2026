import os
import torch
from PIL import Image
from pymilvus import MilvusClient
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel
from translate import analyze_query_offline_mt
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ
# ==========================================    
KEYFRAMES_DIR = "DATASET"
MILVUS_DB_PATH = "aic_kis_database_siglip.db"
COLLECTION_NAME = "kis_keyframes_siglip"

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
florence_model = AutoModelForCausalLM.from_pretrained(florence_model_name, trust_remote_code=True).to(device)
florence_model.eval()

# ==========================================
# 3. HÀM KIỂM TRA BẰNG FLORENCE-2
# ==========================================
def get_florence_score(image_path, required_objects):
    """
    Yêu cầu Florence-2 tìm các vật thể và chấm điểm dựa trên tỷ lệ tìm thấy.
    """
    if not required_objects:
        return 1.0
        
    # Ghép các vật thể thành một cụm văn bản
    text_input = " and ".join(required_objects)
    
    # Task Grounding: Yêu cầu mô hình gắn tọa độ cho các từ khóa
    prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {text_input}"
    
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            
        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = florence_processor.post_process_generation(
            generated_text, 
            task="<CAPTION_TO_PHRASE_GROUNDING>", 
            image_size=(image.width, image.height)
        )
        
        # Lấy danh sách các vật thể mà Florence-2 thực sự tìm thấy trong ảnh
        results = parsed_answer.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
        labels_found = results.get('labels', [])
        
        if not labels_found:
            return 0.0
            
        # Tính điểm dựa trên số lượng từ khóa được tìm thấy
        unique_labels_found = set(labels_found)
        score = len(unique_labels_found) / len(required_objects)
        
        return min(score, 1.0) # Đảm bảo điểm tối đa không vượt quá 1.0
        
    except Exception as e:
        print(f"Lỗi Florence-2 đọc ảnh tại {image_path}: {e}")
        return 0.0

# ==========================================
# 4. HÀM TÌM KIẾM VÀ RE-RANK KẾT HỢP (SIGLIP + FLORENCE-2)
# ==========================================
def search_kis_with_florence(client, parsed_query_data, top_k=5, alpha=0.7):
    clip_query = parsed_query_data["clip_query"]
    required_objects = parsed_query_data["required_objects"]
    
    print(f"\n[SigLIP] Đang mã hóa câu truy vấn: '{clip_query}'")
    
    # Mã hóa văn bản bằng SigLIP
    text_inputs = siglip_processor(
        text=[clip_query], 
        padding="max_length", 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        text_features = siglip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_vector = text_features[0].cpu().tolist()

    # BƯỚC 1: LỌC THÔ BẰNG SIGLIP (Top 50)
    print("[Milvus] Đang truy xuất Top 50 khung hình gần giống nhất...") 
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[text_vector],
        limit=50, 
        output_fields=["video_id", "frame_id"], # Đã sửa lỗi chính tả ở đây
        search_params={"metric_type": "IP"}
    )

    all_hits = []
    for hits in search_results:
        for hit in hits:
            all_hits.append(hit)
            
    # BƯỚC 2: CHẤM ĐIỂM KẾT HỢP (ENSEMBLE SCORING)
    print(f"[Florence-2 & Re-rank] Đang quét và chấm điểm hỗn hợp cho: {required_objects}...")

    scored_results = []
    
    for hit in tqdm(all_hits, desc="Tiến độ xử lý", unit="ảnh"):
        v_id = hit["entity"]["video_id"]
        f_id = hit["entity"]["frame_id"]
        clip_score = hit["distance"] # Điểm Similarity của SigLIP
        
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
        
        if not image_path:
            continue
            
        # Lấy điểm Florence-2 score
        florence_score = get_florence_score(image_path, required_objects)

        # Điểm tổng hợp = alpha * điểm_SigLIP + (1 - alpha) * điểm_Florence
        final_score = alpha * clip_score + (1 - alpha) * florence_score
        
        # Lưu lại thông tin để sort
        scored_results.append({
            "hit": hit,
            "clip_score": clip_score,
            "florence_score": florence_score,
            "final_score": final_score
        })
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            print(f"video_id = {v_id}, frame_id = {f_id} | Điểm tổng: {item['final_score']:.4f} (SigLIP: {item['clip_score']:.4f}, Florence: {item['florence_score']:.4f})")

# ==========================================
# 5. CHẠY CHÍNH
# ==========================================
if __name__ == "__main__":
    print("Đang kết nối vào Database...")
    try:
        milvus_client = MilvusClient(MILVUS_DB_PATH)
        milvus_client.load_collection(COLLECTION_NAME) 
        
        raw_query = input("M kiếm gì?\nT kiếm: ")
        parsed_query_data = analyze_query_offline_mt(raw_query)
        
        # Chạy tìm kiếm với alpha = 0.7 (70% ưu tiên SigLIP, 30% Florence-2 điều chỉnh)
        search_kis_with_florence(milvus_client, parsed_query_data, top_k=5, alpha=0.7)
        
    except Exception as e:
        print(f"Lỗi! Chi tiết: {e}")
