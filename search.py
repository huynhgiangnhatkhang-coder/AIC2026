import os
import torch
import clip 
from PIL import Image
from pymilvus import MilvusClient
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from translate import analyze_query_offline_mt 

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (Giữ nguyên)
# ==========================================
MILVUS_DB_PATH = "aic_kis_database.db"
COLLECTION_NAME = "kis_keyframes"
KEYFRAMES_DIR = "/home/khang/Documents/PROJECT/AIC2026/DATASET"

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH (Giữ nguyên)
# ==========================================
print("Đang tải các mô hình AI lên bộ nhớ (CLIP & Grounding DINO)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

# ==========================================
# 3. HÀM KIỂM TRA BẰNG GROUNDING DINO (ĐÃ TỐI ƯU Ý 1)
# ==========================================
def get_dino_score(image_path, text_prompt):
    """
    Sử dụng toàn bộ cụm từ (VD: 'A woman in a blue shirt') để lấy điểm chính xác nhất.
    Không dùng threshold cứng để loại bỏ, chỉ trả về điểm số cao nhất tìm được.
    """
    # DINO yêu cầu prompt kết thúc bằng dấu chấm và viết thường
    dino_query = text_prompt.lower()
    if not dino_query.endswith("."):
        dino_query += " ."
        
    try:
        image = Image.open(image_path).convert("RGB")
        
        inputs = dino_processor(images=image, text=dino_query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
            
        target_sizes = torch.tensor([image.size[::-1]])
        
        # Hạ threshold xuống mức tối thiểu (0.05) để bắt mọi bounding box tiềm năng, 
        # ta lấy điểm (score) thay vì quan tâm box đó có qua ngưỡng hay không.
        results = dino_processor.image_processor.post_process_object_detection(
            outputs, threshold=0.05, target_sizes=target_sizes
        )[0]
        
        if len(results["scores"]) > 0:
            return results["scores"].max().item() # Trả về điểm tự tin cao nhất
            
        return 0.0
    except Exception as e:
        print(f"Lỗi đọc ảnh tại {image_path}: {e}")
        return 0.0

# ==========================================
# 4. HÀM TÌM KIẾM VÀ RE-RANK (ĐÃ TỐI ƯU Ý 2)
# ==========================================
def search_kis_with_dino(client, parsed_query_data, top_k=5, alpha=0.75):
    """
    alpha: Trọng số của CLIP score (0.75 nghĩa là CLIP chiếm 75%, DINO chiếm 25%).
    Bạn có thể tinh chỉnh thông số này tùy thuộc vào độ tin cậy của CLIP vs DINO.
    """
    clip_query = parsed_query_data["clip_query"]
    
    print(f"\n[CLIP] Đang mã hóa câu truy vấn: '{clip_query}'")
    text_inputs = clip.tokenize([clip_query], truncate=True).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_vector = text_features[0].cpu().tolist()

    # BƯỚC 1: LỌC THÔ BẰNG CLIP (LẤY TOP 30)
    print("[Milvus] Đang truy xuất Top 30 khung hình gần giống nhất...") 
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[text_vector],
        limit=30, 
        output_fields=["video_id", "frame_id"],
        search_params={"metric_type": "IP"}
    )

    print(f"[DINO] Đang quét ảnh và chấm điểm tổng hợp (Ensemble Scoring)...")
    
    scored_frames = []
    
    # search_results trả về list of hits cho query đầu tiên [0]
    for hit in search_results[0]: 
        v_id = hit["entity"]["video_id"]
        f_id = hit["entity"]["frame_id"]
        clip_score = hit["distance"] # Dải điểm IP của CLIP (thường từ 0.15 - 0.45)
        
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
            print(f"❌ Không tìm thấy đường dẫn: {video_folder} - Frame {f_id}")
            continue
            
        # Tính điểm DINO dựa trên toàn bộ câu query (Ý 1)
        dino_score = get_dino_score(image_path, text_prompt=clip_query)
        
        # Tính điểm tổng hợp (Ý 2)
        final_score = (alpha * clip_score) + ((1 - alpha) * dino_score)
        
        scored_frames.append({
            "video_id": v_id,
            "frame_id": f_id,
            "clip_score": clip_score,
            "dino_score": dino_score,
            "final_score": final_score
        })

    # BƯỚC 3: SẮP XẾP LẠI (RE-RANK) DỰA TRÊN FINAL SCORE
    scored_frames = sorted(scored_frames, key=lambda x: x["final_score"], reverse=True)
    final_top_k = scored_frames[:top_k]

    print("\n=== KẾT QUẢ TÌM KIẾM CHÍNH THỨC ===")
    for rank, frame in enumerate(final_top_k, 1):
        print(f"Top {rank}: video_id = {frame['video_id']}, frame_id = {frame['frame_id']}")
        print(f"    -> Tổng điểm: {frame['final_score']:.4f} (CLIP: {frame['clip_score']:.4f} | DINO: {frame['dino_score']:.4f})")

# ==========================================
# 5. CHẠY CHÍNH (Giữ nguyên)
# ==========================================
if __name__ == "__main__":
    print("Đang kết nối vào Database...")
    try:
        milvus_client = MilvusClient(MILVUS_DB_PATH)
        milvus_client.load_collection(COLLECTION_NAME) 
        
        raw_query = "Phóng viên thời sự mặc áo đầm màu trắng"
        parsed_query_data = analyze_query_offline_mt(raw_query)
        
        # Thử nghiệm với alpha=0.75 (Tôn trọng ngữ cảnh CLIP nhiều hơn)
        search_kis_with_dino(milvus_client, parsed_query_data, top_k=5, alpha=0.75)
        
    except Exception as e:
        print(f"Lỗi! Chi tiết: {e}")