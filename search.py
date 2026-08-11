import os
import torch
import clip 
from PIL import Image
from pymilvus import MilvusClient
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from translate import analyze_query_offline_mt  # Import hàm từ file translate.py của bạn

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ
# ==========================================
MILVUS_DB_PATH = "aic_kis_database.db"
COLLECTION_NAME = "kis_keyframes"

# ĐƯỜNG DẪN TỚI THƯ MỤC CHỨA ẢNH GỐC (Trỏ thẳng vào thư mục DATASET của bạn)
KEYFRAMES_DIR = "DATASET"

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH (CLIP & GROUNDING DINO)
# ==========================================
print("Đang tải các mô hình AI lên bộ nhớ (CLIP & Grounding DINO)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Load Grounding DINO
dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

# ==========================================
# 3. HÀM KIỂM TRA VẬT THỂ BẰNG GROUNDING DINO
# ==========================================
def check_objects_with_dino(image_path, required_objects, threshold=0.25):
    # Nếu không có object nào bắt buộc, cho qua luôn
    if not required_objects:
        return True, 1.0
        
    # Chuyển đổi list ["Tree", "Duck"] thành format DINO: "tree . duck ."
    dino_query = " . ".join(required_objects).lower() + " ."
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Đưa ảnh và text vào DINO
        inputs = dino_processor(images=image, text=dino_query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dino_model(**inputs)
            
        # Lấy điểm số tự tin (confidence scores)
        target_sizes = torch.tensor([image.size[::-1]])
        results = dino_processor.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        
        # Nếu tìm thấy ít nhất 1 bounding box thỏa mãn, trả về True
        if len(results["scores"]) > 0:
            return True, results["scores"].max().item()
            
        return False, 0.0
    except Exception as e:
        print(f"Lỗi đọc ảnh tại {image_path}: {e}")
        return False, 0.0

# ==========================================
# 4. HÀM TÌM KIẾM VÀ RE-RANK
# ==========================================
def search_kis_with_dino(client, parsed_query_data, top_k=5):
    clip_query = parsed_query_data["clip_query"]
    required_objects = parsed_query_data["required_objects"]
    
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

    # BƯỚC 2: LỌC TINH (RE-RANK) BẰNG GROUNDING DINO
    if required_objects:
        print(f"[DINO] Đang quét ảnh thật để tìm vật thể: {required_objects}...")
    else:
        print("[DINO] Không có vật thể bắt buộc, bỏ qua bước quét DINO...")

    final_top_k = []
    
    for hits in search_results:
        for hit in hits:
            v_id = hit["entity"]["video_id"]
            f_id = hit["entity"]["frame_id"]
            clip_score = hit["distance"]
            
            # 1. Lấy tên thư mục video (VD: L21_V010)
            video_folder = v_id.replace(".mp4", "")
            
            # 2. TỰ ĐỘNG SUY LUẬN THƯ MỤC BATCH (MỚI THÊM)
            # Từ "L21_V010" -> Lấy "L21" -> Tạo thành "Keyframes_L21"
            batch_prefix = video_folder.split("_")[0]  
            batch_folder_name = f"Keyframes_{batch_prefix}" 
            
            possible_formats = [
                f"{f_id}.jpg", f"{f_id:03d}.jpg", f"{f_id:04d}.jpg", 
                f"{f_id:05d}.jpg", f"{f_id:06d}.jpg",
                f"{f_id}.png", f"{f_id:04d}.png" 
            ]
            
            image_path = None
            for fmt in possible_formats:
                # 3. NỐI ĐƯỜNG DẪN CHUẨN XÁC VỚI CẤU TRÚC BÊN TRÁI MÀN HÌNH CỦA BẠN
                temp_path = os.path.join(KEYFRAMES_DIR, batch_folder_name, "keyframes", video_folder, fmt)
                if os.path.exists(temp_path):
                    image_path = temp_path
                    break
            
            # --- BẬT LOG BẮT BỆNH SỐ 1 ---
            if not image_path:
                print(f"❌ LỖI ĐƯỜNG DẪN: Không thấy {batch_folder_name}/keyframes/{video_folder}/...")
                continue
                
            # Hạ nhẹ threshold xuống 0.15 
            has_object, dino_score = check_objects_with_dino(image_path, required_objects, threshold=0.15)
            
            # --- BẬT LOG BẮT BỆNH SỐ 2 ---
            if has_object:
                print(f" -> 🟢 DINO thấy '{required_objects}' tại {video_folder}/{os.path.basename(image_path)} (Score: {dino_score:.2f})")
                final_top_k.append(hit)
            else:
                print(f" -> 🔴 DINO ĐÁNH TRƯỢT {video_folder}/{os.path.basename(image_path)} (Không thấy {required_objects})")
                
            if len(final_top_k) == top_k:
                break

    print("\n=== KẾT QUẢ TÌM KIẾM CHÍNH THỨC ===")
    if not final_top_k:
        print("Không có bức ảnh nào trong Top 30 vượt qua được bài kiểm tra của Grounding DINO!")
    else:
        for hit in final_top_k:
            v_id = hit["entity"]["video_id"]
            f_id = hit["entity"]["frame_id"]
            score = hit["distance"]
            print(f"video_id = {v_id}, frame_id = {f_id} (Độ tương đồng CLIP: {score:.4f})")

# ==========================================
# 5. CHẠY CHÍNH
# ==========================================
if __name__ == "__main__":
    print("Đang kết nối vào Database...")
    try:
        milvus_client = MilvusClient(MILVUS_DB_PATH)
        milvus_client.load_collection(COLLECTION_NAME) 
        
        # 1. Nhập câu truy vấn tiếng Việt thô từ Ban Giám Khảo
        raw_query = "Phụ nữ, áo đầm vàng"
        
        # 2. Xử lý ngôn ngữ tự nhiên Offline (Dịch + Trích xuất object)
        parsed_query_data = analyze_query_offline_mt(raw_query)
        
        # 3. Tìm kiếm bằng CLIP và DINO
        search_kis_with_dino(milvus_client, parsed_query_data, top_k=5)
        
    except Exception as e:
        print(f"Lỗi! Chi tiết: {e}")
