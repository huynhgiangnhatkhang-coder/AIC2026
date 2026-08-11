import torch
import clip 
from pymilvus import MilvusClient
from translate import analyze_query_offline_mt  # Import hàm từ file translate.py của bạn

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ
# ==========================================
MILVUS_DB_PATH = "aic_kis_database.db"
COLLECTION_NAME = "kis_keyframes"

# ==========================================
# 2. HÀM TÌM KIẾM VÀ TRUY XUẤT CLIP (KHÔNG LỌC OBJECT)
# ==========================================
def search_kis(client, clip_query, top_k=5):
    print("Đang tải Text Encoder (OpenAI CLIP gốc)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Mã hóa câu truy vấn tiếng Anh bằng CLIP
    text_inputs = clip.tokenize([clip_query], truncate=True).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        # Chuẩn hóa L2
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_vector = text_features[0].cpu().tolist()

    print(f"Đang truy xuất Top {top_k} trực tiếp từ Milvus...") 
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[text_vector],
        limit=top_k, # Lấy đúng top_k kết quả cao nhất
        output_fields=["video_id", "frame_id"],
        search_params={"metric_type": "IP"}
    )

    print("\n=== KẾT QUẢ TÌM KIẾM CHÍNH THỨC ===")
    for hits in search_results:
        for hit in hits:
            v_id = hit["entity"]["video_id"]
            f_id = hit["entity"]["frame_id"]
            score = hit["distance"]
            print(f"video_id = {v_id}, frame_id = {f_id} (Độ tương đồng CLIP: {score:.4f})")

# ==========================================
# 3. CHẠY CHÍNH
# ==========================================
if __name__ == "__main__":
    print("Đang kết nối vào Database...")
    try:
        milvus_client = MilvusClient(MILVUS_DB_PATH)
        milvus_client.load_collection(COLLECTION_NAME) 
        
        # 1. Nhập câu truy vấn tiếng Việt thô từ Ban Giám Khảo
        raw_query = "Cây xanh, vịt, đồng cỏ"
        
        # 2. Xử lý ngôn ngữ tự nhiên Offline (Dịch + Trích xuất object)
        # Dù hàm trả về cả required_objects, ta chỉ lấy clip_query để dùng
        parsed_query_data = analyze_query_offline_mt(raw_query)
        clip_query = parsed_query_data["clip_query"]
        
        # 3. Tìm kiếm bằng CLIP
        search_kis(milvus_client, clip_query, top_k=5)
        
    except Exception as e:
        print(f"Lỗi! Chi tiết: {e}")