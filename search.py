import torch
import clip 
from pymilvus import MilvusClient

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ
# ==========================================
MILVUS_DB_PATH = "aic_kis_database.db"
COLLECTION_NAME = "kis_keyframes"

# ==========================================
# 2. HÀM TÌM KIẾM
# ==========================================
def search_kis(client, query_text, top_k=5):
    print(f"\nĐang xử lý truy vấn: '{query_text}'")
    
    print("Đang tải Text Encoder (OpenAI CLIP gốc)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    text_inputs = clip.tokenize([query_text], truncate=True).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        # Chuẩn hóa L2
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_vector = text_features[0].cpu().tolist()

    print("Đang tìm kiếm trong Database...")
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[text_vector],
        limit=top_k, 
        output_fields=["video_id", "frame_id"],
        search_params={"metric_type": "IP"}
    )

    print("\n=== KẾT QUẢ TÌM KIẾM ===")
    for hits in search_results:
        for hit in hits:
            v_id = hit["entity"]["video_id"]
            f_id = hit["entity"]["frame_id"]
            score = hit["distance"]
            print(f"video_id = {v_id}, frame_id = {f_id} (Độ tương đồng: {score:.4f})")

# ==========================================
if __name__ == "__main__":
    print("Đang kết nối vào Database có sẵn...")
    try:
        milvus_client = MilvusClient(MILVUS_DB_PATH)
        # Bắt buộc gọi load_collection để nạp index vào bộ nhớ trước khi search
        milvus_client.load_collection(COLLECTION_NAME) 
        
        # Nhập câu truy vấn của bạn tại đây
        query_text = "Night scene outdoors on a road with cargo container trucks, multiple police officers and security personnel in reflective safety vests standing and inspecting"
        
        search_kis(milvus_client, query_text, top_k=5)
    except Exception as e:
        print(f"Lỗi kết nối Database! Vui lòng chạy file build_db.py trước. Chi tiết lỗi: {e}")