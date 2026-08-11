import os
import glob
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from pymilvus import MilvusClient

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ
# ==========================================
MILVUS_DB_PATH = "aic_kis_database_siglip.db" 
COLLECTION_NAME = "kis_keyframes_siglip"
KEYFRAMES_DIR = "DATASET"

# Kích thước lô (Batch Size) phụ thuộc vào VRAM của bạn:
# VRAM 6GB - 8GB -> Để 32 - 64
# VRAM 12GB+ -> Có thể đẩy lên 128 - 256
BATCH_SIZE = 64 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def init_milvus_db():
    """Khởi tạo database Milvus với vector 768 chiều"""
    print("-> Đang khởi tạo Milvus Database...")
    client = MilvusClient(MILVUS_DB_PATH)
    
    # Xóa collection cũ nếu đã tồn tại để tránh lỗi trùng lặp dữ liệu
    if client.has_collection(COLLECTION_NAME):
        print(f"Phát hiện Collection '{COLLECTION_NAME}' đã tồn tại. Tiến hành làm mới...")
        client.drop_collection(COLLECTION_NAME)

    # Tạo Collection mới (Mặc định MilvusClient dùng tên trường là 'vector' và tự động tạo 'id')
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=768, # Chiều của SigLIP
        metric_type="IP", # Inner Product (Tích vô hướng) phù hợp nhất với SigLIP/CLIP
        auto_id=True,
        enable_dynamic_field=True # Bật để tự do thêm các trường (video_id, frame_id)
    )
    return client

def main():
    client = init_milvus_db()
    
    print("-> Đang tải mô hình Google SigLIP (768d)...")
    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval() # Bật chế độ suy luận để tiết kiệm bộ nhớ

    # Tìm toàn bộ ảnh trong thư mục DATASET
    print(f"-> Đang quét thư mục {DATASET_DIR}...")
    search_pattern = os.path.join(DATASET_DIR, "**", "*.[jp][pn]g")
    image_paths = glob.glob(search_pattern, recursive=True)
    
    if not image_paths:
        print("❌ Không tìm thấy bức ảnh nào. Vui lòng kiểm tra lại đường dẫn DATASET_DIR.")
        return
        
    print(f"-> Đã tìm thấy {len(image_paths)} bức ảnh. Bắt đầu trích xuất...")

    # Xử lý theo lô (Batch Processing)
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Tiến độ", unit="Lô"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = []
        metadata = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                
                # Bóc tách Tên thư mục cha (video_id) và Tên file (frame_id)
                # Ví dụ: DATASET/Keyframes_L21/keyframes/L21_V010/0150.jpg
                folder_name = os.path.basename(os.path.dirname(p)) # Sẽ ra "L21_V010"
                file_name = os.path.basename(p) # Sẽ ra "0150.jpg"
                
                # Lấy số của khung hình (VD: 150)
                frame_id = int(os.path.splitext(file_name)[0])
                
                # Định dạng lại giống hệt với code cũ của bạn
                metadata.append({
                    "video_id": f"{folder_name}.mp4",
                    "frame_id": frame_id
                })
            except Exception as e:
                print(f"\n[Cảnh báo] Bỏ qua ảnh lỗi {p}: {e}")

        if not images:
            continue

        # Đưa ảnh qua SigLIP
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # CHUẨN HÓA L2 (Bắt buộc để chấm điểm Cosine Similarity / IP chính xác)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            vectors = image_features.cpu().tolist()

        # Đóng gói dữ liệu để đưa vào Milvus
        insert_data = []
        for idx, vec in enumerate(vectors):
            data = metadata[idx]
            data["vector"] = vec # Trường bắt buộc của Milvus Lite
            insert_data.append(data)

        # Chèn vào Database
        client.insert(collection_name=COLLECTION_NAME, data=insert_data)

    print("\n✅ HOÀN TẤT! Toàn bộ vector 768 chiều đã được lưu vào database mới.")

if __name__ == "__main__":
    main()
