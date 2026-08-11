import os
import glob
import numpy as np
from pymilvus import MilvusClient, DataType

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THÔNG SỐ
# ==========================================
NPY_DIRS = ["/home/khang/Downloads/clip-features-32-aic25-b1/clip-features-32"]  
KEYFRAMES_DIRS = ["/home/khang/Downloads/Keyframes_L21/keyframes"]          
MILVUS_DB_PATH = "aic_kis_database.db"
COLLECTION_NAME = "kis_keyframes"

# ==========================================
# 2. HÀM ĐỌC VÀ ÁNH XẠ DỮ LIỆU
# ==========================================
def load_data_and_mapping(npy_dirs, keyframes_dirs):
    print("Đang quét các thư mục và ghép nối đặc trưng...")
    data_to_insert = []
    if len(npy_dirs) != len(keyframes_dirs):
        print("Lỗi: Số lượng thư mục trong NPY_DIRS và KEYFRAMES_DIRS không khớp nhau!")
        return []

    for batch_idx, (npy_dir, keyframes_dir) in enumerate(zip(npy_dirs, keyframes_dirs)):
        print(f"\n--- Đang xử lý Tập dữ liệu {batch_idx + 1} ---")
        print(f"Thư mục ảnh: {keyframes_dir}")
        print(f"Thư mục vector: {npy_dir}")
        
        try:
            video_folders = sorted(os.listdir(keyframes_dir))
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy thư mục {keyframes_dir}. Bỏ qua tập này.")
            continue

        for video_folder in video_folders:
            folder_path = os.path.join(keyframes_dir, video_folder)
            if not os.path.isdir(folder_path):
                continue
                
            npy_filename = f"{video_folder}.npy"
            npy_path = os.path.join(npy_dir, npy_filename)
            
            if not os.path.exists(npy_path):
                continue
            image_features = np.load(npy_path).astype(np.float32)
            
            # Chuẩn hóa L2
            image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
            
            # Sắp xếp ảnh theo thứ tự số học
            frames = glob.glob(os.path.join(folder_path, "*.jpg"))
            frames = sorted(frames, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            
            if len(frames) != len(image_features):
                min_len = min(len(frames), len(image_features))
                frames = frames[:min_len]
                image_features = image_features[:min_len]

            for i, frame_path in enumerate(frames):
                video_id = f"{video_folder}.mp4"
                frame_filename = os.path.basename(frame_path)
                frame_id = int(os.path.splitext(frame_filename)[0])
                
                record = {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "embedding": image_features[i].tolist()
                }
                data_to_insert.append(record)
                
    print(f"\nHoàn tất! Đã ánh xạ (map) thành công tổng cộng {len(data_to_insert)} khung hình từ tất cả các tập.")
    return data_to_insert

# ==========================================
# 3. KHỞI TẠO MILVUS & IMPORT DATA
# ==========================================
def setup_milvus_and_insert(data):
    print("\nĐang kết nối Milvus Lite...")
    client = MilvusClient(MILVUS_DB_PATH)
    
    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="video_id", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="frame_id", datatype=DataType.INT64)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=512) 

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    
    print("Đang chèn dữ liệu vào cơ sở dữ liệu...")
    client.insert(collection_name=COLLECTION_NAME, data=data)

    print("Đang đánh chỉ mục (Indexing)...")
    index_params = client.prepare_index_params()
    
    index_params.add_index(
        field_name="embedding", 
        metric_type="IP", 
        index_type="FLAT" 
    )
    client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
    client.load_collection(COLLECTION_NAME)
    
    print("Database đã sẵn sàng!")

# ==========================================
if __name__ == "__main__":
    # Đảm bảo xóa database cũ (nếu có) trước khi tạo mới để tránh lỗi
    if os.path.exists(MILVUS_DB_PATH):
        os.remove(MILVUS_DB_PATH)
        print(f"Đã xóa database cũ: {MILVUS_DB_PATH}")

    data = load_data_and_mapping(NPY_DIRS, KEYFRAMES_DIRS)
    if data:
        setup_milvus_and_insert(data)