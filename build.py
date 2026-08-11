import os
import glob
import numpy as np
from pymilvus import MilvusClient, DataType
import shutil

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN TỔNG (TỰ ĐỘNG QUÉT)
# ==========================================
# THAY ĐỔI: Trỏ đến thư mục cha chứa tất cả các thư mục/file .npy
BASE_NPY_DIR = "../clip-features-32-aic25-b1/clip-features-32"  

# THAY ĐỔI: Trỏ đến thư mục cha chứa các folder "Keyframes_L21", "Keyframes_L22",...
BASE_KEYFRAMES_DIR = "../DATASET"          

MILVUS_DB_PATH = "aic_kis_database.db"
COLLECTION_NAME = "kis_keyframes"

# ==========================================
# 2. HÀM ĐỌC VÀ ÁNH XẠ DỮ LIỆU TỰ ĐỘNG
# ==========================================
def load_data_and_mapping(base_npy_dir, base_keyframes_dir):
    print("Đang chuẩn bị từ điển Vector...")
    data_to_insert = []
    
    # 1. Dùng os.walk để tìm tự động TẤT CẢ các file .npy trong thư mục gốc
    npy_map = {}
    for root, dirs, files in os.walk(base_npy_dir):
        for file in files:
            if file.endswith(".npy"):
                # Cắt bỏ đuôi .npy để lấy tên video (vd: L21_V001)
                video_folder_name = file.replace(".npy", "")
                npy_map[video_folder_name] = os.path.join(root, file)
                
    print(f"Đã tìm thấy sẵn sàng {len(npy_map)} file vector .npy.")

    # 2. Quét tự động các thư mục chứa ảnh
    print("\nĐang quét thư mục ảnh và tiến hành ghép nối...")
    if not os.path.exists(base_keyframes_dir):
        print(f"Lỗi: Không tìm thấy thư mục cha {base_keyframes_dir}.")
        return []

    # Lấy tất cả các folder bên trong thư mục cha (L21, L22,...)
    batch_folders = sorted(os.listdir(base_keyframes_dir))
    
    for batch_folder in batch_folders:
        # Bỏ qua nếu không phải là cấu trúc thư mục của BTC (Keyframes_Lxx)
        if not batch_folder.startswith("Keyframes_L"):
            continue
            
        # Đường dẫn tới thư mục "keyframes" bên trong
        keyframes_dir = os.path.join(base_keyframes_dir, batch_folder, "keyframes")
        
        if not os.path.exists(keyframes_dir):
            continue
            
        print(f"--- Đang xử lý tập: {batch_folder} ---")
        video_folders = sorted(os.listdir(keyframes_dir))
        
        for video_folder in video_folders:
            folder_path = os.path.join(keyframes_dir, video_folder)
            if not os.path.isdir(folder_path):
                continue
                
            # ĐỐI CHIẾU THÔNG MINH: Tự động tìm đường dẫn file .npy tương ứng
            if video_folder not in npy_map:
                # Nếu BTC thiếu file npy cho video này, tự động bỏ qua
                continue
                
            npy_path = npy_map[video_folder]
            
            # Đọc và chuẩn hóa vector
            image_features = np.load(npy_path).astype(np.float32)
            image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
            
            # Đọc và sắp xếp ảnh (đảm bảo đúng thứ tự số học 1, 2, 3...)
            frames = glob.glob(os.path.join(folder_path, "*.jpg"))
            frames = sorted(frames, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            
            # Khắc phục lỗi lệch số lượng nếu có
            if len(frames) != len(image_features):
                min_len = min(len(frames), len(image_features))
                frames = frames[:min_len]
                image_features = image_features[:min_len]

            for i, frame_path in enumerate(frames):
                video_id = f"{video_folder}.mp4"
                frame_id = int(os.path.splitext(os.path.basename(frame_path))[0])
                
                record = {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "embedding": image_features[i].tolist()
                }
                data_to_insert.append(record)
                
    print(f"\nHoàn tất! Đã ánh xạ (map) thành công tổng cộng {len(data_to_insert)} khung hình.")
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
    if os.path.exists(MILVUS_DB_PATH):
        if os.path.isdir(MILVUS_DB_PATH):
            shutil.rmtree(MILVUS_DB_PATH)
        else:
            os.remove(MILVUS_DB_PATH)
        print(f"Đã xóa database cũ: {MILVUS_DB_PATH}")

    data = load_data_and_mapping(BASE_NPY_DIR, BASE_KEYFRAMES_DIR)
    if data:
        setup_milvus_and_insert(data)
