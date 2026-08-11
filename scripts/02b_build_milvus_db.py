"""
AIC 2026 Baseline — Script 02b: Build Milvus Lite Database
============================================================
Tích hợp code xây dựng database Milvus từ per-video .npy files.
Thay thế / bổ sung cho FAISS index (script 02_build_clip_index.py).

Ưu điểm so với FAISS:
  - Lưu trực tiếp video_id + frame_id → không cần frame_map.json
  - Hỗ trợ filter theo video_id khi search
  - Dễ cập nhật thêm batch mới mà không cần rebuild toàn bộ
  - Milvus Lite: chạy local, không cần server

Cấu trúc dữ liệu kỳ vọng:
  npy_dir/
    L01_V001.npy     ← CLIP embeddings cho tất cả frame của video L01_V001
    L01_V002.npy
    ...
  keyframes_dir/
    L01_V001/
      0000.jpg       ← frame_id = 0
      0005.jpg       ← frame_id = 5
      ...

Usage:
  python scripts/02b_build_milvus_db.py --config config.yaml
  python scripts/02b_build_milvus_db.py --config config.yaml --force-rebuild
"""
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==========================================
# PHẦN 1: ĐỌC VÀ ÁNH XẠ DỮ LIỆU
# (từ code của user, đã refactor + thêm logging)
# ==========================================

def load_data_and_mapping(npy_dirs: List[str], keyframes_dirs: List[str]) -> List[Dict]:
    """
    Đọc per-video .npy files và map với frame filenames.

    Args:
        npy_dirs:       list thư mục chứa per-video .npy files
        keyframes_dirs: list thư mục Keyframes (khớp 1-1 với npy_dirs)

    Returns:
        List[Dict] với mỗi dict là:
        {
            "video_id":  str  (vd: "L01_V001.mp4"),
            "frame_id":  int  (frame index thực, từ tên file jpg),
            "embedding": list[float]  (vector 512-d, đã L2-normalize)
        }
    """
    if len(npy_dirs) != len(keyframes_dirs):
        raise ValueError(
            f"Số lượng npy_dirs ({len(npy_dirs)}) và keyframes_dirs "
            f"({len(keyframes_dirs)}) phải bằng nhau!"
        )

    print("=" * 60)
    print("  Đang quét thư mục và ghép nối đặc trưng CLIP...")
    print("=" * 60)

    data_to_insert = []
    total_skipped = 0
    total_mismatch = 0

    for batch_idx, (npy_dir, keyframes_dir) in enumerate(zip(npy_dirs, keyframes_dirs)):
        print(f"\n[Batch {batch_idx + 1}/{len(npy_dirs)}]")
        print(f"  Keyframes : {keyframes_dir}")
        print(f"  NPY files : {npy_dir}")

        if not os.path.isdir(keyframes_dir):
            print(f"  [ERROR] Không tìm thấy: {keyframes_dir} — Bỏ qua.")
            continue
        if not os.path.isdir(npy_dir):
            print(f"  [ERROR] Không tìm thấy: {npy_dir} — Bỏ qua.")
            continue

        video_folders = sorted(os.listdir(keyframes_dir))
        video_folders = [v for v in video_folders
                         if os.path.isdir(os.path.join(keyframes_dir, v))]

        print(f"  Số video: {len(video_folders)}")
        batch_count = 0

        for video_folder in tqdm(video_folders, desc=f"  Batch {batch_idx + 1}"):
            folder_path = os.path.join(keyframes_dir, video_folder)

            # Tìm file .npy tương ứng
            npy_path = os.path.join(npy_dir, f"{video_folder}.npy")
            if not os.path.exists(npy_path):
                total_skipped += 1
                continue

            # Load và normalize embeddings
            try:
                image_features = np.load(npy_path).astype(np.float32)
            except Exception as e:
                print(f"\n  [WARN] Lỗi đọc {npy_path}: {e}")
                total_skipped += 1
                continue

            # L2-normalize (để dùng Inner Product = cosine similarity)
            norms = np.linalg.norm(image_features, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)  # tránh chia 0
            image_features = image_features / norms

            # Lấy danh sách frames, sort theo số học (0 < 5 < 10, không phải "0" < "10" < "5")
            frames = glob.glob(os.path.join(folder_path, "*.jpg"))
            frames += glob.glob(os.path.join(folder_path, "*.png"))
            frames = sorted(
                frames,
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            )

            # Xử lý mismatch số frame vs số vector
            if len(frames) != len(image_features):
                total_mismatch += 1
                min_len = min(len(frames), len(image_features))
                frames = frames[:min_len]
                image_features = image_features[:min_len]

            # Tạo records
            for i, frame_path in enumerate(frames):
                frame_filename = os.path.basename(frame_path)
                frame_id = int(os.path.splitext(frame_filename)[0])

                data_to_insert.append({
                    "video_id": f"{video_folder}.mp4",
                    "frame_id": frame_id,
                    "frame_filename": frame_filename,       # metadata bổ sung
                    "frame_path": frame_path,               # metadata bổ sung (không lưu Milvus)
                    "embedding": image_features[i].tolist()
                })
                batch_count += 1

        print(f"  ✓ {batch_count} frames từ batch này")

    print(f"\n{'='*60}")
    print(f"  Tổng frame đã map : {len(data_to_insert)}")
    print(f"  Video bị bỏ qua   : {total_skipped} (không có .npy)")
    print(f"  Video bị mismatch : {total_mismatch} (số frame ≠ số vector → lấy min)")
    print(f"{'='*60}")

    return data_to_insert


# ==========================================
# PHẦN 2: KHỞI TẠO MILVUS & IMPORT DATA
# (từ code của user + thêm batching, validation)
# ==========================================

def setup_milvus_and_insert(data: List[Dict], db_path: str,
                             collection_name: str,
                             embedding_dim: int = 512,
                             batch_size: int = 10000):
    """
    Tạo Milvus Lite collection và insert dữ liệu theo batch.

    Args:
        data:            list records từ load_data_and_mapping()
        db_path:         đường dẫn file .db Milvus Lite
        collection_name: tên collection
        embedding_dim:   chiều vector (512 cho CLIP ViT-B/32)
        batch_size:      số record insert mỗi lần (tránh OOM)
    """
    from pymilvus import MilvusClient, DataType

    print(f"\n[Milvus] Kết nối Milvus Lite: {db_path}")
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    client = MilvusClient(db_path)

    # Xóa collection cũ nếu có
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        print(f"[Milvus] Đã xóa collection cũ: {collection_name}")

    # Định nghĩa schema
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id",             datatype=DataType.INT64,
                     is_primary=True)
    schema.add_field(field_name="video_id",        datatype=DataType.VARCHAR,
                     max_length=200)
    schema.add_field(field_name="frame_id",        datatype=DataType.INT64)
    schema.add_field(field_name="frame_filename",  datatype=DataType.VARCHAR,
                     max_length=100)
    schema.add_field(field_name="embedding",       datatype=DataType.FLOAT_VECTOR,
                     dim=embedding_dim)

    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"[Milvus] Collection '{collection_name}' đã tạo | dim={embedding_dim}")

    # Insert theo batch
    print(f"[Milvus] Inserting {len(data)} records (batch_size={batch_size})...")
    total_inserted = 0

    # Chỉ lấy các fields Milvus cần (bỏ frame_path)
    milvus_data = [
        {k: v for k, v in record.items() if k != "frame_path"}
        for record in data
    ]

    for i in tqdm(range(0, len(milvus_data), batch_size), desc="Inserting"):
        batch = milvus_data[i:i + batch_size]
        client.insert(collection_name=collection_name, data=batch)
        total_inserted += len(batch)

    print(f"[Milvus] Inserted: {total_inserted} records")

    # Tạo HNSW index (nhanh hơn FLAT cho dataset lớn)
    print("[Milvus] Đang tạo index HNSW...")
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="IP",                   # Inner Product = Cosine (sau khi L2-normalize)
        index_type="HNSW",                  # HNSW nhanh hơn FLAT trên dataset lớn
        params={"M": 16, "efConstruction": 256}
    )
    client.create_index(collection_name=collection_name, index_params=index_params)

    # Load collection vào memory
    client.load_collection(collection_name)

    # Verify
    count = client.get_collection_stats(collection_name)
    print(f"[Milvus] ✓ Database sẵn sàng!")
    print(f"         Collection : {collection_name}")
    print(f"         Tổng records: {total_inserted}")
    print(f"         DB path     : {db_path}")

    client.release_collection(collection_name)  # giải phóng memory sau khi build


# ==========================================
# PHẦN 3: XÁC ĐỊNH CHIỀU VECTOR TỰ ĐỘNG
# ==========================================

def detect_embedding_dim(npy_dirs: List[str], keyframes_dirs: List[str]) -> int:
    """
    Đọc 1 file .npy mẫu để xác định embedding dimension.
    """
    for npy_dir, kf_dir in zip(npy_dirs, keyframes_dirs):
        if not os.path.isdir(npy_dir):
            continue
        for fname in os.listdir(npy_dir):
            if fname.endswith(".npy"):
                npy_path = os.path.join(npy_dir, fname)
                try:
                    arr = np.load(npy_path)
                    dim = arr.shape[1] if arr.ndim == 2 else arr.shape[0]
                    print(f"[Detect] Embedding dim = {dim} (từ {fname})")
                    return dim
                except Exception:
                    pass
    print("[Detect] Không xác định được dim, dùng mặc định 512")
    return 512


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="AIC2026 — Build Milvus Lite Database từ per-video .npy files"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Xóa DB cũ và build lại từ đầu")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Số record insert mỗi lần (mặc định: 10000)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Lấy đường dẫn từ config
    npy_dirs = cfg["data"].get("npy_dirs", [])
    keyframes_dirs = cfg["data"].get("keyframes_dirs_list", [])

    if not npy_dirs or not keyframes_dirs:
        print("[ERROR] Chưa cấu hình npy_dirs / keyframes_dirs_list trong config.yaml")
        print("  → Mở config.yaml và chỉnh section data.npy_dirs")
        return

    db_path = cfg["index"]["milvus_db_path"]
    collection_name = cfg["index"]["milvus_collection"]

    # Xóa DB cũ nếu force rebuild
    if args.force_rebuild and os.path.exists(db_path):
        os.remove(db_path)
        print(f"[Rebuild] Đã xóa DB cũ: {db_path}")

    # Kiểm tra DB đã tồn tại chưa
    if os.path.exists(db_path) and not args.force_rebuild:
        print(f"[Info] DB đã tồn tại: {db_path}")
        print("  → Dùng --force-rebuild để build lại")
        print("  → Tiếp tục sang bước BM25...")
    else:
        # Bước 1: Load & map dữ liệu
        data = load_data_and_mapping(npy_dirs, keyframes_dirs)

        if not data:
            print("[ERROR] Không có dữ liệu để insert!")
            return

        # Tự động detect embedding dim
        embedding_dim = detect_embedding_dim(npy_dirs, keyframes_dirs)

        # Bước 2: Build Milvus DB
        setup_milvus_and_insert(
            data=data,
            db_path=db_path,
            collection_name=collection_name,
            embedding_dim=embedding_dim,
            batch_size=args.batch_size
        )

    print("\n✓ Xong! Bước tiếp theo:")
    print("  python scripts/03_build_bm25_index.py --config config.yaml")


if __name__ == "__main__":
    main()
