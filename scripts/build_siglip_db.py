"""
AIC 2026 — Build SigLIP Milvus Database
========================================
Trích xuất SigLIP image embeddings từ keyframes và nạp vào Milvus Lite.

Usage:
    python scripts/build_siglip_db.py --config config.yaml
    python scripts/build_siglip_db.py --config config.yaml --force-rebuild
"""
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import torch
from PIL import Image


SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
SIGLIP_DB_PATH = "aic_kis_database_siglip.db"
SIGLIP_COLLECTION = "kis_keyframes_siglip"


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_siglip_db(keyframes_dirs, db_path, collection_name,
                    batch_size=32, insert_batch_size=5000):
    """
    Extract SigLIP features từ keyframe images và insert vào Milvus.
    """
    from transformers import AutoProcessor, AutoModel
    from pymilvus import MilvusClient, DataType

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SigLIP model
    print(f"[SigLIP] Loading model: {SIGLIP_MODEL_NAME} on {device}...")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(device).eval()

    # Lấy embedding dim
    with torch.no_grad():
        dummy = processor(images=Image.new("RGB", (224, 224)), return_tensors="pt").to(device)
        dummy_out = model.get_image_features(**dummy)
        embedding_dim = dummy_out.shape[1]
    print(f"[SigLIP] Embedding dim: {embedding_dim}")

    # Thu thập tất cả ảnh keyframe
    print(f"\n[SigLIP] Scanning keyframes from {len(keyframes_dirs)} directories...")
    all_records = []  # (video_id, frame_id, image_path)

    for kf_dir in keyframes_dirs:
        if not os.path.isdir(kf_dir):
            print(f"  [SKIP] Không tìm thấy: {kf_dir}")
            continue

        video_folders = sorted([
            d for d in os.listdir(kf_dir)
            if os.path.isdir(os.path.join(kf_dir, d))
        ])
        print(f"  {kf_dir}: {len(video_folders)} videos")

        for vf in video_folders:
            folder_path = os.path.join(kf_dir, vf)
            frames = glob.glob(os.path.join(folder_path, "*.jpg"))
            frames += glob.glob(os.path.join(folder_path, "*.png"))
            frames = sorted(frames, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            for fp in frames:
                frame_id = int(os.path.splitext(os.path.basename(fp))[0])
                all_records.append({
                    "video_id": f"{vf}.mp4",
                    "frame_id": frame_id,
                    "image_path": fp,
                })

    print(f"\n[SigLIP] Tổng frames: {len(all_records)}")
    if not all_records:
        print("[ERROR] Không có frame nào!")
        return

    # Extract features theo batch
    print(f"[SigLIP] Extracting features (batch_size={batch_size})...")
    all_embeddings = []

    for i in tqdm(range(0, len(all_records), batch_size), desc="Extracting"):
        batch = all_records[i:i + batch_size]
        images = []
        for rec in batch:
            try:
                img = Image.open(rec["image_path"]).convert("RGB")
                images.append(img)
            except Exception:
                # Tạo ảnh trống nếu lỗi
                images.append(Image.new("RGB", (224, 224)))

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy().astype(np.float32))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"[SigLIP] Extracted: {all_embeddings.shape}")

    # Setup Milvus
    print(f"\n[Milvus] Kết nối: {db_path}")
    client = MilvusClient(db_path)

    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        print(f"[Milvus] Đã xóa collection cũ: {collection_name}")

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="video_id", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="frame_id", datatype=DataType.INT64)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)

    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"[Milvus] Collection '{collection_name}' created | dim={embedding_dim}")

    # Insert data
    print(f"[Milvus] Inserting {len(all_records)} records...")
    for i in tqdm(range(0, len(all_records), insert_batch_size), desc="Inserting"):
        batch_records = all_records[i:i + insert_batch_size]
        batch_embeddings = all_embeddings[i:i + insert_batch_size]

        data = [
            {
                "video_id": rec["video_id"],
                "frame_id": rec["frame_id"],
                "embedding": emb.tolist(),
            }
            for rec, emb in zip(batch_records, batch_embeddings)
        ]
        client.insert(collection_name=collection_name, data=data)

    # Build HNSW index
    print("[Milvus] Building HNSW index...")
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="IP",
        index_type="HNSW",
        params={"M": 16, "efConstruction": 256}
    )
    client.create_index(collection_name=collection_name, index_params=index_params)

    client.load_collection(collection_name)

    stats = client.get_collection_stats(collection_name)
    print(f"\n[Milvus] ✓ SigLIP Database sẵn sàng!")
    print(f"         Collection : {collection_name}")
    print(f"         Records    : {stats.get('row_count', '?')}")
    print(f"         DB path    : {db_path}")

    client.release_collection(collection_name)


def main():
    parser = argparse.ArgumentParser(description="Build SigLIP Milvus Database")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)

    keyframes_dirs = cfg["data"].get("keyframes_dirs_list", [])
    if not keyframes_dirs:
        print("[ERROR] Chưa cấu hình keyframes_dirs_list trong config.yaml")
        return

    db_path = SIGLIP_DB_PATH
    collection_name = SIGLIP_COLLECTION

    if os.path.exists(db_path) and not args.force_rebuild:
        # Check if collection exists
        from pymilvus import MilvusClient
        client = MilvusClient(db_path)
        if client.has_collection(collection_name):
            stats = client.get_collection_stats(collection_name)
            print(f"[Info] DB đã tồn tại: {db_path}")
            print(f"       Collection: {collection_name}, Records: {stats.get('row_count', '?')}")
            print("  → Dùng --force-rebuild để build lại")
            return

    build_siglip_db(
        keyframes_dirs=keyframes_dirs,
        db_path=db_path,
        collection_name=collection_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
