"""
AIC 2026 — Resume SigLIP Extraction (Optimized)
================================================
Trích xuất SigLIP cho L23-L30 và THÊM VÀO collection hiện có.
Tối ưu bằng DataLoader (multi-worker) + Mixed Precision (FP16).
"""
import os
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
SIGLIP_DB_PATH = "aic_kis_database_siglip.db"
SIGLIP_COLLECTION = "kis_keyframes_siglip"

# Chỉ xử lý L23-L30 (L21, L22 đã có trong DB)
KEYFRAME_DIRS = [
    "./DATASET/Keyframes_L23/keyframes",
    "./DATASET/Keyframes_L24/keyframes",
    "./DATASET/Keyframes_L25/keyframes",
    "./DATASET/Keyframes_L26/keyframes",
    "./DATASET/Keyframes_L27/keyframes",
    "./DATASET/Keyframes_L28/keyframes",
    "./DATASET/Keyframes_L29/keyframes",
    "./DATASET/Keyframes_L30/keyframes",
]

BATCH_SIZE = 256
NUM_WORKERS = 8
INSERT_BATCH_SIZE = 5000


class KeyframeDataset(Dataset):
    """Dataset đọc ảnh song song bằng nhiều CPU workers."""
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        
        # Processor trả về dict of tensors
        inputs = self.processor(images=img, return_tensors="pt")
        # Squeeze batch dim (DataLoader sẽ tự thêm lại)
        return {k: v.squeeze(0) for k, v in inputs.items()}, idx


def collate_fn(batch):
    """Custom collate: gộp inputs và giữ nguyên indices."""
    inputs_list, indices = zip(*batch)
    # Stack tất cả tensor fields
    batched = {}
    for key in inputs_list[0]:
        batched[key] = torch.stack([inp[key] for inp in inputs_list])
    return batched, list(indices)


def main():
    from transformers import AutoProcessor, AutoModel
    from pymilvus import MilvusClient

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"[SigLIP] Loading model: {SIGLIP_MODEL_NAME} on {device}...")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(device).eval()

    # Embedding dim
    with torch.no_grad():
        dummy = processor(images=Image.new("RGB", (224, 224)), return_tensors="pt").to(device)
        dummy_out = model.get_image_features(**dummy)
        if not isinstance(dummy_out, torch.Tensor):
            if hasattr(dummy_out, 'image_embeds'):
                dummy_out = dummy_out.image_embeds
            elif hasattr(dummy_out, 'pooler_output'):
                dummy_out = dummy_out.pooler_output
            else:
                dummy_out = dummy_out[0]
        embedding_dim = dummy_out.shape[1]
    print(f"[SigLIP] Embedding dim: {embedding_dim}")

    # Scan keyframes
    print(f"\n[SigLIP] Scanning keyframes from {len(KEYFRAME_DIRS)} directories...")
    all_records = []

    for kf_dir in KEYFRAME_DIRS:
        if not os.path.isdir(kf_dir):
            print(f"  [SKIP] Không tìm thấy: {kf_dir}")
            continue

        video_folders = sorted([
            d for d in os.listdir(kf_dir)
            if os.path.isdir(os.path.join(kf_dir, d))
        ])
        dir_count = 0
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
                dir_count += 1
        print(f"  {kf_dir}: {len(video_folders)} videos, {dir_count} frames")

    print(f"\n[SigLIP] Tổng frames cần xử lý: {len(all_records)}")
    if not all_records:
        print("[ERROR] Không có frame mới nào!")
        return

    # Setup DataLoader
    dataset = KeyframeDataset(all_records, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # Extract features với FP16
    print(f"[SigLIP] Extracting features (batch={BATCH_SIZE}, workers={NUM_WORKERS}, FP16)...")
    all_embeddings = np.zeros((len(all_records), embedding_dim), dtype=np.float32)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        for batch_inputs, batch_indices in tqdm(dataloader, desc="Extracting"):
            batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch_inputs.items()}
            features = model.get_image_features(**batch_inputs)

            if not isinstance(features, torch.Tensor):
                if hasattr(features, 'image_embeds'):
                    features = features.image_embeds
                elif hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                else:
                    features = features[0]

            features = features.float()  # back to fp32 for normalization
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            embeddings_np = features.cpu().numpy()

            for i, idx in enumerate(batch_indices):
                all_embeddings[idx] = embeddings_np[i]

    print(f"[SigLIP] Extracted: {all_embeddings.shape}")

    # Insert vào Milvus (KHÔNG drop collection cũ!)
    print(f"\n[Milvus] Kết nối: {SIGLIP_DB_PATH}")
    client = MilvusClient(SIGLIP_DB_PATH)

    existing_stats = client.get_collection_stats(SIGLIP_COLLECTION)
    print(f"[Milvus] Collection hiện có: {existing_stats.get('row_count', '?')} records")

    # Insert data
    print(f"[Milvus] Inserting {len(all_records)} NEW records...")
    for i in tqdm(range(0, len(all_records), INSERT_BATCH_SIZE), desc="Inserting"):
        batch_records = all_records[i:i + INSERT_BATCH_SIZE]
        batch_embeddings = all_embeddings[i:i + INSERT_BATCH_SIZE]

        data = [
            {
                "video_id": rec["video_id"],
                "frame_id": rec["frame_id"],
                "embedding": emb.tolist(),
            }
            for rec, emb in zip(batch_records, batch_embeddings)
        ]
        client.insert(collection_name=SIGLIP_COLLECTION, data=data)

    # Rebuild HNSW index (cần rebuild cho toàn bộ dữ liệu mới + cũ)
    print("[Milvus] Dropping old index and rebuilding HNSW...")
    client.release_collection(SIGLIP_COLLECTION)
    client.drop_index(SIGLIP_COLLECTION, index_name="embedding")
    
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="IP",
        index_type="HNSW",
        params={"M": 16, "efConstruction": 256}
    )
    client.create_index(collection_name=SIGLIP_COLLECTION, index_params=index_params)
    client.load_collection(SIGLIP_COLLECTION)

    final_stats = client.get_collection_stats(SIGLIP_COLLECTION)
    print(f"\n[Milvus] ✓ SigLIP Database hoàn tất!")
    print(f"         Collection : {SIGLIP_COLLECTION}")
    print(f"         Records    : {final_stats.get('row_count', '?')}")
    print(f"         DB path    : {SIGLIP_DB_PATH}")

    client.release_collection(SIGLIP_COLLECTION)


if __name__ == "__main__":
    main()
