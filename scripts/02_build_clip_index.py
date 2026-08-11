"""
AIC 2026 Baseline — Script 02: Build CLIP FAISS Index
======================================================
Đọc clip_features.npy đã precomputed, normalize và build FAISS index.

Usage:
    python scripts/02_build_clip_index.py --config config.yaml

QUAN TRỌNG:
  - clip_features.npy đã được cung cấp sẵn bởi BTC (ViT-B/32)
  - Thứ tự vector trong .npy = thứ tự tăng dần của frame_map.json
  - Nếu chưa có .npy, script sẽ tự encode từ Keyframes/
"""
import os
import json
import argparse
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_faiss_index_from_npy(npy_path: str, index_path: str):
    """
    Load precomputed CLIP features từ .npy và build FAISS IVFFlat index.
    """
    print(f"[FAISS] Loading clip_features.npy: {npy_path}")
    features = np.load(npy_path).astype(np.float32)

    n, dim = features.shape
    print(f"[FAISS] Shape: {n} frames x {dim} dims")

    # L2-normalize để dùng cosine similarity = inner product
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    features = features / norms

    # Chọn index type: IVFFlat cho tìm kiếm nhanh trên tập lớn
    # nlist = số cluster centers (~sqrt(n) là heuristic tốt)
    nlist = max(100, int(np.sqrt(n)))
    nlist = min(nlist, n // 10)  # không vượt quá n/10

    print(f"[FAISS] Building IVFFlat index | dim={dim}, nlist={nlist}")
    quantizer = faiss.IndexFlatIP(dim)  # Inner Product
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train index
    print("[FAISS] Training...")
    index.train(features)

    # Add vectors
    print("[FAISS] Adding vectors...")
    index.add(features)

    index.nprobe = 50  # default nprobe khi search
    print(f"[FAISS] Index total: {index.ntotal} vectors")

    # Lưu index
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"[FAISS] Saved index -> {index_path}")


def build_faiss_index_from_keyframes(keyframes_dir: str, frame_map_path: str,
                                      index_path: str, model_name: str, device: str):
    """
    Encode từng keyframe bằng CLIP model (nếu không có sẵn .npy).
    """
    import clip
    import torch
    from PIL import Image

    print(f"[CLIP] Loading model: {model_name} on {device}")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    with open(frame_map_path, "r", encoding="utf-8") as f:
        frame_map = json.load(f)

    print(f"[CLIP] Encoding {len(frame_map)} frames...")
    all_features = []
    batch_size = 64

    for i in tqdm(range(0, len(frame_map), batch_size)):
        batch = frame_map[i:i+batch_size]
        images = []
        for fm in batch:
            img_path = fm["frame_path"]
            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
                images.append(img)
            except Exception as e:
                print(f"  [WARN] Lỗi đọc {img_path}: {e}")
                # Thay bằng tensor 0
                images.append(torch.zeros(3, 224, 224))

        image_input = torch.stack(images).to(device)
        with torch.no_grad():
            features = model.encode_image(image_input)
        features = features.cpu().numpy().astype(np.float32)
        all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)
    print(f"[CLIP] Encoded shape: {all_features.shape}")

    # Lưu .npy
    npy_out = os.path.join(os.path.dirname(index_path), "clip_features_encoded.npy")
    np.save(npy_out, all_features)
    print(f"[CLIP] Saved features -> {npy_out}")

    # Build FAISS
    n, dim = all_features.shape
    norms = np.linalg.norm(all_features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    all_features = all_features / norms

    nlist = max(100, min(int(np.sqrt(n)), n // 10))
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(all_features)
    index.add(all_features)

    faiss.write_index(index, index_path)
    print(f"[FAISS] Saved index -> {index_path}")


def main():
    parser = argparse.ArgumentParser(description="AIC2026 — Build CLIP FAISS Index")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--from-npy", action="store_true", default=True,
                        help="Build từ clip_features.npy đã có (mặc định)")
    parser.add_argument("--from-images", action="store_true",
                        help="Encode từ Keyframes/ (nếu không có .npy)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"])
    npy_path = root / cfg["data"]["clip_features_npy"]
    index_path = cfg["index"]["faiss_index_path"]
    frame_map_path = cfg["index"]["frame_map_path"]

    if args.from_images:
        print("[Mode] Encoding từ Keyframes/ images")
        build_faiss_index_from_keyframes(
            keyframes_dir=str(root / cfg["data"]["keyframes_dir"]),
            frame_map_path=frame_map_path,
            index_path=index_path,
            model_name=cfg["clip"]["model_name"],
            device=cfg["clip"]["device"]
        )
    else:
        # Kiểm tra .npy tồn tại
        if not npy_path.exists():
            print(f"[ERROR] clip_features.npy không tồn tại tại: {npy_path}")
            print("  → Dùng --from-images để encode từ Keyframes/")
            return

        build_faiss_index_from_npy(str(npy_path), index_path)

    print("\n[Done] FAISS index đã được build xong!")
    print(f"  Index: {index_path}")
    print(f"  Frame map: {frame_map_path}")
    print("\nBước tiếp theo:")
    print("  python scripts/03_build_bm25_index.py --config config.yaml")


if __name__ == "__main__":
    main()
