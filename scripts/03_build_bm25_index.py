"""
AIC 2026 Baseline — Script 03: Build BM25 Index
================================================
Xây dựng BM25 corpus từ Metadata (title, description, tags)
để hỗ trợ text-based retrieval.

Usage:
    python scripts/03_build_bm25_index.py --config config.yaml
"""
import json
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_bm25_corpus(metadata_dir: Path, frame_map_path: str,
                      objects_dir: Path, output_path: str):
    """
    Xây dựng corpus cho BM25 search:
    - Mỗi document = 1 video (text từ metadata)
    - Lưu corpus dưới dạng JSON list để load lại nhanh
    
    corpus[i] = {
        "video_id": str,
        "text": str,        # nối title + description + tags
        "tokens": [str],    # đã tokenize
    }
    """
    with open(frame_map_path, "r", encoding="utf-8") as f:
        frame_map = json.load(f)

    # Thu thập unique video ids
    video_ids = list(dict.fromkeys([fm["video_id"] for fm in frame_map]))
    print(f"[BM25] Tìm thấy {len(video_ids)} video")

    corpus = []
    for video_id in tqdm(video_ids, desc="Building BM25 corpus"):
        meta_file = metadata_dir / f"{video_id}.json"
        text_parts = [video_id.replace("_", " ")]  # luôn include video_id

        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                # Các trường phổ biến trong YouTube metadata
                for field in ["title", "description", "name"]:
                    val = meta.get(field, "")
                    if val:
                        text_parts.append(str(val))

                # Tags (thường là list)
                tags = meta.get("tags", meta.get("keywords", []))
                if isinstance(tags, list):
                    text_parts.extend([str(t) for t in tags])
                elif isinstance(tags, str):
                    text_parts.append(tags)

                # Watch_title, channel_title nếu có
                for field in ["watch_title", "channel_title", "uploader"]:
                    val = meta.get(field, "")
                    if val:
                        text_parts.append(str(val))

            except Exception as e:
                print(f"  [WARN] Lỗi metadata {video_id}: {e}")

        # Ghép text và tokenize đơn giản
        full_text = " ".join(text_parts).lower()
        tokens = full_text.split()  # simple whitespace tokenization

        corpus.append({
            "video_id": video_id,
            "text": full_text,
            "tokens": tokens
        })

    # Lưu corpus
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    print(f"[BM25] Saved corpus ({len(corpus)} docs) -> {output_path}")
    return corpus


def main():
    parser = argparse.ArgumentParser(description="AIC2026 — Build BM25 Corpus")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"])
    metadata_dir = root / cfg["data"]["metadata_dir"]
    objects_dir = root / cfg["data"]["objects_dir"]
    frame_map_path = cfg["index"]["frame_map_path"]
    output_path = cfg["index"]["bm25_corpus_path"]

    build_bm25_corpus(metadata_dir, frame_map_path, objects_dir, output_path)
    print("\n[Done] BM25 corpus đã sẵn sàng!")
    print("\nBây giờ có thể chạy search:")
    print("  python run_search.py --config config.yaml --query-file queries.json")


if __name__ == "__main__":
    main()
