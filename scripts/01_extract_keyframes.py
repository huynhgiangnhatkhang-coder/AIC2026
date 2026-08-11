"""
AIC 2026 Baseline — Script 01: Extract Keyframes
=================================================
Trích xuất keyframe từ video và xây dựng frame_map.json

Usage:
    python scripts/01_extract_keyframes.py --config config.yaml

Nếu dataset đã có thư mục Keyframes/ sẵn, script này sẽ chỉ
xây dựng frame_map.json từ metadata (skip video extraction).
"""
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_frame_map_from_keyframes(keyframes_dir: Path, metadata_dir: Path) -> list[dict]:
    """
    Quét thư mục Keyframes/ để xây dựng frame_map.
    
    frame_map[i] = {
        "global_idx": i,
        "video_id": "L01_V001",
        "frame_filename": "0000.jpg",
        "frame_index": 0          # frame index thực trong video (từ metadata)
    }
    """
    frame_map = []
    global_idx = 0

    video_dirs = sorted([d for d in keyframes_dir.iterdir() if d.is_dir()])
    print(f"[Frame Map] Tìm thấy {len(video_dirs)} video trong Keyframes/")

    for video_dir in tqdm(video_dirs, desc="Building frame map"):
        video_id = video_dir.name  # e.g. "L01_V001"

        # Đọc metadata để lấy frame_index thực
        meta_file = metadata_dir / f"{video_id}.json"
        frame_index_map = {}  # filename -> actual frame index
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                # Metadata AIC thường có dạng: {"frames": [{"n": "0000.jpg", "idx": 42}, ...]}
                # hoặc có thể là dạng khác tuỳ dataset
                if "frames" in meta:
                    for frame_info in meta["frames"]:
                        fname = frame_info.get("n", frame_info.get("name", ""))
                        idx = frame_info.get("idx", frame_info.get("frame_idx", -1))
                        frame_index_map[fname] = idx
            except Exception as e:
                print(f"  [WARN] Không đọc được metadata {meta_file}: {e}")

        # Quét các file jpg trong video_dir
        frame_files = sorted(
            [f for f in video_dir.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")]
        )

        for frame_file in frame_files:
            fname = frame_file.name
            # Lấy frame_index: ưu tiên từ metadata, fallback sang tên file
            if fname in frame_index_map:
                frame_index = frame_index_map[fname]
            else:
                # Tên file thường là "0000.jpg" -> index = 0
                try:
                    frame_index = int(frame_file.stem)
                except ValueError:
                    frame_index = global_idx

            frame_map.append({
                "global_idx": global_idx,
                "video_id": video_id,
                "frame_filename": fname,
                "frame_index": frame_index,
                "frame_path": str(frame_file)
            })
            global_idx += 1

    print(f"[Frame Map] Tổng số frame: {global_idx}")
    return frame_map


def extract_keyframes_from_videos(videos_dir: Path, keyframes_dir: Path, fps: float = 1.0):
    """
    Trích xuất keyframe từ video bằng OpenCV (nếu thư mục Keyframes chưa có).
    Mặc định: 1 frame/giây
    """
    import cv2

    video_files = sorted(videos_dir.glob("*.mp4"))
    print(f"[Extract] Tìm thấy {len(video_files)} video")

    for video_file in tqdm(video_files, desc="Extracting keyframes"):
        video_id = video_file.stem  # "L01_V001"
        out_dir = keyframes_dir / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Bỏ qua nếu đã extract rồi
        existing = list(out_dir.glob("*.jpg"))
        if len(existing) > 0:
            continue

        cap = cv2.VideoCapture(str(video_file))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1, int(video_fps / fps))

        frame_idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                out_path = out_dir / f"{saved:04d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1
            frame_idx += 1
        cap.release()


def main():
    parser = argparse.ArgumentParser(description="AIC2026 — Extract Keyframes & Build Frame Map")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--extract", action="store_true",
                        help="Trích xuất keyframe từ video (nếu chưa có)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Số frame/giây khi extract (mặc định: 1)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"])
    keyframes_dir = root / cfg["data"]["keyframes_dir"]
    videos_dir = root / cfg["data"]["videos_dir"]
    metadata_dir = root / cfg["data"]["metadata_dir"]

    index_dir = Path(cfg["index"]["faiss_index_path"]).parent
    index_dir.mkdir(parents=True, exist_ok=True)

    # Bước 1: Trích xuất keyframe nếu cần
    if args.extract:
        print("[Step 1] Extracting keyframes from videos...")
        keyframes_dir.mkdir(parents=True, exist_ok=True)
        extract_keyframes_from_videos(videos_dir, keyframes_dir, fps=args.fps)
    else:
        print("[Step 1] Bỏ qua extract (dùng Keyframes/ sẵn có)")

    # Bước 2: Xây dựng frame_map
    print("[Step 2] Building frame_map.json...")
    frame_map = build_frame_map_from_keyframes(keyframes_dir, metadata_dir)

    frame_map_path = cfg["index"]["frame_map_path"]
    with open(frame_map_path, "w", encoding="utf-8") as f:
        json.dump(frame_map, f, ensure_ascii=False)

    print(f"[Done] Đã lưu frame_map.json -> {frame_map_path}")
    print(f"       Tổng frame: {len(frame_map)}")


if __name__ == "__main__":
    main()
