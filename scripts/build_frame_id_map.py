"""
AIC 2026 — Build frame-id map from map-keyframes CSVs
======================================================
Mỗi file map-keyframes/<video>.csv (cột: n, pts_time, fps, frame_idx) cho biết
keyframe thứ `n` (index 1-based, khớp frame_id nội bộ / tên file 001.jpg,...)
tương ứng với frame gốc `frame_idx` trong video. BTC yêu cầu nộp frame_idx.

Script gộp tất cả CSV thành một JSON để tra cứu O(1) tại runtime:

    indexes/frame_id_map.json
    {
        "L21_V001": { "1": 0, "2": 90, ..., "307": 37716 },
        ...
    }

Usage:
    python scripts/build_frame_id_map.py [--map-dir map-keyframes]
                                         [--output indexes/frame_id_map.json]
"""
import argparse
import csv
import json
import os


def build_frame_id_map(map_dir: str) -> dict:
    """Đọc toàn bộ CSV trong map_dir → {video_folder: {str(n): frame_idx}}."""
    result: dict = {}
    if not os.path.isdir(map_dir):
        print(f"[ERROR] Không tìm thấy thư mục: {map_dir}")
        return result

    files = sorted(f for f in os.listdir(map_dir) if f.lower().endswith(".csv"))
    print(f"[build] Đang đọc {len(files)} file CSV từ {map_dir}")

    for fname in files:
        video_folder = os.path.splitext(fname)[0]
        path = os.path.join(map_dir, fname)
        mapping: dict = {}
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                n = (row.get("n") or "").strip()
                frame_idx = (row.get("frame_idx") or "").strip()
                if not n or not frame_idx:
                    continue
                try:
                    mapping[str(int(n))] = int(frame_idx)
                except (TypeError, ValueError):
                    continue
        if mapping:
            result[video_folder] = mapping

    total = sum(len(v) for v in result.values())
    print(f"[build] OK: {len(result)} videos, {total} frames")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build frame-id map từ map-keyframes")
    parser.add_argument("--map-dir", default="map-keyframes")
    parser.add_argument("--output", default="indexes/frame_id_map.json")
    args = parser.parse_args()

    mapping = build_frame_id_map(args.map_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, sort_keys=True)
    print(f"[build] Đã ghi -> {args.output}")


if __name__ == "__main__":
    main()
