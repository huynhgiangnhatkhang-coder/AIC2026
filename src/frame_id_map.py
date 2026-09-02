"""
AIC 2026 — Frame-id lookup helper (keyframe index → frame_idx)
===============================================================
Nội bộ hệ thống (Milvus/DB, tên file 001.jpg,...) dùng frame_id = chỉ số keyframe
= cột `n` trong map-keyframes/<video>.csv. BTC yêu cầu frame gốc trong video
(cột `frame_idx`). Module này cung cấp hàm tra cứu index → frame_idx dùng ở
tầng output (format nộp bài / field frame_id trả về), dựa trên JSON đã build:

    indexes/frame_id_map.json = { "L21_V001": { "1": 0, "2": 90, ... }, ... }

Nếu video/index không có trong map, trả về chính giá trị index (no-op an toàn).
"""
import json
import os
from typing import Dict, Optional

_cache: Optional[Dict[str, Dict[str, int]]] = None
_cache_path: Optional[str] = None


def _default_map_path() -> str:
    """Tìm frame_id_map.json dựa trên config.yaml nếu có, ngược lại dùng mặc định."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(here, "..", "config.yaml")
        if os.path.exists(cfg_path):
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            idx_cfg = cfg.get("index") or {}
            if idx_cfg.get("frame_id_map_path"):
                p = idx_cfg["frame_id_map_path"]
                return p if os.path.isabs(p) else os.path.normpath(os.path.join(here, "..", p))
    except Exception:
        pass
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "indexes", "frame_id_map.json"))


def load_map(path: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    """Lazy-load + cache JSON map theo đường dẫn."""
    global _cache, _cache_path
    if path is None:
        path = _default_map_path()
    key = os.path.abspath(path)
    if _cache is not None and _cache_path == key:
        return _cache
    data: Dict[str, Dict[str, int]] = {}
    try:
        with open(key, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        for video, inner in (loaded or {}).items():
            data[video] = {str(k): int(v) for k, v in inner.items()}
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        data = {}
    _cache = data
    _cache_path = key
    return data


def normalize_video_id(video_id: str) -> str:
    """'L21_V001.mp4' → 'L21_V001' (bỏ đuôi .mp4/.MP4)."""
    v = (video_id or "").strip()
    if v.lower().endswith(".mp4"):
        v = v[:-4]
    return v


def frame_id_from_index(video_id: str, index: int,
                        map_path: Optional[str] = None) -> int:
    """
    Chuyển chỉ số keyframe (n) → frame_idx gốc trong video.

    Args:
        video_id:  'L21_V001' hoặc 'L21_V001.mp4'
        index:     chỉ số keyframe nội bộ (int)
        map_path:  (tuỳ chọn) đường dẫn frame_id_map.json; None = auto từ config

    Returns:
        frame_idx nếu tra được, ngược lại trả nguyên `index` (fallback an toàn).
    """
    try:
        index = int(index)
    except (TypeError, ValueError):
        return index if isinstance(index, int) else (index or 0)

    mapping = load_map(map_path).get(normalize_video_id(video_id), {})
    return mapping.get(str(index), index)


def frame_ids_from_indexes(video_id: str, indexes,
                           map_path: Optional[str] = None) -> list:
    """Convert danh sách chỉ số keyframe → danh sách frame_idx."""
    return [frame_id_from_index(video_id, i, map_path) for i in indexes]
