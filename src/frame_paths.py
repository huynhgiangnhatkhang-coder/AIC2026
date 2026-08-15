"""
AIC 2026 — Keyframe path resolution helpers
============================================
Shared module để chuyển một cặp (video_id, frame_id) thành đường dẫn tới
file ảnh keyframe trên đĩa. Dùng bởi REST API (/frames/...) và các searcher.

Chiến lược resolution (ưu tiên index đã build trước, rồi mới heuristic):
  1. Tra cứu chính xác từ index sẵn có:
     - indexes/frame_map.json  → {video_id, frame_filename, frame_path}
     - ocr_database.json       → rel_path như Keyframes_L22/keyframes/L22_V019/069.jpg
  2. Dùng đúng filename từ index để nối với các root keyframes.
  3. Heuristic cuối cùng: thử nhiều định dạng zero-pad (.jpg/.png).
"""
import os
import re
from typing import Dict, Iterator, List, Optional, Set, Tuple

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# Module-level cache: key = abs path của index file
_frame_map_cache: Dict[str, Dict[Tuple[str, int], Tuple[str, str]]] = {}
_ocr_cache: Dict[str, Dict[Tuple[str, int], str]] = {}


def normalize_video_id(video_id: str) -> str:
    """
    Chuẩn hoá video_id → tên thư mục video.
    Bỏ đuôi .mp4/.MP4 nếu có: "L21_V001.mp4" → "L21_V001".
    """
    v = (video_id or "").strip()
    if v.lower().endswith(".mp4"):
        v = v[:-4]
    return v


def _candidate_filenames(frame_id: int) -> Iterator[str]:
    """Các tên file khả dĩ theo zero-padding (giống textual_kis._find_image_path)."""
    yield f"{frame_id}.jpg"
    for w in (3, 4, 5, 6):
        yield f"{frame_id:0{w}d}.jpg"
    yield f"{frame_id}.png"
    yield f"{frame_id:04d}.png"


def _load_frame_map(path: str) -> Dict[Tuple[str, int], Tuple[str, str]]:
    """
    Đọc indexes/frame_map.json → {(video_id, int_stem): (frame_filename, frame_path)}.
    Lazy-load + cache theo đường dẫn.
    """
    key = os.path.abspath(path)
    if key in _frame_map_cache:
        return _frame_map_cache[key]

    import json

    idx: Dict[Tuple[str, int], Tuple[str, str]] = {}
    try:
        with open(key, "r", encoding="utf-8") as f:
            for rec in json.load(f):
                stem = os.path.splitext(rec.get("frame_filename", ""))[0]
                if not stem:
                    continue
                try:
                    fid = int(stem)
                except (TypeError, ValueError):
                    continue
                idx[(rec.get("video_id", ""), fid)] = (
                    rec.get("frame_filename", ""),
                    rec.get("frame_path", ""),
                )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    _frame_map_cache[key] = idx
    return idx


def _load_ocr(path: str) -> Dict[Tuple[str, int], str]:
    """
    Đọc ocr_database.json → {(video_folder, int_stem): rel_path}.
    Key của file có dạng: Keyframes_L22/keyframes/L22_V019/069.jpg
    """
    key = os.path.abspath(path)
    if key in _ocr_cache:
        return _ocr_cache[key]

    import json

    idx: Dict[Tuple[str, int], str] = {}
    try:
        with open(key, "r", encoding="utf-8") as f:
            for rel in json.load(f).keys():
                parts = rel.replace("\\", "/").split("/")
                if len(parts) < 2:
                    continue
                video_folder = parts[-2]
                stem = os.path.splitext(parts[-1])[0]
                try:
                    fid = int(stem)
                except (TypeError, ValueError):
                    continue
                idx[(video_folder, fid)] = rel
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    _ocr_cache[key] = idx
    return idx


def candidate_roots(cfg: Optional[dict] = None,
                    extra_roots: Optional[List[str]] = None) -> List[str]:
    """Gom các thư mục root khả dĩ (data.root, keyframes_dirs_list, extra)."""
    roots: List[str] = []
    if cfg:
        data = cfg.get("data") or {}
        if data.get("root"):
            roots.append(str(data["root"]))
        for d in data.get("keyframes_dirs_list") or []:
            if d:
                roots.append(str(d))
    for r in extra_roots or []:
        if r:
            roots.append(str(r))

    seen: Set[str] = set()
    out: List[str] = []
    for r in roots:
        norm = os.path.normpath(r)
        if norm not in seen:
            seen.add(norm)
            out.append(r)
    return out


def _candidate_subpaths(video_folder: str, filename: str) -> Iterator[str]:
    """
    Các đường dẫn con dưới mỗi root:
      Keyframes_<batch>/keyframes/<video>/<file>   (cấu trúc BTC chuẩn)
      Keyframes/<video>/<file>                     (cấu trúc phẳng)
      <video>/<file>
    """
    batch = video_folder.split("_")[0] if "_" in video_folder else video_folder
    yield os.path.join(f"Keyframes_{batch}", "keyframes", video_folder, filename)
    yield os.path.join("Keyframes", video_folder, filename)
    yield os.path.join(video_folder, filename)


def resolve_frame_path(video_id: str,
                       frame_id: int,
                       cfg: Optional[dict] = None,
                       extra_roots: Optional[List[str]] = None,
                       exact_filename: Optional[str] = None) -> Optional[str]:
    """
    Giải quyết (video_id, frame_id) → absolute path của ảnh keyframe.

    Args:
        video_id:      "L21_V001" hoặc "L21_V001.mp4"
        frame_id:      số frame (int)
        cfg:           config.yaml dict (để đọc data.root, keyframes_dirs_list,
                       index.frame_map_path, data.ocr_database_path)
        extra_roots:   thêm các root keyframes (vd. keyframes_dir của searcher)
        exact_filename: filename chính xác từ index vector (Milvus), nếu có

    Returns:
        str path nếu tìm thấy, ngược lại None.
    """
    video_folder = normalize_video_id(video_id)
    if not video_folder or not _VIDEO_ID_RE.match(video_folder):
        return None
    try:
        frame_id = int(frame_id)
    except (TypeError, ValueError):
        return None

    roots = candidate_roots(cfg, extra_roots)
    filenames: Set[str] = set()
    direct: List[str] = []

    # ── 1. Tra cứu chính xác từ các index đã build ──
    if cfg:
        index_cfg = cfg.get("index") or {}
        if index_cfg.get("frame_map_path"):
            idx = _load_frame_map(index_cfg["frame_map_path"])
            hit = idx.get((video_folder, frame_id))
            if hit:
                fn, fp = hit
                if fn:
                    filenames.add(fn)
                if fp:
                    direct.append(fp)

        data_cfg = cfg.get("data") or {}
        ocr_path = data_cfg.get("ocr_database_path", "ocr_database.json")
        ocr_hit = _load_ocr(ocr_path).get((video_folder, frame_id))
        if ocr_hit:
            filenames.add(os.path.basename(ocr_hit.replace("\\", "/")))
            if data_cfg.get("root"):
                direct.append(os.path.join(str(data_cfg["root"]), ocr_hit))

    if exact_filename:
        filenames.add(exact_filename)

    # ── 2. Đường dẫn trực tiếp từ index ──
    for p in direct:
        ap = os.path.abspath(p)
        if os.path.isfile(ap):
            return ap

    # ── 3. Dùng đúng filename từ index để nối với các root ──
    for fn in filenames:
        for root in roots:
            for sub in _candidate_subpaths(video_folder, fn):
                p = os.path.join(root, sub)
                if os.path.isfile(p):
                    return os.path.abspath(p)

    # ── 4. Heuristic cuối cùng: zero-padding formats ──
    for fmt in _candidate_filenames(frame_id):
        for root in roots:
            for sub in _candidate_subpaths(video_folder, fmt):
                p = os.path.join(root, sub)
                if os.path.isfile(p):
                    return os.path.abspath(p)

    return None
