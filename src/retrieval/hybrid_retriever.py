"""
AIC 2026 Baseline — Hybrid Retriever
=======================================
Kết hợp CLIP/Milvus (semantic) + BM25 (keyword) bằng Reciprocal Rank Fusion (RRF).

RRF score(d) = Σ 1 / (k + rank_i(d))
  với k=60 (hằng số để giảm tác động của rank thấp)

Hỗ trợ cả hai backend:
  - MilvusRetriever (mặc định, dùng Milvus Lite)
  - CLIPRetriever   (fallback, dùng FAISS)
"""
from typing import List, Dict, Optional, Union
from .bm25_retriever import BM25Retriever

# Import cả hai retriever — chỉ dùng 1 trong runtime
try:
    from .milvus_retriever import MilvusRetriever
    _MILVUS_AVAILABLE = True
except ImportError:
    _MILVUS_AVAILABLE = False

try:
    from .clip_retriever import CLIPRetriever
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class HybridRetriever:
    """
    Hybrid retriever kết hợp CLIP/Milvus và BM25 qua Reciprocal Rank Fusion.
    
    vector_retriever có thể là MilvusRetriever hoặc CLIPRetriever —
    cả hai đều có interface .search(query, top_k, video_filter) giống nhau.
    """

    def __init__(self, vector_retriever,
                 bm25_retriever,           # BM25Retriever hoac None
                 clip_weight: float = 0.7,
                 bm25_weight: float = 0.3,
                 rrf_k: int = 60):
        self.clip = vector_retriever
        self.vector = vector_retriever
        self.bm25 = bm25_retriever         # co the la None neu chua co corpus
        self.clip_weight = clip_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        if bm25_retriever is None:
            print("[HybridRetriever] BM25 disabled — running in CLIP-only mode")

    def search(self, query: str, top_k: int = 100,
               top_k_clip: int = 500, top_k_bm25_videos: int = 30) -> List[Dict]:
        """
        Hybrid search: CLIP + BM25 → RRF fusion → top_k results.
        
        Args:
            query:            câu query văn bản
            top_k:            số kết quả cuối cùng
            top_k_clip:       số frame lấy từ CLIP search
            top_k_bm25_videos: số video lấy từ BM25 search
            
        Returns:
            List of ranked frame dicts với hybrid_score
        """
        # 1. CLIP search
        clip_results = self.clip.search(query, top_k=top_k_clip)

        # 2. BM25 search (video-level, expand sang frames)
        clip_filtered = []
        if self.bm25 is not None:
            bm25_candidate_videos = self.bm25.get_candidate_video_ids(
                query, top_k=top_k_bm25_videos
            )
            # 3. CLIP search voi filter tu BM25
            if bm25_candidate_videos:
                clip_filtered = self.clip.search(
                    query,
                    top_k=top_k_clip,
                    video_filter=bm25_candidate_videos
                )

        # 4. RRF Fusion
        scores: Dict[str, Dict] = {}

        def add_rrf(results: List[Dict], weight: float):
            for rank, item in enumerate(results, start=1):
                key = f"{item['video_id']}::{item['frame_index']}"
                rrf_score = weight / (self.rrf_k + rank)
                if key not in scores:
                    scores[key] = {**item, "hybrid_score": 0.0}
                scores[key]["hybrid_score"] += rrf_score

        add_rrf(clip_results, self.clip_weight)
        if clip_filtered:
            add_rrf(clip_filtered, self.bm25_weight)

        # 5. Sort by hybrid_score
        ranked = sorted(scores.values(), key=lambda x: x["hybrid_score"], reverse=True)

        # 6. Thêm rank và trả về top_k
        for i, item in enumerate(ranked[:top_k]):
            item["rank"] = i + 1

        return ranked[:top_k]

    def search_with_object_boost(self, query: str, object_keywords: List[str],
                                  top_k: int = 100, objects_dir: str = None) -> List[Dict]:
        """
        Boost score cho các frame có chứa object liên quan (từ Objects/*.json).
        
        Args:
            query:           câu query chính
            object_keywords: danh sách object cần boost (vd: ["laptop", "person"])
            objects_dir:     đường dẫn thư mục Objects/
        """
        # Base hybrid search
        results = self.search(query, top_k=top_k * 3)

        if not objects_dir or not object_keywords:
            return results[:top_k]

        import json
        from pathlib import Path
        import re

        objects_path = Path(objects_dir)
        kw_lower = [k.lower() for k in object_keywords]

        # Boost score cho frame có chứa object keywords
        for item in results:
            vid = item["video_id"]
            frame_fn = item["frame_filename"]
            frame_stem = Path(frame_fn).stem  # "0000"
            obj_file = objects_path / vid / f"{frame_stem}.json"

            if obj_file.exists():
                try:
                    with open(obj_file, "r") as f:
                        obj_data = json.load(f)
                    # Lấy danh sách object labels
                    detected = []
                    for ann in obj_data.get("annotations", obj_data.get("objects", [])):
                        label = ann.get("display_name", ann.get("label", "")).lower()
                        detected.append(label)

                    # Boost nếu có match
                    match_count = sum(1 for kw in kw_lower
                                      if any(kw in det for det in detected))
                    if match_count > 0:
                        item["hybrid_score"] += 0.1 * match_count
                        item["object_boost"] = match_count
                except Exception:
                    pass

        # Re-sort
        results = sorted(results, key=lambda x: x["hybrid_score"], reverse=True)
        for i, item in enumerate(results[:top_k]):
            item["rank"] = i + 1

        return results[:top_k]
