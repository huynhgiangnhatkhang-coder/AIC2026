"""
AIC 2026 Baseline — BM25 Retriever
=====================================
Module tìm kiếm video bằng BM25 full-text search trên metadata.
"""
import json
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    BM25 retriever trên metadata của video.
    Kết quả là danh sách video_id phù hợp (video-level),
    sau đó sẽ được kết hợp với frame-level results từ CLIP.
    """

    def __init__(self, corpus_path: str, frame_map_path: str):
        """
        Args:
            corpus_path:    path đến bm25_corpus.json
            frame_map_path: path đến frame_map.json
        """
        print(f"[BM25] Loading corpus: {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        # Video_id list (để map từ BM25 index -> video_id)
        self.video_ids = [doc["video_id"] for doc in self.corpus]
        tokenized = [doc["tokens"] for doc in self.corpus]

        print(f"[BM25] Building BM25 index for {len(self.corpus)} videos...")
        self.bm25 = BM25Okapi(tokenized)

        # Load frame map để expand video -> frames
        print(f"[BM25] Loading frame_map: {frame_map_path}")
        with open(frame_map_path, "r", encoding="utf-8") as f:
            frame_map = json.load(f)

        # Build video -> [frames] mapping
        self.video_to_frames: Dict[str, List[Dict]] = {}
        for fm in frame_map:
            vid = fm["video_id"]
            if vid not in self.video_to_frames:
                self.video_to_frames[vid] = []
            self.video_to_frames[vid].append(fm)

        print(f"[BM25] Ready | {len(self.corpus)} videos")

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def search_videos(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Tìm kiếm video phù hợp nhất với query.
        
        Returns:
            List of {"video_id": str, "bm25_score": float}
        """
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Lấy top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                break
            results.append({
                "video_id": self.video_ids[idx],
                "bm25_score": score
            })
        return results

    def search_frames(self, query: str, top_k_videos: int = 20,
                      frames_per_video: int = 10) -> List[Dict]:
        """
        Tìm kiếm và expand sang danh sách frames.
        
        Returns:
            List of frame dicts với bm25_score
        """
        video_results = self.search_videos(query, top_k=top_k_videos)

        frames = []
        for vr in video_results:
            vid = vr["video_id"]
            vid_frames = self.video_to_frames.get(vid, [])

            # Lấy đều frames từ video (để cover toàn video)
            if len(vid_frames) > frames_per_video:
                step = len(vid_frames) // frames_per_video
                selected = vid_frames[::step][:frames_per_video]
            else:
                selected = vid_frames

            for fm in selected:
                frames.append({
                    "video_id": vid,
                    "frame_index": fm["frame_index"],
                    "frame_filename": fm["frame_filename"],
                    "frame_path": fm.get("frame_path", ""),
                    "global_idx": fm["global_idx"],
                    "score": vr["bm25_score"],
                    "bm25_score": vr["bm25_score"]
                })

        return frames

    def get_candidate_video_ids(self, query: str, top_k: int = 50) -> List[str]:
        """
        Trả về danh sách video_id có khả năng chứa đáp án.
        Dùng để filter kết quả CLIP search.
        """
        results = self.search_videos(query, top_k=top_k)
        return [r["video_id"] for r in results]
