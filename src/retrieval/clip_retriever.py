"""
AIC 2026 Baseline — CLIP Retriever
====================================
Module tìm kiếm frame bằng CLIP text-to-image similarity.
Dùng FAISS index để tìm kiếm nhanh.
"""
import json
import numpy as np
import faiss
import torch
from typing import List, Dict, Optional
from pathlib import Path


class CLIPRetriever:
    """
    Retriever sử dụng CLIP ViT-B/32 để text→image retrieval.
    
    Dùng FAISS IVFFlat index đã build từ script 02_build_clip_index.py.
    """

    def __init__(self, index_path: str, frame_map_path: str,
                 model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Args:
            index_path:     path đến FAISS .index file
            frame_map_path: path đến frame_map.json
            model_name:     CLIP model name (phải khớp với lúc tạo .npy)
            device:         "cuda" hoặc "cpu"
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_name = model_name
        self._model = None
        self._tokenize = None

        # Load FAISS index
        print(f"[CLIPRetriever] Loading FAISS index: {index_path}")
        self.index = faiss.read_index(index_path)
        self.index.nprobe = 64  # tăng nprobe để tìm kiếm chính xác hơn

        # Load frame map
        print(f"[CLIPRetriever] Loading frame_map: {frame_map_path}")
        with open(frame_map_path, "r", encoding="utf-8") as f:
            self.frame_map = json.load(f)

        print(f"[CLIPRetriever] Ready | {self.index.ntotal} frames indexed")

    def _load_model(self):
        """Lazy-load CLIP model."""
        if self._model is None:
            try:
                import clip
                print(f"[CLIPRetriever] Loading CLIP model: {self.model_name}")
                self._model, _ = clip.load(self.model_name, device=self.device)
                self._model.eval()
                self._tokenize = clip.tokenize
            except ImportError:
                # Fallback: dùng open_clip
                import open_clip
                print(f"[CLIPRetriever] Loading via open_clip: {self.model_name}")
                model, _, _ = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai", device=self.device
                )
                self._model = model
                self._tokenize = open_clip.get_tokenizer("ViT-B-32")

    def encode_text(self, query: str) -> np.ndarray:
        """
        Encode query text thành vector embedding (normalized).
        
        Args:
            query: Chuỗi văn bản mô tả cần tìm
            
        Returns:
            np.ndarray shape (512,), đã normalize
        """
        self._load_model()

        with torch.no_grad():
            tokens = self._tokenize([query]).to(self.device)
            text_features = self._model.encode_text(tokens)
            text_features = text_features.cpu().numpy().astype(np.float32)

        # Normalize
        norm = np.linalg.norm(text_features)
        if norm > 0:
            text_features = text_features / norm

        return text_features.squeeze()

    def search(self, query: str, top_k: int = 100,
               video_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        Tìm kiếm frames theo query text.
        
        Args:
            query:        văn bản mô tả cần tìm
            top_k:        số frame kết quả muốn lấy
            video_filter: nếu không None, chỉ giữ frame thuộc các video này
            
        Returns:
            List of dicts: [{
                "video_id": str,
                "frame_index": int,
                "frame_filename": str,
                "frame_path": str,
                "score": float,     # cosine similarity
                "rank": int
            }]
        """
        # Encode query
        query_vec = self.encode_text(query).reshape(1, -1)

        # Tăng top_k nếu cần filter
        search_k = top_k * 5 if video_filter else top_k

        # FAISS search
        distances, indices = self.index.search(query_vec, search_k)

        results = []
        rank = 1
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.frame_map):
                continue

            fm = self.frame_map[idx]
            video_id = fm["video_id"]

            # Filter by video nếu cần
            if video_filter and video_id not in video_filter:
                continue

            results.append({
                "video_id": video_id,
                "frame_index": fm["frame_index"],
                "frame_filename": fm["frame_filename"],
                "frame_path": fm.get("frame_path", ""),
                "global_idx": fm["global_idx"],
                "score": float(dist),
                "rank": rank
            })
            rank += 1

            if len(results) >= top_k:
                break

        return results

    def search_batch(self, queries: List[str], top_k: int = 100) -> List[List[Dict]]:
        """
        Tìm kiếm batch nhiều queries cùng lúc.
        """
        all_results = []
        for query in queries:
            results = self.search(query, top_k=top_k)
            all_results.append(results)
        return all_results
