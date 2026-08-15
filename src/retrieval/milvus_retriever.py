"""
AIC 2026 Baseline — Milvus Retriever
=======================================
Module tìm kiếm frame bằng CLIP text embedding + Milvus Lite.

Tích hợp logic search từ code gốc của user:
  - Load CLIP ViT-B/32 (openai/clip) để encode query text
  - L2-normalize text vector
  - Gọi Milvus client.search() với metric_type=IP
  - Tự động detect index type (FLAT / HNSW / IVF) → chọn search_params phù hợp

Ưu điểm so với FAISS:
  - video_id + frame_id lưu trực tiếp → không cần frame_map.json
  - Hỗ trợ filter theo video_id (dùng cho TRAKE)
  - HNSW index: nhanh hơn IVFFlat khi tập nhỏ-vừa (<10M vectors)
"""
import numpy as np
import torch
from typing import List, Dict, Optional
from pathlib import Path


class MilvusRetriever:
    """
    Retriever sử dụng Milvus Lite + CLIP ViT-B/32.

    Giao diện giống CLIPRetriever (FAISS) để có thể thay thế nhau trong pipeline.
    """

    def __init__(self, db_path: str, collection_name: str,
                 model_name: str = "ViT-B/32", device: str = "cuda"):
        """
        Args:
            db_path:         đường dẫn file .db Milvus Lite
            collection_name: tên collection
            model_name:      CLIP model name (phải khớp với lúc build DB)
            device:          "cuda" hoặc "cpu"
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"

        self._model = None
        self._tokenize = None
        self._client = None
        self._index_type = None     # "FLAT", "HNSW", "IVF_FLAT", ...

        self._connect()

    # ──────────────────────────────────────────
    # KẾT NỐI MILVUS  (từ code gốc user)
    # ──────────────────────────────────────────

    def _connect(self):
        """
        Kết nối tới Milvus Lite và load collection vào memory.
        (Bắt buộc gọi load_collection trước khi search — như trong code gốc)
        """
        from pymilvus import MilvusClient

        if not Path(self.db_path).exists():
            raise FileNotFoundError(
                f"Milvus DB chưa tồn tại: {self.db_path}\n"
                f"→ Chạy: python scripts/02b_build_milvus_db.py --config config.yaml"
            )

        print(f"[MilvusRetriever] Kết nối Database: {self.db_path}")
        self._client = MilvusClient(self.db_path)

        # Bắt buộc gọi load_collection để nạp index vào bộ nhớ trước khi search
        self._client.load_collection(self.collection_name)

        # Detect index type để chọn search_params đúng
        self._index_type = self._detect_index_type()

        stats = self._client.get_collection_stats(self.collection_name)
        row_count = stats.get("row_count", "?")
        print(f"[MilvusRetriever] Collection: '{self.collection_name}' | "
              f"Records: {row_count} | Index: {self._index_type}")

    def _detect_index_type(self) -> str:
        """
        Tự động detect loại index đang dùng trong collection.
        Trả về: "HNSW", "FLAT", "IVF_FLAT", "IVF_SQ8", "AUTOINDEX", ...
        """
        try:
            indexes = self._client.list_indexes(self.collection_name)
            if indexes:
                idx_info = self._client.describe_index(
                    self.collection_name, indexes[0]
                )
                return idx_info.get("index_type", "FLAT")
        except Exception:
            pass
        return "FLAT"   # fallback an toàn

    def _build_search_params(self, top_k: int) -> dict:
        """
        Xây dựng search_params phù hợp với index type.

        FLAT   → không cần params đặc biệt
        HNSW   → cần ef >= top_k
        IVF_*  → cần nprobe
        """
        idx = self._index_type.upper() if self._index_type else "FLAT"

        if idx in ("FLAT", "BIN_FLAT"):
            # FLAT: chỉ cần metric_type, không params thêm
            return {"metric_type": "IP"}

        elif idx == "HNSW":
            # HNSW: ef phải >= top_k (thường đặt 2*top_k để recall tốt hơn)
            ef = max(128, top_k * 2)
            return {"metric_type": "IP", "params": {"ef": ef}}

        elif idx.startswith("IVF"):
            # IVF_FLAT, IVF_SQ8, IVF_PQ: cần nprobe
            return {"metric_type": "IP", "params": {"nprobe": 64}}

        elif idx == "AUTOINDEX":
            # Milvus managed index — không cần params
            return {"metric_type": "IP"}

        else:
            # Unknown → dùng metric_type đơn giản như code gốc
            return {"metric_type": "IP"}

    # ──────────────────────────────────────────
    # CLIP TEXT ENCODER  (từ code gốc user)
    # ──────────────────────────────────────────

    def _load_model(self):
        """
        Lazy-load CLIP/SigLIP model.
        """
        if self._model is not None:
            return

        print(f"[MilvusRetriever] Đang tải Text Encoder: {self.model_name}")

        if "siglip" in self.model_name.lower():
            from transformers import AutoModel, AutoProcessor
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            self._tokenize = AutoProcessor.from_pretrained(self.model_name)
            print(f"[MilvusRetriever] SigLIP loaded (HF) | device={self.device}")
        else:
            try:
                import clip
                self._model, _ = clip.load(self.model_name, device=self.device)
                self._model.eval()
                self._tokenize = clip.tokenize
                print(f"[MilvusRetriever] CLIP loaded (openai/clip) | device={self.device}")
            except ImportError:
                import open_clip
                print(f"[MilvusRetriever] Fallback: open_clip | model={self.model_name}")
                model, _, _ = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai", device=self.device
                )
                self._model = model.eval()
                self._tokenize = open_clip.get_tokenizer("ViT-B-32")

    def encode_text(self, query: str) -> np.ndarray:
        """
        Encode query text → embedding, L2-normalized.
        """
        self._load_model()

        with torch.no_grad():
            if "siglip" in self.model_name.lower():
                inputs = self._tokenize(text=[query], padding="max_length", return_tensors="pt").to(self.device)
                text_features = self._model.get_text_features(**inputs)
            else:
                text_inputs = self._tokenize([query], truncate=True).to(self.device)
                text_features = self._model.encode_text(text_inputs)

            # L2-normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_vector = text_features[0].cpu().numpy().astype(np.float32)

        return text_vector

    # ──────────────────────────────────────────
    # SEARCH  (từ code gốc user + mở rộng)
    # ──────────────────────────────────────────

    def search(self, query: str, top_k: int = 100,
               video_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        Tìm kiếm frames theo query text.

        Logic core giống code gốc của user:
          1. Encode text → vector
          2. client.search() với metric_type=IP
          3. Parse kết quả → list dict

        Mở rộng thêm:
          - video_filter: chỉ tìm trong danh sách video cụ thể
          - Auto search_params theo index type

        Args:
            query:        văn bản mô tả cần tìm
            top_k:        số frame kết quả trả về
            video_filter: nếu không None, filter theo video_id
                         (format: ["L01_V001.mp4", ...])

        Returns:
            List[Dict]: [{
                "video_id":       str,
                "frame_id":       int,
                "frame_filename": str,
                "score":          float,   # cosine similarity (IP)
                "rank":           int
            }]
        """
        # Bước 1: Encode text (giống code gốc)
        text_vector = self.encode_text(query).tolist()

        # Bước 2: Xây dựng filter expression (Milvus DSL)
        filter_expr = None
        if video_filter:
            ids_str = ", ".join(f'"{v}"' for v in video_filter)
            filter_expr = f"video_id in [{ids_str}]"

        # Bước 3: Chọn search_params theo index type
        search_params = self._build_search_params(top_k)

        # Bước 4: Gọi client.search() (giống code gốc)
        try:
            results = self._client.search(
                collection_name=self.collection_name,
                data=[text_vector],
                anns_field="embedding",
                search_params=search_params,
                limit=top_k,
                output_fields=["video_id", "frame_id", "frame_filename"],
                filter=filter_expr
            )
        except Exception as e:
            print(f"[MilvusRetriever] Search error: {e}")
            # Retry với params đơn giản hơn (fallback như code gốc)
            try:
                results = self._client.search(
                    collection_name=self.collection_name,
                    data=[text_vector],
                    limit=top_k,
                    output_fields=["video_id", "frame_id", "frame_filename"],
                    search_params={"metric_type": "IP"},
                    filter=filter_expr
                )
            except Exception as e2:
                print(f"[MilvusRetriever] Retry also failed: {e2}")
                return []

        # Bước 5: Parse kết quả (giống code gốc, thêm rank)
        output = []
        for rank, hit in enumerate(results[0], start=1):
            entity = hit.get("entity", hit)
            output.append({
                "video_id":       entity.get("video_id", ""),
                "frame_id":       entity.get("frame_id", 0),
                "frame_index":    entity.get("frame_id", 0),   # alias cho compat
                "frame_filename": entity.get("frame_filename", ""),
                "frame_path":     "",
                "score":          float(hit.get("distance", 0.0)),
                "hybrid_score":   float(hit.get("distance", 0.0)),
                "rank":           rank
            })

        return output

    def search_and_print(self, query: str, top_k: int = 5):
        """
        Search và in kết quả ra màn hình — giống hệt output của code gốc user.

        Usage:
            retriever.search_and_print("Night scene with trucks", top_k=5)
        """
        print(f"\nĐang xử lý truy vấn: '{query}'")
        print("Đang tìm kiếm trong Database...")

        results = self.search(query, top_k=top_k)

        print("\n=== KẾT QUẢ TÌM KIẾM ===")
        if not results:
            print("  Không tìm thấy kết quả.")
            return results

        for hit in results:
            print(f"  video_id = {hit['video_id']}, "
                  f"frame_id = {hit['frame_id']} "
                  f"(Độ tương đồng: {hit['score']:.4f})")
        return results

    def search_batch(self, queries: List[str], top_k: int = 100) -> List[List[Dict]]:
        """Batch search nhiều queries."""
        return [self.search(q, top_k=top_k) for q in queries]

    def get_video_ids(self) -> List[str]:
        """Lấy danh sách tất cả video_id trong database."""
        try:
            results = self._client.query(
                collection_name=self.collection_name,
                filter="frame_id == 0",
                output_fields=["video_id"],
                limit=100000
            )
            return list(set(r["video_id"] for r in results))
        except Exception as e:
            print(f"[MilvusRetriever] get_video_ids error: {e}")
            return []

    def get_frame_filename(self, video_id: str, frame_id: int) -> Optional[str]:
        """
        Tra cứu filename chính xác của một frame trong index (nếu được lưu).

        Args:
            video_id: "L21_V001.mp4" hoặc "L21_V001"
            frame_id: số thứ tự frame

        Returns:
            str filename (vd. "069.jpg") nếu index có lưu field frame_filename,
            ngược lại None.
        """
        try:
            hits = self._client.query(
                collection_name=self.collection_name,
                filter=f"video_id == '{video_id}' and frame_id == {int(frame_id)}",
                output_fields=["video_id", "frame_id", "frame_filename"],
                limit=1,
            )
        except Exception as e:
            print(f"[MilvusRetriever] get_frame_filename error: {e}")
            return None

        for hit in hits:
            fn = hit.get("frame_filename")
            if fn:
                return fn
        return None

    def close(self):
        """Giải phóng collection khỏi memory."""
        if self._client:
            try:
                self._client.release_collection(self.collection_name)
            except Exception:
                pass

    def __del__(self):
        self.close()
