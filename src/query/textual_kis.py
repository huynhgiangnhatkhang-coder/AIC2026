"""
AIC 2026 Baseline — Query Type: Textual KIS (CLIP + Grounding DINO)
=====================================================================
Xử lý truy vấn loại 1: Textual Known-Item Search

Pipeline (giống AIC2026/search.py):
  1. Dịch query tiếng Việt → tiếng Anh (offline, Opus-MT)
  2. Lọc thô bằng Milvus/CLIP → Top-N candidates (thường 30)
  3. Re-rank bằng Grounding DINO (chấm điểm tổng hợp: alpha*CLIP + (1-alpha)*DINO)
  4. Trả về top-K kết quả cuối cùng

Format nộp: video_id, frame_id
"""
from typing import List, Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from .translate import analyze_query_offline_mt


class TextualKISSearcher:
    """
    Giải quyết truy vấn Textual KIS bằng CLIP + Grounding DINO.

    Loại bỏ BM25. Thay vào đó dùng:
      - Milvus/CLIP để lọc thô (top_k_coarse candidates)
      - Grounding DINO để re-rank ensemble scoring
    """

    def __init__(self,
                 retriever,                          # MilvusRetriever hoặc CLIPRetriever
                 keyframes_dir: str,                 # Đường dẫn gốc tới thư mục chứa ảnh
                 max_answers: int = 100,
                 top_k_coarse: int = 30,             # Số frame lấy từ CLIP trước khi DINO
                 alpha: float = 0.75,                # Trọng số CLIP (DINO = 1 - alpha)
                 dino_threshold: float = 0.05,       # Ngưỡng detect tối thiểu của DINO
                 dino_model_name: str = "IDEA-Research/grounding-dino-base",
                 device: Optional[str] = None,
                 # Tương thích ngược với baseline cũ (bỏ qua)
                 objects_dir: Optional[str] = None,
                 max_frames_per_video: int = 10):
        """
        Args:
            retriever:       MilvusRetriever đã khởi tạo
            keyframes_dir:   Đường dẫn gốc tới Dataset, ví dụ:
                             "f:/aic/AIC2026/DATASET"
                             Cấu trúc bên trong phải là:
                             <keyframes_dir>/Keyframes_L21/keyframes/L21_V001/0001.jpg
            max_answers:     Số kết quả cuối cùng trả về (default=100)
            top_k_coarse:    Số frame CLIP lấy ra để DINO chấm điểm (default=30)
            alpha:           Trọng số CLIP trong công thức ensemble (default=0.75)
            dino_threshold:  Ngưỡng tự tin tối thiểu của DINO để ghi nhận box (default=0.05)
            dino_model_name: Tên model Grounding DINO (HuggingFace ID)
            device:          "cuda" hoặc "cpu" (None = tự detect)
        """
        self.retriever = retriever
        self.keyframes_dir = keyframes_dir
        self.max_answers = max_answers
        self.top_k_coarse = top_k_coarse
        self.alpha = alpha
        self.dino_threshold = dino_threshold
        self.max_frames_per_video = max_frames_per_video  # giữ tương thích

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"

        # Lazy-load DINO (chỉ load khi search lần đầu)
        self._dino_processor = None
        self._dino_model = None
        self._dino_model_name = dino_model_name

        print(f"[TextualKISSearcher] Khởi tạo xong | device={self.device} | "
              f"alpha={self.alpha} | top_k_coarse={self.top_k_coarse}")

    # ──────────────────────────────────────────
    # GROUNDING DINO
    # ──────────────────────────────────────────

    def _load_dino(self):
        """Lazy-load Grounding DINO (chỉ load 1 lần)."""
        if self._dino_model is not None:
            return
        print(f"[DINO] Đang tải Grounding DINO: {self._dino_model_name} ...")
        self._dino_processor = AutoProcessor.from_pretrained(self._dino_model_name)
        self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._dino_model_name
        ).to(self.device)
        self._dino_model.eval()
        print(f"[DINO] Grounding DINO đã tải xong | device={self.device}")

    def _get_dino_score(self, image_path: str, text_prompt: str) -> float:
        """
        Tính điểm Grounding DINO cho một ảnh với text_prompt.
        Dùng toàn bộ câu query (không tách object) để lấy điểm chính xác nhất.
        Không dùng threshold cứng để loại — chỉ lấy điểm cao nhất tìm được.

        Returns:
            float: Điểm tự tin cao nhất từ DINO (0.0 nếu không detect được gì)
        """
        self._load_dino()

        # DINO yêu cầu prompt viết thường và kết thúc bằng dấu chấm
        dino_query = text_prompt.lower().strip()
        if not dino_query.endswith("."):
            dino_query += " ."

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self._dino_processor(
                images=image, text=dino_query, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self._dino_model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = self._dino_processor.image_processor.post_process_object_detection(
                outputs, threshold=self.dino_threshold, target_sizes=target_sizes
            )[0]

            if len(results["scores"]) > 0:
                return results["scores"].max().item()
            return 0.0

        except Exception as e:
            print(f"[DINO] ❌ Lỗi đọc ảnh {image_path}: {e}")
            return 0.0

    # ──────────────────────────────────────────
    # TÌM ĐƯỜNG DẪN ẢNH GỐC
    # ──────────────────────────────────────────

    def _find_image_path(self, video_id: str, frame_id: int) -> Optional[str]:
        """
        Tìm đường dẫn tới file ảnh keyframe theo cấu trúc của BTC AIC:
          <keyframes_dir>/Keyframes_<batch>/keyframes/<video_folder>/<frame_id>.jpg
          hoặc <keyframes_dir>/<video_folder>/<frame_id>.jpg

        Hỗ trợ nhiều định dạng tên frame: 1.jpg, 001.jpg, 0001.jpg, 00001.jpg, 000001.jpg
        và cả .png.
        """
        from src.frame_paths import resolve_frame_path
        return resolve_frame_path(
            video_id, frame_id, cfg=None, extra_roots=[self.keyframes_dir]
        )

    # ──────────────────────────────────────────
    # SEARCH CHÍNH
    # ──────────────────────────────────────────

    def search(self, query: str,
               object_hints: Optional[List[str]] = None) -> List[Dict]:
        """
        Thực hiện Textual KIS search bằng CLIP + Grounding DINO.

        Args:
            query:        Mô tả văn bản (tiếng Việt hoặc tiếng Anh)
            object_hints: Không còn dùng (giữ lại để tương thích với baseline cũ)

        Returns:
            List[Dict] sắp xếp theo final_score giảm dần:
            [{
                "video_id":      str,
                "frame_id":      int,
                "frame_index":   int,   (alias)
                "frame_filename": str,
                "clip_score":    float,
                "dino_score":    float,
                "score":         float, (= final_score)
                "rank":          int
            }]
        """
        # Bước 1: Dịch query tiếng Việt → tiếng Anh
        parsed = analyze_query_offline_mt(query)
        english_query = parsed["clip_query"]

        # Bước 2: Lọc thô bằng Milvus/CLIP → lấy Top-N candidates
        print(f"\n[CLIP] Đang lấy Top {self.top_k_coarse} candidates từ Milvus...")
        raw_results = self.retriever.search(english_query, top_k=self.top_k_coarse)

        if not raw_results:
            print("[KIS] Không có kết quả từ CLIP retriever.")
            return []

        # Bước 3: Grounding DINO re-rank
        print(f"[DINO] Đang chấm điểm tổng hợp (Ensemble) cho {len(raw_results)} frames...")
        scored_frames = []

        for hit in raw_results:
            v_id = hit["video_id"]
            f_id = hit.get("frame_id", hit.get("frame_index", 0))
            clip_score = float(hit.get("score", hit.get("hybrid_score", 0.0)))

            # Tìm đường dẫn ảnh gốc
            image_path = self._find_image_path(v_id, f_id)
            if not image_path:
                print(f"[KIS] ❌ Không tìm thấy ảnh: {v_id} - Frame {f_id}")
                # Vẫn giữ lại với dino_score=0 để không bỏ mất candidate
                dino_score = 0.0
            else:
                dino_score = self._get_dino_score(image_path, english_query)

            final_score = (self.alpha * clip_score) + ((1 - self.alpha) * dino_score)

            scored_frames.append({
                "video_id":       v_id,
                "frame_id":       f_id,
                "frame_index":    f_id,
                "frame_filename": hit.get("frame_filename", f"{f_id}.jpg"),
                "clip_score":     clip_score,
                "dino_score":     dino_score,
                "score":          final_score,
                "hybrid_score":   final_score,
                "rank":           0   # sẽ cập nhật sau khi sort
            })

        # Bước 4: Sort theo final_score giảm dần
        scored_frames.sort(key=lambda x: x["score"], reverse=True)

        # Gán rank và giới hạn kết quả
        answers = []
        for rank, frame in enumerate(scored_frames[:self.max_answers], 1):
            frame["rank"] = rank
            answers.append(frame)

        print(f"\n[KIS] === KẾT QUẢ TÌM KIẾM ===")
        for frame in answers[:10]:
            print(f"  Top {frame['rank']:3d}: {frame['video_id']} | frame {frame['frame_id']:6d} "
                  f"| final={frame['score']:.4f} (CLIP={frame['clip_score']:.4f} | DINO={frame['dino_score']:.4f})")
        if len(answers) > 10:
            print(f"  ... ({len(answers)} total)")

        return answers

    def format_submission(self, answers: List[Dict]) -> List[str]:
        """
        Format kết quả theo định dạng nộp bài AIC2026.
        Output format: "video_id, frame_id"
        """
        lines = []
        for ans in answers:
            line = f"{ans['video_id']}, {ans['frame_id']}"
            lines.append(line)
        return lines
