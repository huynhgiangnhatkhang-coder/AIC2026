"""
AIC 2026 Baseline — Query Type: Q&A (Visual Question Answering)
================================================================
Xử lý truy vấn loại 2: Visual Question Answering
- Input:  câu hỏi về nội dung của một khung hình/video
- Output: (video_id, frame_id, answer)

Format nộp: video_id, frame_id, answer

Pipeline:
  1. Textual KIS để tìm top candidate frames
  2. Chạy VQA model (BLIP-2) trên top-K frames
  3. Kết hợp retrieval score + answer confidence → rank
"""
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import base64
import requests
from ..retrieval.hybrid_retriever import HybridRetriever
from .florence_kis import FlorenceKISSearcher


class QASearcher:
    """
    Giải quyết truy vấn Q&A (Visual Question Answering).
    
    Sử dụng BLIP-2 để trả lời câu hỏi trực quan trên top frames.
    Hỗ trợ lazy loading model để tiết kiệm memory khi không cần VQA.
    """

    def __init__(self, kis_searcher: FlorenceKISSearcher,
                 vqa_model_name: str = "local-model",
                 api_url: str = "http://aicpc.sytes.net:1234/v1/chat/completions",
                 device: str = "cuda",
                 top_k_frames_for_vqa: int = 5,
                 max_answers: int = 100,
                 keyframes_dir: str = "DATASET"):
        """
        Args:
            kis_searcher:           FlorenceKISSearcher đã khởi tạo
            vqa_model_name:         Model giả lập cho LM Studio
            api_url:                URL của máy chủ LM Studio
            device:                 "cuda" hoặc "cpu"
            top_k_frames_for_vqa:   số frame đưa vào VQA model
            max_answers:            số câu trả lời tối đa
            keyframes_dir:          thư mục chứa ảnh gốc
        """
        self.kis_searcher = kis_searcher
        self.vqa_model_name = vqa_model_name
        self.api_url = api_url
        self.device = device
        self.top_k_frames_for_vqa = top_k_frames_for_vqa
        self.max_answers = max_answers
        self.keyframes_dir = keyframes_dir

    def _resolve_image_path(self, video_id: str, frame_id: int) -> Optional[str]:
        video_folder = video_id.replace(".mp4", "")
        batch_prefix = video_folder.split("_")[0]
        batch_folder_name = f"Keyframes_{batch_prefix}"
        import os
        possible_formats = [
            f"{frame_id}.jpg", f"{frame_id:03d}.jpg", f"{frame_id:04d}.jpg",
            f"{frame_id:05d}.jpg", f"{frame_id:06d}.jpg", f"{frame_id}.png",
            f"{frame_id:04d}.png",
        ]
        for fmt in possible_formats:
            temp_path = os.path.join(
                self.keyframes_dir, batch_folder_name, "keyframes",
                video_folder, fmt,
            )
            if os.path.exists(temp_path):
                return temp_path
        return None

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encodes a local image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _run_vqa(self, image_path: str, question: str) -> Tuple[str, float]:
        """
        Chạy VQA bằng cách gửi request tới LM Studio Server.
        
        Returns:
            (answer_text, confidence_score)
        """
        try:
            base64_image = self._encode_image_to_base64(image_path)
            
            payload = {
                "model": self.vqa_model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant for video visual question answering (VQA). You must NOT output internal thoughts or reasoning. Skip all analysis and output ONLY the final answer precisely and concisely."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.2,
                "stream": False,
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                return (answer, 0.9)
            else:
                print(f"[VQA Error] HTTP {response.status_code}: {response.text}")
                return ("unknown", 0.1)

        except Exception as e:
            print(f"[VQA Error] {e}")
            return ("unknown", 0.1)

    def search(self, query: str, question: str,
               use_vqa: bool = True) -> List[Dict]:
        """
        Thực hiện Q&A search.
        
        Args:
            query:    mô tả cảnh cần tìm (dùng cho retrieval)
            question: câu hỏi VQA (dùng cho VQA model)
            use_vqa:  có chạy VQA hay không (False = chỉ retrieval, answer = "")
            
        Returns:
            List of answer dicts:
            [{
                "video_id": str,
                "frame_id": int,
                "answer": str,
                "score": float,
                "vqa_confidence": float,
                "rank": int
            }]
        """
        # Bước 1: Retrieval — tìm top candidate frames bằng toàn bộ KIS pipeline
        # Áp dụng nguyên xi KIS sang với top_k = 300
        raw_results = self.kis_searcher.search(
            raw_query=query,
            search_mode="visual",  # Có thể dùng "hybrid", nhưng visual nhanh hơn
            top_k=75
        )

        if not raw_results:
            return []

        answers = []

        if use_vqa:
            # Bước 2: Chạy VQA trên top-K frames
            vqa_candidates = raw_results[:self.top_k_frames_for_vqa]
            vqa_results = []

            for item in vqa_candidates:
                # KIS đã check path tồn tại, nhưng cẩn thận check lại
                img_path = item.get("image_path")
                if not img_path:
                    img_path = self._resolve_image_path(item["video_id"], item["frame_id"])
                
                if not img_path:
                    # Fallback: không có ảnh → skip VQA, dùng dummy answer
                    vqa_results.append((item, "unknown", 0.0))
                    continue

                answer, confidence = self._run_vqa(img_path, question)
                vqa_results.append((item, answer, confidence))
                print(f"  [VQA] {item['video_id']} frame {item['frame_id']}: "
                      f"'{answer}' (conf={confidence:.3f})")

            # Bước 3: Scoring & Ranking có lọc nhiễu
            for item, answer, vqa_conf in vqa_results:
                retrieval_score = item.get("score", 0.0)
                
                ans_clean = answer.strip().lower()
                if not ans_clean or ans_clean == "unknown" or ans_clean == "unanswerable":
                    # Phạt cực nặng nếu trả lời unknown hoặc rỗng (tránh frame lỗi lên top)
                    combined_score = retrieval_score * 0.1
                else:
                    combined_score = 0.5 * retrieval_score + 0.5 * vqa_conf
                    
                answers.append({
                    "video_id": item["video_id"],
                    "frame_id": item["frame_id"],
                    "frame_filename": item.get("frame_filename", f"{item['frame_id']:04d}.jpg"),
                    "frame_path": item.get("image_path", ""),
                    "answer": answer,
                    "retrieval_score": retrieval_score,
                    "vqa_confidence": vqa_conf,
                    "score": combined_score,
                    "rank": 0
                })

            # Thêm remaining frames (không chạy VQA) với answer rỗng
            for item in raw_results[self.top_k_frames_for_vqa:]:
                if len(answers) >= self.max_answers:
                    break
                answers.append({
                    "video_id": item["video_id"],
                    "frame_id": item["frame_id"],
                    "frame_filename": item.get("frame_filename", f"{item['frame_id']:04d}.jpg"),
                    "frame_path": item.get("image_path", ""),
                    "answer": "",  # chưa biết answer
                    "retrieval_score": item.get("score", 0.0),
                    "vqa_confidence": 0.0,
                    # Phạt cực kỳ nặng (0.01) vì không có answer, để không vượt qua các frame VQA
                    "score": item.get("score", 0.0) * 0.01,
                    "rank": 0
                })
        else:
            # Không dùng VQA: chỉ retrieval
            for item in raw_results:
                answers.append({
                    "video_id": item["video_id"],
                    "frame_id": item["frame_id"],
                    "frame_filename": item.get("frame_filename", f"{item['frame_id']:04d}.jpg"),
                    "frame_path": item.get("image_path", ""),
                    "answer": "",
                    "retrieval_score": item.get("score", 0.0),
                    "vqa_confidence": 0.0,
                    "score": item.get("score", 0.0),
                    "rank": 0
                })

        # Sort và assign ranks
        answers = sorted(answers, key=lambda x: x["score"], reverse=True)
        for i, ans in enumerate(answers[:self.max_answers]):
            ans["rank"] = i + 1

        return answers[:self.max_answers]

    def format_submission(self, answers: List[Dict]) -> List[str]:
        """
        Format kết quả theo định dạng nộp bài AIC2026.
        
        Output format: "video_id, frame_id, answer"
        """
        lines = []
        for ans in answers:
            from src.frame_id_map import frame_id_from_index
            answer = ans.get("answer", "")
            frame_id = frame_id_from_index(ans["video_id"], ans["frame_id"])
            line = f"{ans['video_id']}, {frame_id}, {answer}"
            lines.append(line)
        return lines
