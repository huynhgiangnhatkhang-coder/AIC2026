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
from ..retrieval.hybrid_retriever import HybridRetriever


class QASearcher:
    """
    Giải quyết truy vấn Q&A (Visual Question Answering).
    
    Sử dụng BLIP-2 để trả lời câu hỏi trực quan trên top frames.
    Hỗ trợ lazy loading model để tiết kiệm memory khi không cần VQA.
    """

    def __init__(self, retriever: HybridRetriever,
                 vqa_model_name: str = "Salesforce/blip2-opt-2.7b",
                 device: str = "cuda",
                 top_k_frames_for_vqa: int = 5,
                 max_answers: int = 100):
        """
        Args:
            retriever:              HybridRetriever đã khởi tạo
            vqa_model_name:         HuggingFace model ID cho VQA
            device:                 "cuda" hoặc "cpu"
            top_k_frames_for_vqa:   số frame đưa vào VQA model
            max_answers:            số câu trả lời tối đa
        """
        self.retriever = retriever
        self.vqa_model_name = vqa_model_name
        self.device = device
        self.top_k_frames_for_vqa = top_k_frames_for_vqa
        self.max_answers = max_answers
        self._vqa_model = None
        self._vqa_processor = None

    def _load_vqa_model(self):
        """Lazy-load BLIP-2 VQA model."""
        if self._vqa_model is None:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch

            print(f"[QA] Loading VQA model: {self.vqa_model_name}")
            self._vqa_processor = Blip2Processor.from_pretrained(self.vqa_model_name)
            self._vqa_model = Blip2ForConditionalGeneration.from_pretrained(
                self.vqa_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self._vqa_model.eval()
            print("[QA] VQA model loaded!")

    def _run_vqa(self, image_path: str, question: str) -> Tuple[str, float]:
        """
        Chạy VQA model trên một ảnh.
        
        Returns:
            (answer_text, confidence_score)
        """
        self._load_vqa_model()

        import torch
        from PIL import Image

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return ("unknown", 0.0)

        # Dùng Q&A prompt cho BLIP-2
        prompt = f"Question: {question} Answer:"

        inputs = self._vqa_processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._vqa_model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=5,
                output_scores=True,
                return_dict_in_generate=True
            )

        answer = self._vqa_processor.decode(
            outputs.sequences[0], skip_special_tokens=True
        ).strip()

        # Confidence từ beam scores (approximation)
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            confidence = float(outputs.sequences_scores[0].exp().item())
        else:
            confidence = 0.5

        return (answer, confidence)

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
        # Bước 1: Retrieval — tìm top candidate frames
        # Sửa lỗi AIC 2026: Không ghép query và question nữa vì sẽ làm nhiễu không gian vector.
        # Dùng query (mô tả ngữ cảnh) để tìm frame chuẩn xác nhất.
        raw_results = self.retriever.search(
            query=query,
            top_k=self.max_answers * 3
        )

        if not raw_results:
            return []

        answers = []

        if use_vqa:
            # Bước 2: Chạy VQA trên top-K frames
            vqa_candidates = raw_results[:self.top_k_frames_for_vqa]
            vqa_results = []

            for item in vqa_candidates:
                img_path = item.get("frame_path", "")
                if not img_path or not Path(img_path).exists():
                    # Fallback: không có ảnh → skip VQA, dùng dummy answer
                    vqa_results.append((item, "unknown", 0.0))
                    continue

                answer, confidence = self._run_vqa(img_path, question)
                vqa_results.append((item, answer, confidence))
                print(f"  [VQA] {item['video_id']} frame {item['frame_index']}: "
                      f"'{answer}' (conf={confidence:.3f})")

            # Bước 3: Scoring & Ranking có lọc nhiễu
            for item, answer, vqa_conf in vqa_results:
                retrieval_score = item.get("hybrid_score", item.get("score", 0.0))
                
                ans_clean = answer.strip().lower()
                if not ans_clean or ans_clean == "unknown" or ans_clean == "unanswerable":
                    # Phạt cực nặng nếu trả lời unknown hoặc rỗng (tránh frame lỗi lên top)
                    combined_score = retrieval_score * 0.1
                else:
                    combined_score = 0.5 * retrieval_score + 0.5 * vqa_conf
                    
                answers.append({
                    "video_id": item["video_id"],
                    "frame_id": item["frame_index"],
                    "frame_filename": item["frame_filename"],
                    "frame_path": item.get("frame_path", ""),
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
                    "frame_id": item["frame_index"],
                    "frame_filename": item["frame_filename"],
                    "frame_path": item.get("frame_path", ""),
                    "answer": "",  # chưa biết answer
                    "retrieval_score": item.get("hybrid_score", item.get("score", 0.0)),
                    "vqa_confidence": 0.0,
                    # Phạt cực kỳ nặng (0.01) vì không có answer, để không vượt qua các frame VQA
                    "score": item.get("hybrid_score", item.get("score", 0.0)) * 0.01,
                    "rank": 0
                })
        else:
            # Không dùng VQA: chỉ retrieval
            for item in raw_results:
                answers.append({
                    "video_id": item["video_id"],
                    "frame_id": item["frame_index"],
                    "frame_filename": item["frame_filename"],
                    "frame_path": item.get("frame_path", ""),
                    "answer": "",
                    "retrieval_score": item.get("hybrid_score", item.get("score", 0.0)),
                    "vqa_confidence": 0.0,
                    "score": item.get("hybrid_score", item.get("score", 0.0)),
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
            answer = ans.get("answer", "")
            line = f"{ans['video_id']}, {ans['frame_id']}, {answer}"
            lines.append(line)
        return lines
