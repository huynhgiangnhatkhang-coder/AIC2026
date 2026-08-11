"""
AIC 2026 Baseline — Submission Module
========================================
Xử lý và xuất file kết quả theo format nộp bài của BTC.

Format file nộp (CSV):
  query_id,answer_1,answer_2,...,answer_100
  
Hoặc theo format từng dòng:
  query_id: video_id, frame_id [, answer] [, frame_id_2, ...]
"""
import json
import csv
import os
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


class SubmissionManager:
    """
    Quản lý việc tạo file nộp bài cho AIC2026.
    
    Hỗ trợ 3 loại query và format CSV/JSON output.
    """

    def __init__(self, output_dir: str, max_answers: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_answers = max_answers

    # ── Format per query type ──────────────────────────────

    def format_kis_answer(self, video_id: str, frame_id: int) -> str:
        """Format: video_id, frame_id"""
        return f"{video_id}, {frame_id}"

    def format_qa_answer(self, video_id: str, frame_id: int, answer: str) -> str:
        """Format: video_id, frame_id, answer"""
        return f"{video_id}, {frame_id}, {answer}"

    def format_trake_answer(self, video_id: str, frame_ids: List[int]) -> str:
        """Format: video_id, frame_id_1, frame_id_2, ..., frame_id_N"""
        fids = ", ".join(str(f) for f in frame_ids)
        return f"{video_id}, {fids}"

    # ── Build submission dict ──────────────────────────────

    def build_query_submission(self, query: Dict, results: List[Dict]) -> Dict:
        """
        Build submission cho một query.
        
        Args:
            query: {
                "query_id": str,
                "query_type": str,   # "textual_kis" | "qa" | "trake"
                "query_text": str,
            }
            results: list kết quả từ searcher
            
        Returns:
            {
                "query_id": str,
                "query_type": str,
                "answers": [str, ...]   # list formatted answer strings
            }
        """
        query_id = query.get("query_id", "unknown")
        query_type = query.get("query_type", "textual_kis").lower()

        formatted_answers = []

        for result in results[:self.max_answers]:
            try:
                if query_type in ("textual_kis", "kis"):
                    line = self.format_kis_answer(
                        result["video_id"],
                        result["frame_id"]
                    )
                elif query_type in ("qa", "vqa"):
                    line = self.format_qa_answer(
                        result["video_id"],
                        result["frame_id"],
                        result.get("answer", "")
                    )
                elif query_type == "trake":
                    line = self.format_trake_answer(
                        result["video_id"],
                        result.get("frame_ids", [result.get("frame_id", 0)])
                    )
                else:
                    line = self.format_kis_answer(result["video_id"], result.get("frame_id", 0))

                formatted_answers.append(line)
            except Exception as e:
                print(f"  [WARN] Lỗi format answer: {e}")

        return {
            "query_id": query_id,
            "query_type": query_type,
            "answers": formatted_answers
        }

    # ── Save outputs ───────────────────────────────────────

    def save_json(self, submissions: List[Dict], filename: str = None) -> str:
        """Lưu submission dưới dạng JSON (để debug/review)."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{ts}.json"

        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(submissions, f, ensure_ascii=False, indent=2)

        print(f"[Submission] Saved JSON -> {out_path}")
        return str(out_path)

    def save_csv(self, submissions: List[Dict], filename: str = None) -> str:
        """
        Lưu submission dưới dạng CSV.
        Format: query_id | answer_rank | answer_string
        """
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{ts}.csv"

        out_path = self.output_dir / filename
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query_id", "query_type", "rank", "answer"])

            for sub in submissions:
                for rank, answer in enumerate(sub["answers"], start=1):
                    writer.writerow([
                        sub["query_id"],
                        sub["query_type"],
                        rank,
                        answer
                    ])

        print(f"[Submission] Saved CSV -> {out_path}")
        return str(out_path)

    def save_txt(self, submissions: List[Dict], filename: str = None) -> str:
        """
        Lưu submission dưới dạng TXT (mỗi dòng = 1 answer).
        Format: query_id\tanswer
        """
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{ts}.txt"

        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for sub in submissions:
                for answer in sub["answers"]:
                    f.write(f"{sub['query_id']}\t{answer}\n")

        print(f"[Submission] Saved TXT -> {out_path}")
        return str(out_path)

    def save_all(self, submissions: List[Dict], prefix: str = None) -> Dict[str, str]:
        """Lưu tất cả format."""
        if prefix is None:
            prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

        paths = {
            "json": self.save_json(submissions, f"{prefix}.json"),
            "csv": self.save_csv(submissions, f"{prefix}.csv"),
            "txt": self.save_txt(submissions, f"{prefix}.txt"),
        }

        # Summary
        total_answers = sum(len(s["answers"]) for s in submissions)
        print(f"\n[Submission] Summary:")
        print(f"  Queries:  {len(submissions)}")
        print(f"  Answers:  {total_answers}")
        print(f"  JSON:     {paths['json']}")
        print(f"  CSV:      {paths['csv']}")
        print(f"  TXT:      {paths['txt']}")

        return paths
