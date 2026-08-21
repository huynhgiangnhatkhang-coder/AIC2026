"""
AIC 2026 Baseline — Query Type: TRAKE (Temporal-alignment)
===========================================================
Xử lý truy vấn loại 3: Temporal-alignment
Áp dụng Heuristic Scoring từ Vortex (O(K log K)) kết hợp Florence-2.
"""
from typing import List, Dict, Optional
from collections import defaultdict
from .florence_kis import FlorenceKISSearcher

class TRAKESearcher:
    def __init__(self, kis_searcher: FlorenceKISSearcher,
                 top_k_per_event: int = 100,
                 max_answers: int = 100):
        self.kis_searcher = kis_searcher
        self.top_k_per_event = top_k_per_event
        self.max_answers = max_answers

    def search(self, event_queries: List[str]) -> List[Dict]:
        event_queries = [eq for eq in event_queries if eq.strip()]
        N = len(event_queries)
        if N == 0:
            return []

        print(f"[TRAKE] Tìm kiếm {N} sự kiện với KIS Searcher (Florence-2)...")

        # 1. Tìm ứng viên cho từng sự kiện độc lập
        stages = []
        for i, eq in enumerate(event_queries):
            print(f"  [Stage 1] Tìm kiếm sự kiện {i+1}/{N}: '{eq}'")
            # Sử dụng KIS Searcher để chạy Florence-2 re-ranking cho top 200
            cands = self.kis_searcher.search(eq, search_mode="hybrid", top_k=self.top_k_per_event)
            stages.append(cands)

        # 2. Heuristic Scoring (Vortex): Gộp theo video_id
        video_stage_max = defaultdict(lambda: [float("-inf")] * N)
        for i, stage_cands in enumerate(stages):
            for c in stage_cands:
                vid = c["video_id"]
                score = c.get("score", c.get("clip_score", 0))
                # Sửa lỗi frame_index do milvus trả về frame_id
                if "frame_index" not in c:
                    c["frame_index"] = c["frame_id"]
                if "frame_filename" not in c:
                    c["frame_filename"] = ""
                video_stage_max[vid][i] = max(video_stage_max[vid][i], score)

        # Lọc các video có mặt ở TẤT CẢ các stage (điểm > -inf)
        valid_videos = {}
        for vid, max_scores in video_stage_max.items():
            if all(s > float("-inf") for s in max_scores):
                valid_videos[vid] = sum(max_scores)

        if not valid_videos:
            print("[TRAKE] Không có video nào chứa đầy đủ các sự kiện trong top K.")
            return []

        # Chọn Top 50 video tiềm năng nhất để chạy DP check thứ tự
        TOP_M_VIDEOS = 50
        top_videos = sorted(valid_videos.keys(), key=lambda v: valid_videos[v], reverse=True)[:TOP_M_VIDEOS]
        top_videos_set = set(top_videos)
        print(f"[TRAKE] Đã chọn Top {len(top_videos)} videos tiềm năng để check thứ tự thời gian.")

        # Lọc ra các frame thuộc top_videos
        dense_stages_by_video = {vid: [[] for _ in range(N)] for vid in top_videos}
        for i, stage_cands in enumerate(stages):
            for c in stage_cands:
                vid = c["video_id"]
                if vid in top_videos_set:
                    dense_stages_by_video[vid][i].append(c)

        # 3. Temporal DP cực nhanh trên Top Videos
        answers = []

        for vid, vid_stages in dense_stages_by_video.items():
            for i in range(N):
                vid_stages[i].sort(key=lambda x: x["frame_index"])
            
            if any(len(stage) == 0 for stage in vid_stages):
                continue
                
            best_seq = None
            best_score = float("-inf")
            
            def dfs(stage_idx, current_seq, current_score):
                nonlocal best_seq, best_score
                if stage_idx == N:
                    if current_score > best_score:
                        best_score = current_score
                        best_seq = list(current_seq)
                    return
                
                for cand in vid_stages[stage_idx]:
                    if stage_idx == 0:
                        dfs(stage_idx + 1, [cand], cand.get("score", 0))
                    else:
                        prev_cand = current_seq[-1]
                        gap = cand["frame_index"] - prev_cand["frame_index"]
                        # Chỉ yêu cầu đúng thứ tự thời gian (gap > 0)
                        if gap > 0:
                            # Phạt rất nhẹ khoảng cách để ưu tiên các kết quả gần nhau hơn nếu điểm bằng nhau
                            gap_penalty = gap * 0.0001
                            new_score = current_score + cand.get("score", 0) - gap_penalty
                            dfs(stage_idx + 1, current_seq + [cand], new_score)

            dfs(0, [], 0)
            
            if best_seq is None:
                continue
            
            seq_events = []
            for i, cand in enumerate(best_seq):
                seq_events.append({
                    "event_index": i,
                    "frame_id": cand["frame_index"],
                    "frame_filename": cand.get("frame_filename", ""),
                    "frame_path": cand.get("frame_path", ""),
                    "score": cand.get("score", 0)
                })
                
            answers.append({
                "video_id": vid,
                "frame_ids": [e["frame_id"] for e in seq_events],
                "total_score": best_score,
                "events": seq_events,
                "rank": 0
            })

        answers = sorted(answers, key=lambda x: x["total_score"], reverse=True)
        for i, ans in enumerate(answers[:self.max_answers]):
            ans["rank"] = i + 1

        return answers[:self.max_answers]

    def format_submission(self, answers: List[Dict]) -> List[str]:
        lines = []
        for ans in answers:
            frame_ids_str = ", ".join(str(fid) for fid in ans["frame_ids"])
            line = f"{ans['video_id']}, {frame_ids_str}"
            lines.append(line)
        return lines
