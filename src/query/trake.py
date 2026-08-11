"""
AIC 2026 Baseline — Query Type: TRAKE (Temporal-alignment)
===========================================================
Xử lý truy vấn loại 3: Temporal-alignment
- Input:  N mô tả sự kiện theo thứ tự thời gian
- Output: (video_id, frame_id_1, ..., frame_id_N)

Format nộp: video_id, frame_id_1, frame_id_2, ..., frame_id_N

Pipeline:
  1. Mỗi sub-event → CLIP retrieval để lấy candidate frames
  2. Gom theo video
  3. Dynamic Programming (monotone chain) để tìm chuỗi frame
     có thứ tự tăng dần và tổng score cao nhất
  4. Rank theo total_score
"""
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from ..retrieval.clip_retriever import CLIPRetriever


class TRAKESearcher:
    """
    Giải quyết truy vấn TRAKE (Temporal-alignment).
    
    Tìm chuỗi N frame theo đúng thứ tự thời gian trong cùng 1 video.
    Dùng thuật toán DP monotone chain (O(N*K*logK)).
    """

    def __init__(self, clip_retriever: CLIPRetriever,
                 top_k_per_event: int = 300,
                 max_answers: int = 100):
        """
        Args:
            clip_retriever:    CLIPRetriever đã khởi tạo
            top_k_per_event:   số frame candidate cho mỗi sub-event
            max_answers:       số video kết quả trả về
        """
        self.retriever = clip_retriever
        self.top_k_per_event = top_k_per_event
        self.max_answers = max_answers

    def _search_event(self, event_query: str) -> List[Dict]:
        """
        Tìm frame candidates cho một sub-event.
        """
        return self.retriever.search(event_query, top_k=self.top_k_per_event)

    def _dp_monotone_chain(self, stages_by_video: Dict[str, List[List[Dict]]]) -> Dict[str, Dict]:
        """
        DP Monotone Chain: tìm chuỗi frame tối ưu trong mỗi video.
        
        Đảm bảo: frame_index_1 < frame_index_2 < ... < frame_index_N
        Tối đa hóa: tổng score của các frame được chọn.
        
        Args:
            stages_by_video: {video_id: [[event0_frames], [event1_frames], ...]}
            
        Returns:
            {video_id: {"total_score": float, "events": [frame_info, ...]}}
        """
        results = {}

        for video_id, stages in stages_by_video.items():
            N = len(stages)
            if N == 0:
                continue
            if any(len(st) == 0 for st in stages):
                continue  # video không có đủ frames cho tất cả events

            # Sort mỗi stage theo frame_index tăng dần
            cands = [sorted(st, key=lambda x: x["frame_index"]) for st in stages]

            # DP
            # dp_prev[j] = best total score khi chọn frame j ở stage hiện tại
            dp_prev = [c["score"] for c in cands[0]]
            back = [[-1] * len(cands[i]) for i in range(N)]  # backtracking

            for i in range(1, N):
                a = cands[i - 1]  # prev stage
                b = cands[i]      # current stage
                dp_curr = [float("-inf")] * len(b)
                back_curr = [-1] * len(b)

                # Monotone scan: a và b đã sort theo frame_index
                k = 0
                best_val = float("-inf")
                best_k = -1

                for j in range(len(b)):
                    fi_j = b[j]["frame_index"]
                    # Advance k: thêm tất cả a[k] có frame_index < fi_j
                    while k < len(a) and a[k]["frame_index"] < fi_j:
                        if dp_prev[k] > best_val:
                            best_val = dp_prev[k]
                            best_k = k
                        k += 1

                    if best_k != -1:
                        dp_curr[j] = b[j]["score"] + best_val
                        back_curr[j] = best_k

                dp_prev = dp_curr
                back[i] = back_curr

            # Tìm kết thúc tốt nhất
            if all(v == float("-inf") for v in dp_prev):
                continue

            j_best = max(range(len(dp_prev)), key=lambda j: dp_prev[j])
            if dp_prev[j_best] == float("-inf"):
                continue

            total_score = dp_prev[j_best]

            # Backtrack để lấy path
            path = []
            cur_j = j_best
            for i in range(N - 1, -1, -1):
                frame = cands[i][cur_j]
                path.append({
                    "event_index": i,
                    "frame_id": frame["frame_index"],
                    "frame_filename": frame["frame_filename"],
                    "frame_path": frame.get("frame_path", ""),
                    "score": frame["score"]
                })
                if i > 0:
                    cur_j = back[i][cur_j]
                    if cur_j == -1:
                        break

            path.reverse()

            if len(path) == N:  # chỉ lấy kết quả có đủ N events
                results[video_id] = {
                    "total_score": total_score,
                    "events": path
                }

        return results

    def search(self, event_queries: List[str]) -> List[Dict]:
        """
        Thực hiện TRAKE search.
        
        Args:
            event_queries: list mô tả N sự kiện theo thứ tự thời gian
                          Ví dụ: ["vận động viên giậm nhảy", "bay qua xà", "tiếp đất"]
                          
        Returns:
            List of answer dicts, sorted by total_score:
            [{
                "video_id": str,
                "frame_ids": [int, ...],   # N frame ids, thứ tự tăng dần
                "total_score": float,
                "events": [{"event_index": int, "frame_id": int, ...}],
                "rank": int
            }]
        """
        N = len(event_queries)
        if N == 0:
            return []

        print(f"[TRAKE] Tìm kiếm {N} sự kiện...")

        # Bước 1: Tìm candidates cho mỗi event
        all_stage_results = []
        for i, eq in enumerate(event_queries):
            print(f"  [Event {i+1}/{N}] Query: '{eq}'")
            candidates = self._search_event(eq)
            all_stage_results.append(candidates)
            print(f"    → {len(candidates)} candidates")

        # Bước 2: Gom theo video
        stages_by_video: Dict[str, List[List[Dict]]] = defaultdict(
            lambda: [[] for _ in range(N)]
        )

        for stage_idx, stage_cands in enumerate(all_stage_results):
            for item in stage_cands:
                vid = item["video_id"]
                stages_by_video[vid][stage_idx].append(item)

        print(f"[TRAKE] {len(stages_by_video)} videos có candidates")

        # Bước 3: DP trên từng video
        if N == 1:
            # Special case: chỉ 1 event → không cần DP
            answers = []
            for vid, stages in stages_by_video.items():
                if not stages[0]:
                    continue
                best = max(stages[0], key=lambda x: x["score"])
                answers.append({
                    "video_id": vid,
                    "frame_ids": [best["frame_index"]],
                    "total_score": best["score"],
                    "events": [{"event_index": 0, "frame_id": best["frame_index"],
                                "score": best["score"]}],
                    "rank": 0
                })
        else:
            best_by_video = self._dp_monotone_chain(dict(stages_by_video))
            answers = []
            for vid, seq in best_by_video.items():
                frame_ids = [e["frame_id"] for e in seq["events"]]
                answers.append({
                    "video_id": vid,
                    "frame_ids": frame_ids,
                    "total_score": seq["total_score"],
                    "events": seq["events"],
                    "rank": 0
                })

        # Bước 4: Sort và rank
        answers = sorted(answers, key=lambda x: x["total_score"], reverse=True)
        for i, ans in enumerate(answers[:self.max_answers]):
            ans["rank"] = i + 1

        return answers[:self.max_answers]

    def format_submission(self, answers: List[Dict]) -> List[str]:
        """
        Format kết quả theo định dạng nộp bài AIC2026.
        
        Output format: "video_id, frame_id_1, frame_id_2, ..., frame_id_N"
        """
        lines = []
        for ans in answers:
            frame_ids_str = ", ".join(str(fid) for fid in ans["frame_ids"])
            line = f"{ans['video_id']}, {frame_ids_str}"
            lines.append(line)
        return lines
