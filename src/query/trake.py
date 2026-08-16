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
        Thực hiện TRAKE search: Two-Stage Retrieval + Bounded Temporal DP
        """
        N = len(event_queries)
        if N == 0:
            return []

        print(f"[TRAKE] Tìm kiếm {N} sự kiện...")

        # ==============================================================
        # STAGE 1: COARSE RETRIEVAL & VIDEO SCORING
        # ==============================================================
        # Tìm danh sách ứng viên mở rộng cho TẤT CẢ các sự kiện
        coarse_results = []
        video_scores = defaultdict(float)
        
        for i, eq in enumerate(event_queries):
            print(f"  [Stage 1] Lọc thô sự kiện {i+1}/{N}: '{eq}'")
            # Tìm rộng hơn một chút (ví dụ top 1000)
            cands = self.retriever.search(eq, top_k=self.top_k_per_event * 3)
            coarse_results.append(cands)
            
            # Ghi nhận điểm max của mỗi video trong sự kiện này
            vid_max_score = defaultdict(float)
            for c in cands:
                vid = c["video_id"]
                vid_max_score[vid] = max(vid_max_score[vid], c["score"])
                
            # Cộng dồn điểm để tìm các video hứa hẹn nhất (có nhiều sự kiện điểm cao)
            for vid, score in vid_max_score.items():
                video_scores[vid] += score

        if not video_scores:
            return []

        # Chọn Top 50 video tiềm năng nhất
        TOP_M_VIDEOS = 50
        top_videos = sorted(video_scores.keys(), key=lambda v: video_scores[v], reverse=True)[:TOP_M_VIDEOS]
        print(f"[TRAKE] Đã chọn Top {len(top_videos)} videos tiềm năng để chạy Stage 2")

        # ==============================================================
        # STAGE 2: FINE-GRAINED RETRIEVAL (DENSE)
        # ==============================================================
        # Vì milvus-lite dễ crash khi truyền filter IN quá dài, ta sẽ lọc trên Python.
        # Bằng cách lấy top 3000-5000 kết quả và chỉ giữ lại những frame thuộc top_videos.
        top_videos_set = set(top_videos)
        dense_stages_by_video: Dict[str, List[List[Dict]]] = {vid: [[] for _ in range(N)] for vid in top_videos}
        
        for i, eq in enumerate(event_queries):
            print(f"  [Stage 2] Quét sâu sự kiện {i+1}/{N} trong {len(top_videos)} videos (lọc bằng Python)...")
            cands = self.retriever.search(eq, top_k=3000)
            for c in cands:
                vid = c["video_id"]
                if vid in top_videos_set:
                    dense_stages_by_video[vid][i].append(c)

        # ==============================================================
        # STAGE 3: TEMPORAL DYNAMIC PROGRAMMING WITH MAX_GAP CONSTRAINT
        # ==============================================================
        MAX_GAP = 13  # Các sự kiện xảy ra trong vòng ~30 keyframes
        
        answers = []
        for vid, stages in dense_stages_by_video.items():
            # stages là list gồm N list candidates
            # Sắp xếp các candidates trong mỗi stage theo thứ tự thời gian (frame_index)
            for i in range(N):
                stages[i].sort(key=lambda x: x["frame_index"])
            
            # Nếu 1 trong các sự kiện không có candidate nào, ta không drop hẳn video
            # (như code cũ) mà cho phép bỏ qua nếu có thể, 
            # nhưng tốt nhất là DP vẫn cần đủ. Để đơn giản, nếu thiếu hẳn 1 stage thì rớt.
            if any(len(stage) == 0 for stage in stages):
                continue
                
            # Khởi tạo DP: dp[j] là điểm tối đa kết thúc tại candidates[j] của stage hiện tại
            prev_stage_cands = stages[0]
            dp_prev = [c["score"] for c in prev_stage_cands]
            backtrack = [[] for _ in range(N)]
            backtrack[0] = [-1] * len(prev_stage_cands)
            
            valid_sequence_found = True
            
            for i in range(1, N):
                curr_stage_cands = stages[i]
                dp_curr = [float("-inf")] * len(curr_stage_cands)
                back_curr = [-1] * len(curr_stage_cands)
                
                # Trỏ 2 con trỏ hoặc brute-force O(N*M) vì số frame nhỏ
                for j, curr_cand in enumerate(curr_stage_cands):
                    best_prev_idx = -1
                    best_prev_score = float("-inf")
                    
                    for k, prev_cand in enumerate(prev_stage_cands):
                        gap = curr_cand["frame_index"] - prev_cand["frame_index"]
                        # Ràng buộc thời gian: phải xảy ra sau (gap > 0) và khoảng cách ngắn (gap <= MAX_GAP)
                        if 0 < gap <= MAX_GAP:
                            if dp_prev[k] > best_prev_score:
                                best_prev_score = dp_prev[k]
                                best_prev_idx = k
                                
                    if best_prev_idx != -1:
                        # Thưởng thêm nếu frame liên tiếp sát nhau (gap nhỏ)
                        gap_penalty = (gap / MAX_GAP) * 0.05
                        dp_curr[j] = curr_cand["score"] + best_prev_score - gap_penalty
                        back_curr[j] = best_prev_idx
                        
                if all(v == float("-inf") for v in dp_curr):
                    valid_sequence_found = False
                    break
                    
                dp_prev = dp_curr
                prev_stage_cands = curr_stage_cands
                backtrack[i] = back_curr
                
            if not valid_sequence_found:
                continue
                
            # Lấy đỉnh có điểm cao nhất ở stage cuối
            best_last_idx = max(range(len(dp_prev)), key=lambda idx: dp_prev[idx])
            total_score = dp_prev[best_last_idx]
            
            # Truy vết ngược
            seq_events = []
            curr_idx = best_last_idx
            for i in range(N - 1, -1, -1):
                cand = stages[i][curr_idx]
                seq_events.append({
                    "event_index": i,
                    "frame_id": cand["frame_index"],
                    "frame_filename": cand["frame_filename"],
                    "frame_path": cand.get("frame_path", ""),
                    "score": cand["score"]
                })
                curr_idx = backtrack[i][curr_idx]
                
            seq_events.reverse()
            
            answers.append({
                "video_id": vid,
                "frame_ids": [e["frame_id"] for e in seq_events],
                "total_score": total_score,
                "events": seq_events,
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
