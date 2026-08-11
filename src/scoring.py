"""
AIC 2026 Baseline — Scoring Module
=====================================
Tính toán R-Score và Final Score theo đúng tiêu chí của BTC.

Dùng để đánh giá kết quả trên tập validation (nếu có ground truth).
"""
from typing import List, Dict, Optional, Tuple


# ──────────────────────────────────────────────
# R-Score per query type
# ──────────────────────────────────────────────

def r_score_textual_kis(answer: Dict, ground_truth: Dict) -> float:
    """
    Tính R-Score cho truy vấn Textual KIS.
    
    Điều kiện đúng:
      - video_id khớp: answer["video_id"] == gt["video_id"]
      - frame_id nằm trong đoạn [s, e]: gt["frame_start"] <= answer["frame_id"] <= gt["frame_end"]
      
    Args:
        answer: {"video_id": str, "frame_id": int}
        ground_truth: {"video_id": str, "frame_start": int, "frame_end": int}
        
    Returns:
        R-Score ∈ {0, 1}
    """
    if answer.get("video_id") != ground_truth.get("video_id"):
        return 0.0

    frame_id = answer.get("frame_id", -1)
    s = ground_truth.get("frame_start", 0)
    e = ground_truth.get("frame_end", 0)

    if s <= frame_id <= e:
        return 1.0
    return 0.0


def r_score_qa(answer: Dict, ground_truth: Dict) -> float:
    """
    Tính R-Score cho truy vấn Q&A (VQA).
    
    Điều kiện đúng:
      - video_id khớp
      - frame_id trong [s, e]
      - answer khớp ngữ nghĩa với gt_answer
      
    Note: Trong baseline, ta dùng exact/substring match thay vì semantic match thực sự.
          Trong thực tế BTC sử dụng semantic match (có thể dùng LLM để đánh giá).
    """
    if answer.get("video_id") != ground_truth.get("video_id"):
        return 0.0

    frame_id = answer.get("frame_id", -1)
    s = ground_truth.get("frame_start", 0)
    e = ground_truth.get("frame_end", 0)

    if not (s <= frame_id <= e):
        return 0.0

    # Semantic answer matching (simplified: substring or exact)
    pred_answer = str(answer.get("answer", "")).strip().lower()
    gt_answer = str(ground_truth.get("answer", "")).strip().lower()

    if not gt_answer:
        return 0.0

    # Exact match
    if pred_answer == gt_answer:
        return 1.0
    # Substring match (simplified semantic)
    if gt_answer in pred_answer or pred_answer in gt_answer:
        return 0.8
    # Partial token overlap
    pred_tokens = set(pred_answer.split())
    gt_tokens = set(gt_answer.split())
    overlap = pred_tokens & gt_tokens
    if overlap:
        return len(overlap) / max(len(gt_tokens), 1) * 0.7

    return 0.0


def r_score_trake(answer: Dict, ground_truth: Dict) -> float:
    """
    Tính R-Score cho truy vấn TRAKE.
    
    Điều kiện:
      - video_id phải khớp (nếu sai → 0.0 ngay)
      - Mỗi frame_id_j được coi là đúng nếu nằm trong đoạn [s_j, e_j]
      - R-Score = (số frame đúng) / N
      
    Args:
        answer: {"video_id": str, "frame_ids": [int, ...]}
        ground_truth: {
            "video_id": str,
            "events": [{"frame_start": int, "frame_end": int}, ...]
        }
    """
    if answer.get("video_id") != ground_truth.get("video_id"):
        return 0.0

    frame_ids = answer.get("frame_ids", [])
    events = ground_truth.get("events", [])
    N = len(events)

    if N == 0 or len(frame_ids) == 0:
        return 0.0

    correct = 0
    for j, event in enumerate(events):
        if j >= len(frame_ids):
            break
        fi = frame_ids[j]
        s = event.get("frame_start", 0)
        e = event.get("frame_end", 0)
        if s <= fi <= e:
            correct += 1

    return correct / N


# ──────────────────────────────────────────────
# Final Score calculation
# ──────────────────────────────────────────────

def compute_r_at_k(r_scores: List[float], k: int) -> float:
    """
    R@k = max R-Score trong k câu trả lời đầu tiên.
    """
    top_k = r_scores[:k]
    return max(top_k) if top_k else 0.0


def compute_final_score(r_scores: List[float]) -> float:
    """
    Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5
    
    Args:
        r_scores: list R-Score theo thứ tự submit (index 0 = câu đầu tiên)
        
    Returns:
        Final Score ∈ [0, 1]
    """
    thresholds = [1, 5, 20, 50, 100]
    r_at_k_values = [compute_r_at_k(r_scores, k) for k in thresholds]
    return sum(r_at_k_values) / len(thresholds)


# ──────────────────────────────────────────────
# Evaluate full query set
# ──────────────────────────────────────────────

def evaluate_query(
    query_type: str,
    answers: List[Dict],
    ground_truth: Dict
) -> Dict:
    """
    Đánh giá một query đơn lẻ.
    
    Args:
        query_type: "textual_kis" | "qa" | "trake"
        answers:    list câu trả lời theo thứ tự (tối đa 100)
        ground_truth: dict chứa đáp án đúng
        
    Returns:
        {
            "r_scores": [...],
            "r_at_1": float, "r_at_5": float, "r_at_20": float,
            "r_at_50": float, "r_at_100": float,
            "final_score": float
        }
    """
    scorer = {
        "textual_kis": r_score_textual_kis,
        "kis": r_score_textual_kis,
        "qa": r_score_qa,
        "vqa": r_score_qa,
        "trake": r_score_trake,
    }.get(query_type.lower(), r_score_textual_kis)

    r_scores = [scorer(ans, ground_truth) for ans in answers]

    thresholds = [1, 5, 20, 50, 100]
    r_at_k = {f"r_at_{k}": compute_r_at_k(r_scores, k) for k in thresholds}
    final = compute_final_score(r_scores)

    return {
        "r_scores": r_scores,
        **r_at_k,
        "final_score": final
    }


def evaluate_dataset(queries: List[Dict]) -> Dict:
    """
    Đánh giá toàn bộ tập query.
    
    Args:
        queries: list của {
            "query_id": str,
            "query_type": str,
            "answers": [...],
            "ground_truth": {...}
        }
        
    Returns:
        {
            "mean_final_score": float,
            "per_query": [...],
            "per_type": {...}
        }
    """
    per_query = []
    per_type: Dict[str, List[float]] = {}

    for q in queries:
        result = evaluate_query(
            q["query_type"],
            q["answers"],
            q["ground_truth"]
        )
        result["query_id"] = q.get("query_id", "")
        result["query_type"] = q["query_type"]
        per_query.append(result)

        qtype = q["query_type"]
        if qtype not in per_type:
            per_type[qtype] = []
        per_type[qtype].append(result["final_score"])

    mean_final = sum(r["final_score"] for r in per_query) / len(per_query) if per_query else 0.0

    return {
        "mean_final_score": mean_final,
        "num_queries": len(per_query),
        "per_query": per_query,
        "per_type_mean": {
            qtype: sum(scores) / len(scores)
            for qtype, scores in per_type.items()
        }
    }


def print_evaluation_report(eval_result: Dict):
    """In báo cáo đánh giá ra màn hình."""
    print("\n" + "="*60)
    print("  AIC2026 EVALUATION REPORT")
    print("="*60)
    print(f"  Total queries:     {eval_result['num_queries']}")
    print(f"  Mean Final Score:  {eval_result['mean_final_score']:.4f}")
    print()
    print("  Per type breakdown:")
    for qtype, score in eval_result.get("per_type_mean", {}).items():
        print(f"    {qtype:15s}: {score:.4f}")
    print("="*60 + "\n")
