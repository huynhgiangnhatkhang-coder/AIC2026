"""
AIC 2026 Baseline — CLI Entry Point (UPDATED: SigLIP + Florence-2 + OCR)
======================================
Công cụ tìm kiếm video chuyên sâu. Mặc định tự động hiển thị Grid 10 ảnh kết quả.

HƯỚNG DẪN SỬ DỤNG:

    # 1. Chế độ Mặc định (Tự động chạy và hiển thị 3 Grid: Thuần Hình / Thuần Chữ / Hỗn hợp để so sánh)
    python run_search.py --query "đoàn người đua xe đạp về đích ở Tam Kì, Quảng Nam"

    # 2. Chế độ Thuần Chữ (Chỉ dùng OCR - Rất nhanh, lý tưởng để tìm số nhà, địa danh, chữ trên áo)
    python run_search.py --query "Tam Kì" --search-mode text

    # 3. Chế độ Thuần Hình Ảnh (SigLIP + Florence - Lờ đi các chữ cái gây nhiễu, lý tưởng tìm sự kiện)
    python run_search.py --query "đoàn người đua xe đạp về đích" --search-mode visual

    # 4. Chạy theo batch từ file JSON để nộp bài (Sẽ tự động chọn hybrid, tắt popup ảnh)
    python run_search.py --query-file queries.json --output submissions/

    # 5. Tìm kiếm Q&A (Video Question Answering)
    python run_search.py --query "cảnh bữa tiệc" --question "Váy của cô gái màu gì?" --type qa
"""

import sys
import os
import re
import json
import unicodedata
import argparse
from pathlib import Path
import numpy as np
# Shim: NumPy 1.24+ đã xóa np.long/np.ulong/np.bool/..., nhưng SciPy/Milvus vẫn dùng
_NP_SHIMS = {
    'long': np.int64, 'ulong': np.uint64,
    'bool': np.bool_, 'int': np.int_, 'float': np.float64,
    'complex': np.complex128, 'object': np.object_, 'str': np.str_,
}
for _attr, _fallback in _NP_SHIMS.items():
    if not hasattr(np, _attr):
        setattr(np, _attr, _fallback)
import yaml
import torch
from PIL import Image
from pymilvus import MilvusClient
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel
from translate import analyze_query_offline_mt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.retrieval import CLIPRetriever, BM25Retriever, HybridRetriever
from src.query import QASearcher, TRAKESearcher
from src.submission import SubmissionManager
from src.scoring import evaluate_dataset, print_evaluation_report


# =======================================================================
# CẤU HÌNH ĐƯỜNG DẪN & MÔ HÌNH (Sửa tại đây)
# =======================================================================
SIGLIP_DB_PATH = "aic_kis_database_siglip.db"
SIGLIP_COLLECTION = "kis_keyframes_siglip"
OCR_DB_PATH = "ocr_database.json"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
FLORENCE_MODEL_NAME = "microsoft/Florence-2-base"
# =======================================================================


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


from src.query.florence_kis import FlorenceKISSearcher

def show_top_k_images(results, k=5, query_text="", output_path="search_results_preview.jpg"):
    """Hiển thị top-k kết quả dưới dạng lưới ảnh."""
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont
    except ImportError:
        print("[WARN] Cần cài PIL để hiển thị ảnh: pip install Pillow")
        return

    k = min(k, len(results))
    if k == 0:
        print("[WARN] Không có kết quả để hiển thị")
        return

    # Load images
    images = []
    labels = []
    for i, r in enumerate(results[:k]):
        img_path = r.get("image_path") or r.get("frame_path", "")
        if img_path and os.path.exists(img_path):
            img = PILImage.open(img_path).convert("RGB")
            images.append(img)
            
            # Check if answer exists (QA mode)
            ans = f" | Ans: {r.get('answer', '')}" if 'answer' in r else ""
            labels.append(
                f"#{i + 1} {r['video_id']} - {r['frame_id']}{ans}\n"
                f"Score: {r['score']:.4f} | SigLIP: {r.get('clip', 0):.4f} | Flo: {r.get('flo', 0):.2f} | OCR: {r.get('ocr', 0):.2f}"
            )
        else:
            # Placeholder
            img = PILImage.new("RGB", (480, 270), color=(40, 40, 40))
            images.append(img)
            labels.append(
                f"#{i + 1} {r['video_id']} - {r['frame_id']} (không tìm thấy ảnh)"
            )

    # Tạo grid
    cols = min(k, 5)
    rows = (k + cols - 1) // cols
    thumb_w, thumb_h = 480, 270
    label_h = 50
    padding = 10

    grid_w = cols * thumb_w + (cols + 1) * padding
    grid_h = rows * (thumb_h + label_h) + (rows + 1) * padding + 40  # +40 cho tiêu đề

    grid = PILImage.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(grid)

    # Tiêu đề
    title = f'Query: "{query_text}"' if query_text else "Search Results"
    draw.text((padding, 8), title, fill=(255, 255, 255))

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        x = col * thumb_w + (col + 1) * padding
        y = row * (thumb_h + label_h) + (row + 1) * padding + 40

        # Resize ảnh
        img_resized = img.resize((thumb_w, thumb_h), PILImage.LANCZOS)
        grid.paste(img_resized, (x, y))

        # Viết label
        draw.text((x + 4, y + thumb_h + 2), label, fill=(200, 255, 200))

    # Lưu và mở
    grid.save(output_path, quality=90)
    print(f"\n[Preview] Đã lưu ảnh kết quả: {output_path}")

    # Thử mở ảnh tự động
    try:
        import subprocess

        subprocess.Popen(
            ["xdg-open", output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def init_components(cfg: dict):
    backend = cfg.get("retrieval_backend", "milvus").lower()

    # --- CLIP retriever (cho QA/TRAKE) — optional, không crash nếu chưa build ---
    vector_retriever = None
    bm25_retriever = None
    hybrid = None
    qa_searcher = None
    trake_searcher = None

    try:
        if backend == "milvus":
            from src.retrieval import MilvusRetriever

            vector_retriever = MilvusRetriever(
                db_path=cfg["index"]["milvus_db_path"],
                collection_name=cfg["index"]["milvus_collection"],
                model_name=cfg["clip"]["model_name"],
                device=cfg["clip"]["device"],
            )
        else:
            from src.retrieval import CLIPRetriever

            vector_retriever = CLIPRetriever(
                index_path=cfg["index"]["faiss_index_path"],
                frame_map_path=cfg["index"]["frame_map_path"],
                model_name=cfg["clip"]["model_name"],
                device=cfg["clip"]["device"],
            )

        bm25_corpus_path = cfg["index"]["bm25_corpus_path"]
        frame_map_path = cfg["index"]["frame_map_path"]
        bm25_retriever = (
            BM25Retriever(corpus_path=bm25_corpus_path, frame_map_path=frame_map_path)
            if os.path.exists(bm25_corpus_path) and os.path.exists(frame_map_path)
            else None
        )

        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            clip_weight=cfg["retrieval"]["clip_weight"],
            bm25_weight=cfg["retrieval"]["bm25_weight"],
        )

        qa_searcher = QASearcher(
            retriever=hybrid,
            vqa_model_name=cfg["vqa"]["model"],
            device=cfg["vqa"]["device"],
            top_k_frames_for_vqa=cfg["vqa"]["top_k_frames"],
            max_answers=cfg["retrieval"]["final_top_k"],
        )

        trake_searcher = TRAKESearcher(
            clip_retriever=vector_retriever,
            top_k_per_event=cfg["trake"]["top_k_per_event"],
            max_answers=cfg["retrieval"]["final_top_k"],
        )
    except Exception as e:
        print(f"[WARN] Không thể khởi tạo CLIP/QA/TRAKE retriever: {e}")
        print("  → KIS search (SigLIP + Florence-2) vẫn hoạt động bình thường.")

    # --- KIS searcher (SigLIP + Florence-2) — luôn khởi tạo ---
    kis_searcher = FlorenceKISSearcher(
        vector_retriever=vector_retriever,
        collection_name=SIGLIP_COLLECTION,
        keyframes_dir=cfg["data"].get("keyframes_root", "DATASET"),
        ocr_db_path=OCR_DB_PATH,
        max_answers=cfg["retrieval"]["final_top_k"],
        batch_size=8,
    )

    return {
        "vector": vector_retriever,
        "clip": vector_retriever,
        "bm25": bm25_retriever,
        "hybrid": hybrid,
        "kis": kis_searcher,
        "qa": qa_searcher,
        "trake": trake_searcher,
    }


def run_single_query(components: dict, args, cfg: dict):
    qtype = args.type.lower()

    if qtype in ("kis", "textual_kis"):
        search_mode = getattr(args, "search_mode", "all")
        modes_to_run = ["visual", "text", "hybrid"] if search_mode == "all" else [search_mode]
        
        all_results = []
        show_k = getattr(args, "show_k", 10)
        show_images = getattr(args, "show_images", True)
        
        for mode in modes_to_run:
            if len(modes_to_run) > 1:
                print(f"\n=============================================")
                print(f" Đang chạy tìm kiếm chế độ: {mode.upper()}")
                print(f"=============================================")
                
            results = components["kis"].search(args.query, search_mode=mode)
            
            print(f"\n=== KẾT QUẢ TÌM KIẾM ({mode.upper()}) ===")
            for i, r in enumerate(results[:show_k]):
                print(
                    f"▶ {r['video_id']} - {r['frame_id']} | Tổng: {r['score']:.4f} (SigLIP: {r['clip']:.4f} | Florence: {r['flo']:.4f} | OCR: {r['ocr']:.4f})"
                )
            
            if show_images:
                mode_label = f"{args.query} [{mode.upper()}]" if search_mode == "all" else args.query
                out_name = f"search_results_preview_{mode}.jpg" if search_mode == "all" else "search_results_preview.jpg"
                show_top_k_images(results, k=show_k, query_text=mode_label, output_path=out_name)
                
            all_results.append(results)

        # Dùng kết quả cuối cùng (hybrid) để format (nếu cần lưu file)
        formatted = components["kis"].format_submission(all_results[-1])

    elif qtype in ("qa", "vqa"):
        results = components["qa"].search(
            query=args.query, question=args.question, use_vqa=not args.no_vqa
        )
        show_k = getattr(args, "show_k", 10)
        show_images = getattr(args, "show_images", True)

        print("\n=== KẾT QUẢ TÌM KIẾM (QA/VQA) ===")
        for i, r in enumerate(results[:show_k]):
            ans_str = f" | Answer: {r.get('answer', '')}" if 'answer' in r else ""
            print(f"▶ {r['video_id']} - {r['frame_id']} | Tổng: {r['score']:.4f}{ans_str}")

        if show_images:
            show_top_k_images(results, k=show_k, query_text=args.question or args.query, output_path="search_results_qa.jpg")

        formatted = components["qa"].format_submission(results)

    elif qtype == "trake":
        results = components["trake"].search(args.events)
        show_k = getattr(args, "show_k", 10)
        show_images = getattr(args, "show_images", True)

        print("\n=== KẾT QUẢ TÌM KIẾM (TRAKE) ===")
        for i, r in enumerate(results[:show_k]):
            fids_str = ", ".join(str(f) for f in r['frame_ids'])
            print(f"▶ {r['video_id']} - Frames: [{fids_str}] | Tổng: {r['total_score']:.4f}")

        if show_images:
            flat_results = []
            for r in results:
                # Dùng frame giữa của sequence để đại diện hiển thị
                mid_event = r['events'][len(r['events']) // 2]
                flat_results.append({
                    "video_id": r["video_id"],
                    "frame_id": mid_event["frame_id"],
                    "image_path": mid_event.get("frame_path", ""),
                    "score": r["total_score"],
                    "clip": r["total_score"],
                })
            show_top_k_images(flat_results, k=show_k, query_text=" → ".join(args.events), output_path="search_results_trake.jpg")

        formatted = components["trake"].format_submission(results)

    else:
        print(f"[ERROR] Unknown query type: {qtype}")
        return

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "result.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(formatted) + "\n")


def run_batch_queries(components: dict, args, cfg: dict):
    with open(args.query_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "queries" in data:
            queries = data["queries"]
        else:
            queries = data

    manager = SubmissionManager(
        output_dir=args.output or cfg["submission"]["output_dir"],
        max_answers=cfg["retrieval"]["final_top_k"],
    )

    all_submissions, all_query_results = [], []

    for i, q in enumerate(queries):
        qid = q.get("query_id", f"q{i + 1}")
        qtype = q.get("query_type", "textual_kis").lower()
        print(f"\n[Query {i + 1}/{len(queries)}] ID={qid} Type={qtype}")

        try:
            if qtype in ("textual_kis", "kis"):
                search_mode = getattr(args, "search_mode", "hybrid")
                if search_mode == "all":
                    search_mode = "hybrid"  # Trong chế độ batch, mặc định dùng hybrid
                results = components["kis"].search(q.get("query_text", ""), search_mode=search_mode)
            elif qtype in ("qa", "vqa"):
                results = components["qa"].search(
                    query=q.get("retrieval_query", q.get("query_text", "")),
                    question=q.get("question", ""),
                    use_vqa=not args.no_vqa,
                )
            elif qtype == "trake":
                results = components["trake"].search(q.get("events", []))
            else:
                results = []

            all_submissions.append(
                manager.build_query_submission(
                    {"query_id": qid, "query_type": qtype}, results
                )
            )

            if args.evaluate and "ground_truth" in q:
                all_query_results.append(
                    {
                        "query_id": qid,
                        "query_type": qtype,
                        "answers": results,
                        "ground_truth": q["ground_truth"],
                    }
                )
        except Exception as e:
            all_submissions.append(
                {"query_id": qid, "query_type": qtype, "answers": [], "error": str(e)}
            )

    manager.save_all(all_submissions)

    if args.evaluate and all_query_results:
        eval_result = evaluate_dataset(all_query_results)
        print_evaluation_report(eval_result)
        report_path = (
            Path(args.output or cfg["submission"]["output_dir"]) / "eval_report.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in eval_result.items() if k != "per_query"}, f, indent=2
            )


def main():
    parser = argparse.ArgumentParser(description="AIC2026 Baseline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", type=str, help="Query text (single query)")
    group.add_argument("--query-file", type=str, help="JSON file chứa nhiều queries")

    parser.add_argument(
        "--type", default="kis", choices=["kis", "textual_kis", "qa", "vqa", "trake"]
    )
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--events", nargs="+")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--no-vqa", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--search-mode",
        choices=["all", "hybrid", "visual", "text"],
        default="all",
        help="Chế độ tìm kiếm: all (chạy cả 3), hybrid (kết hợp), visual (chỉ SigLIP), text (chỉ OCR)",
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        default=True,
        help="Hiển thị top-k ảnh kết quả sau khi search (Mặc định: True)",
    )
    parser.add_argument(
        "--show-k", type=int, default=10, help="Số ảnh hiển thị (mặc định 10)"
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    components = init_components(cfg)

    if args.query_file:
        run_batch_queries(components, args, cfg)
    elif args.query or args.events:
        run_single_query(components, args, cfg)
    else:
        while True:
            try:
                query = input("\nQuery> ").strip()
                if not query:
                    continue
                args.query = query
                if args.type in ("qa", "vqa") and not args.question:
                    args.question = input("Question> ").strip()
                run_single_query(components, args, cfg)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
