"""
AIC 2026 Baseline — CLI Entry Point
======================================
Chạy search từ command line.

Usage:
  # Chạy single query
  python run_search.py --query "một người mở laptop" --type kis

  # Chạy từ file JSON
  python run_search.py --query-file queries.json --output submissions/

  # Q&A
  python run_search.py --query "cảnh bữa tiệc" --question "Váy của cô gái màu gì?" --type qa

  # TRAKE
  python run_search.py --events "giậm nhảy" "bay qua xà" "tiếp đất" --type trake

  # Evaluate với ground truth
  python run_search.py --query-file queries.json --gt-file gt.json --evaluate
"""
import sys
import os
import json
import argparse
from pathlib import Path
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.retrieval import CLIPRetriever, BM25Retriever, HybridRetriever
from src.query import TextualKISSearcher, QASearcher, TRAKESearcher
from src.submission import SubmissionManager
from src.scoring import evaluate_dataset, print_evaluation_report


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_components(cfg: dict):
    """
    Khởi tạo tất cả components từ config.
    
    Tự động chọn backend dựa trên cfg["retrieval_backend"]:
      - "milvus" → MilvusRetriever (Milvus Lite, khuyến nghị)
      - "faiss"  → CLIPRetriever   (FAISS index, fallback)
    """
    backend = cfg.get("retrieval_backend", "milvus").lower()
    print(f"[Init] Retrieval backend: {backend.upper()}")

    if backend == "milvus":
        from src.retrieval import MilvusRetriever
        print("[Init] Loading Milvus Retriever...")
        vector_retriever = MilvusRetriever(
            db_path=cfg["index"]["milvus_db_path"],
            collection_name=cfg["index"]["milvus_collection"],
            model_name=cfg["clip"]["model_name"],
            device=cfg["clip"]["device"]
        )
    else:
        from src.retrieval import CLIPRetriever
        print("[Init] Loading FAISS/CLIP Retriever...")
        vector_retriever = CLIPRetriever(
            index_path=cfg["index"]["faiss_index_path"],
            frame_map_path=cfg["index"]["frame_map_path"],
            model_name=cfg["clip"]["model_name"],
            device=cfg["clip"]["device"]
        )

    bm25_corpus_path = cfg["index"]["bm25_corpus_path"]
    frame_map_path   = cfg["index"]["frame_map_path"]

    if os.path.exists(bm25_corpus_path) and os.path.exists(frame_map_path):
        print("[Init] Loading BM25 Retriever...")
        bm25_retriever = BM25Retriever(
            corpus_path=bm25_corpus_path,
            frame_map_path=frame_map_path
        )
    else:
        print("[Init] BM25 corpus not found — skipping BM25 (CLIP-only mode)")
        print(f"       Missing: {bm25_corpus_path}")
        print(f"       Run: python scripts/03_build_bm25_index.py --config config.yaml")
        bm25_retriever = None

    print("[Init] Building Hybrid Retriever...")
    hybrid = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        clip_weight=cfg["retrieval"]["clip_weight"],
        bm25_weight=cfg["retrieval"]["bm25_weight"]
    )

    objects_dir = str(Path(cfg["data"]["root"]) / cfg["data"]["objects_dir"])

    kis_searcher = TextualKISSearcher(
        retriever=vector_retriever,          # Dùng trực tiếp MilvusRetriever (không qua BM25)
        keyframes_dir=cfg["data"]["keyframes_root"],
        max_answers=cfg["retrieval"]["final_top_k"],
        top_k_coarse=cfg["dino"]["top_k_coarse"],
        alpha=cfg["dino"]["alpha_clip"],
        dino_threshold=cfg["dino"]["threshold"],
        dino_model_name=cfg["dino"]["model_name"],
        device=cfg["clip"]["device"]
    )
    qa_searcher = QASearcher(
        retriever=hybrid,
        vqa_model_name=cfg["vqa"]["model"],
        device=cfg["vqa"]["device"],
        top_k_frames_for_vqa=cfg["vqa"]["top_k_frames"],
        max_answers=cfg["retrieval"]["final_top_k"]
    )
    trake_searcher = TRAKESearcher(
        clip_retriever=vector_retriever,    # TRAKE dùng trực tiếp vector retriever
        top_k_per_event=cfg["trake"]["top_k_per_event"],
        max_answers=cfg["retrieval"]["final_top_k"]
    )

    return {
        "vector": vector_retriever,
        "clip":   vector_retriever,         # backward compat alias
        "bm25":   bm25_retriever,
        "hybrid": hybrid,
        "kis":    kis_searcher,
        "qa":     qa_searcher,
        "trake":  trake_searcher
    }


def run_single_query(components: dict, args, cfg: dict):
    """Chạy 1 query đơn từ command line args."""
    qtype = args.type.lower()

    print(f"\n[Search] Type: {qtype}")

    if qtype in ("kis", "textual_kis"):
        print(f"  Query: '{args.query}'")
        results = components["kis"].search(args.query)
        formatted = components["kis"].format_submission(results)

    elif qtype in ("qa", "vqa"):
        print(f"  Retrieval query: '{args.query}'")
        print(f"  Question: '{args.question}'")
        results = components["qa"].search(
            query=args.query,
            question=args.question,
            use_vqa=not args.no_vqa
        )
        formatted = components["qa"].format_submission(results)

    elif qtype == "trake":
        events = args.events
        print(f"  Events ({len(events)}): {events}")
        results = components["trake"].search(events)
        formatted = components["trake"].format_submission(results)

    else:
        print(f"[ERROR] Unknown query type: {qtype}")
        return

    print(f"\n[Results] Top-10 answers:")
    for i, line in enumerate(formatted[:10], 1):
        print(f"  {i:3d}. {line}")

    if len(formatted) > 10:
        print(f"  ... ({len(formatted)} total)")

    # Lưu kết quả
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "result.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for line in formatted:
                f.write(line + "\n")
        print(f"\n[Saved] {out_file}")


def run_batch_queries(components: dict, args, cfg: dict):
    """Chạy batch queries từ file JSON."""
    print(f"[Batch] Loading queries from: {args.query_file}")
    with open(args.query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"[Batch] {len(queries)} queries")

    manager = SubmissionManager(
        output_dir=args.output or cfg["submission"]["output_dir"],
        max_answers=cfg["retrieval"]["final_top_k"]
    )

    all_submissions = []
    all_query_results = []  # để evaluate

    for i, q in enumerate(queries):
        qid = q.get("query_id", f"q{i+1}")
        qtype = q.get("query_type", "textual_kis").lower()
        print(f"\n[Query {i+1}/{len(queries)}] ID={qid} Type={qtype}")

        try:
            if qtype in ("textual_kis", "kis"):
                results = components["kis"].search(q.get("query_text", ""))
            elif qtype in ("qa", "vqa"):
                results = components["qa"].search(
                    query=q.get("retrieval_query", q.get("query_text", "")),
                    question=q.get("question", ""),
                    use_vqa=not args.no_vqa
                )
            elif qtype == "trake":
                results = components["trake"].search(q.get("events", []))
            else:
                results = []

            sub = manager.build_query_submission(
                {"query_id": qid, "query_type": qtype},
                results
            )
            all_submissions.append(sub)

            # Chuẩn bị cho evaluation nếu có GT
            if args.evaluate and "ground_truth" in q:
                all_query_results.append({
                    "query_id": qid,
                    "query_type": qtype,
                    "answers": results,
                    "ground_truth": q["ground_truth"]
                })

            print(f"  → {len(sub['answers'])} answers")

        except Exception as e:
            print(f"  [ERROR] {e}")
            all_submissions.append({
                "query_id": qid,
                "query_type": qtype,
                "answers": [],
                "error": str(e)
            })

    # Lưu submission
    manager.save_all(all_submissions)

    # Evaluate nếu có GT
    if args.evaluate and all_query_results:
        print("\n[Evaluate] Computing scores...")
        eval_result = evaluate_dataset(all_query_results)
        print_evaluation_report(eval_result)

        # Lưu eval report
        report_path = Path(args.output or cfg["submission"]["output_dir"]) / "eval_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            # Chỉ lưu summary (không lưu per-query r_scores chi tiết)
            summary = {k: v for k, v in eval_result.items() if k != "per_query"}
            json.dump(summary, f, indent=2)
        print(f"[Evaluate] Report saved -> {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AIC2026 Baseline — Search CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    # Query input
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", type=str, help="Query text (single query)")
    group.add_argument("--query-file", type=str, help="JSON file chứa nhiều queries")

    # Query type
    parser.add_argument("--type", default="kis",
                        choices=["kis", "textual_kis", "qa", "vqa", "trake"],
                        help="Loại truy vấn")
    parser.add_argument("--question", type=str, default="",
                        help="Câu hỏi VQA (chỉ dùng với --type qa)")
    parser.add_argument("--events", nargs="+",
                        help="Danh sách sự kiện TRAKE (chỉ dùng với --type trake)")

    # Options
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Thư mục lưu kết quả")
    parser.add_argument("--no-vqa", action="store_true",
                        help="Bỏ qua VQA model (chỉ retrieval)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Tính điểm nếu query-file có ground_truth")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Số câu trả lời tối đa")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Kiểm tra indexes đã tồn tại chưa
    backend = cfg.get("retrieval_backend", "milvus").lower()
    if backend == "milvus":
        index_path = cfg["index"]["milvus_db_path"]
        build_cmd = "python scripts/02b_build_milvus_db.py --config config.yaml"
    else:
        index_path = cfg["index"]["faiss_index_path"]
        build_cmd = "python scripts/02_build_clip_index.py --config config.yaml"

    if not os.path.exists(index_path):
        print(f"[ERROR] Index chưa tồn tại: {index_path}")
        print(f"  → Chạy theo thứ tự:")
        print(f"    python scripts/01_extract_keyframes.py --config config.yaml")
        print(f"    {build_cmd}")
        print(f"    python scripts/03_build_bm25_index.py --config config.yaml")
        sys.exit(1)

    # Init components
    components = init_components(cfg)
    print("\n[Ready] Tất cả components đã load xong!\n")

    # Run
    if args.query_file:
        run_batch_queries(components, args, cfg)
    elif args.query:
        run_single_query(components, args, cfg)
    else:
        # Interactive mode
        print("[Interactive Mode] Nhập query (Ctrl+C để thoát)")
        print(f"  Type: {args.type}")
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
                print("\n[Exit]")
                break


if __name__ == "__main__":
    main()
