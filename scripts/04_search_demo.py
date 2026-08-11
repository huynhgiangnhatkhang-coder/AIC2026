"""
AIC 2026 Baseline — Script 04: Search Demo
============================================
Script demo tìm kiếm — tích hợp trực tiếp từ code search gốc.

Tương đương với code gốc của user nhưng:
  - Dùng MilvusRetriever từ pipeline (không reload model mỗi lần)
  - Đọc cấu hình từ config.yaml (không hardcode path)
  - Hỗ trợ cả 3 loại query (KIS / Q&A / TRAKE)
  - Hỗ trợ interactive mode (nhập query liên tục)

Usage:
  # Search nhanh (giống code gốc)
  python scripts/04_search_demo.py --query "Night scene with trucks and police"

  # Chỉ định số kết quả
  python scripts/04_search_demo.py --query "..." --top-k 10

  # Q&A
  python scripts/04_search_demo.py \\
      --query "cảnh bữa tiệc" \\
      --question "Người phụ nữ cầm gì?" \\
      --type qa

  # TRAKE
  python scripts/04_search_demo.py \\
      --type trake \\
      --events "vận động viên giậm nhảy" "bay qua xà" "tiếp đất"

  # Interactive mode (nhập query liên tục)
  python scripts/04_search_demo.py --interactive
"""
import sys
import os
import argparse

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
from src.retrieval.milvus_retriever import MilvusRetriever


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def search_kis_simple(client: MilvusRetriever, query_text: str, top_k: int = 5):
    """
    Hàm search KIS đơn giản — giống hệt hàm search_kis trong code gốc của user.

    Args:
        client:     MilvusRetriever đã kết nối
        query_text: câu truy vấn văn bản
        top_k:      số kết quả trả về
    """
    print(f"\nĐang xử lý truy vấn: '{query_text}'")
    print("Đang tìm kiếm trong Database...")

    results = client.search(query_text, top_k=top_k)

    print("\n=== KẾT QUẢ TÌM KIẾM ===")
    if not results:
        print("  Không tìm thấy kết quả.")
        return

    for hit in results:
        v_id   = hit["video_id"]
        f_id   = hit["frame_id"]
        score  = hit["score"]
        print(f"  video_id = {v_id}, frame_id = {f_id} "
              f"(Độ tương đồng: {score:.4f})")

    return results


def search_qa_demo(cfg: dict, retriever: MilvusRetriever,
                   query: str, question: str, top_k: int = 5):
    """Demo Q&A search sử dụng HybridRetriever + VQA."""
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.query.qa_vqa import QASearcher

    bm25 = BM25Retriever(
        corpus_path=cfg["index"]["bm25_corpus_path"],
        frame_map_path=cfg["index"]["frame_map_path"]
    )
    hybrid = HybridRetriever(
        vector_retriever=retriever,
        bm25_retriever=bm25,
        clip_weight=cfg["retrieval"]["clip_weight"],
        bm25_weight=cfg["retrieval"]["bm25_weight"]
    )
    searcher = QASearcher(
        retriever=hybrid,
        vqa_model_name=cfg["vqa"]["model"],
        device=cfg["vqa"]["device"],
        top_k_frames_for_vqa=cfg["vqa"]["top_k_frames"],
        max_answers=top_k
    )

    print(f"\n[Q&A] Retrieval: '{query}'")
    print(f"[Q&A] Question:  '{question}'")
    results = searcher.search(query=query, question=question, use_vqa=True)

    print("\n=== KẾT QUẢ Q&A ===")
    for r in results[:top_k]:
        print(f"  video_id={r['video_id']}, frame_id={r['frame_id']}, "
              f"answer='{r.get('answer','')}' (score={r['score']:.4f})")
    return results


def search_trake_demo(retriever: MilvusRetriever, events: list, top_k: int = 5):
    """Demo TRAKE temporal search."""
    from src.query.trake import TRAKESearcher

    searcher = TRAKESearcher(
        clip_retriever=retriever,
        top_k_per_event=300,
        max_answers=top_k
    )

    print(f"\n[TRAKE] {len(events)} sự kiện:")
    for i, e in enumerate(events, 1):
        print(f"  Event {i}: '{e}'")

    results = searcher.search(events)

    print("\n=== KẾT QUẢ TRAKE ===")
    for r in results[:top_k]:
        frame_ids = ", ".join(str(f) for f in r["frame_ids"])
        print(f"  video_id={r['video_id']} | frames=[{frame_ids}] "
              f"(total_score={r['total_score']:.4f})")
    return results


def interactive_mode(retriever: MilvusRetriever, top_k: int = 5):
    """
    Mode tương tác: nhập query liên tục từ terminal.
    Tương tự như chỉnh query_text trong code gốc nhưng không cần restart.
    """
    print("\n" + "="*60)
    print("  AIC2026 SEARCH — Interactive Mode")
    print("  Nhập query để tìm kiếm. Gõ 'quit' để thoát.")
    print("="*60)

    while True:
        try:
            query = input("\nQuery> ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("[Exit]")
                break

            search_kis_simple(retriever, query, top_k=top_k)

        except KeyboardInterrupt:
            print("\n[Exit]")
            break


def main():
    parser = argparse.ArgumentParser(
        description="AIC2026 Search Demo — tìm kiếm KIS / Q&A / TRAKE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Câu truy vấn văn bản")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Số kết quả hiển thị (mặc định: 5)")
    parser.add_argument("--type", "-t", default="kis",
                        choices=["kis", "qa", "trake"],
                        help="Loại truy vấn (mặc định: kis)")
    parser.add_argument("--question", type=str, default="",
                        help="Câu hỏi VQA (chỉ dùng với --type qa)")
    parser.add_argument("--events", nargs="+",
                        help="Danh sách sự kiện (chỉ dùng với --type trake)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Bật interactive mode (nhập query liên tục)")
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        # Thử tìm config ở thư mục cha (khi chạy từ scripts/)
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

    cfg = load_config(config_path)

    # Kết nối Milvus (giống code gốc)
    db_path = cfg["index"]["milvus_db_path"]
    collection_name = cfg["index"]["milvus_collection"]

    print("Đang kết nối vào Database có sẵn...")
    try:
        retriever = MilvusRetriever(
            db_path=db_path,
            collection_name=collection_name,
            model_name=cfg["clip"]["model_name"],
            device=cfg["clip"]["device"]
        )
    except FileNotFoundError as e:
        print(f"\nLỗi kết nối Database! Vui lòng chạy build script trước.")
        print(f"Chi tiết lỗi: {e}")
        print("\n→ Chạy:")
        print("  python scripts/02b_build_milvus_db.py --config config.yaml")
        sys.exit(1)

    # Chạy theo mode
    if args.interactive:
        interactive_mode(retriever, top_k=args.top_k)

    elif args.type == "kis":
        query = args.query or input("Query> ").strip()
        search_kis_simple(retriever, query, top_k=args.top_k)

    elif args.type == "qa":
        query = args.query or input("Retrieval query> ").strip()
        question = args.question or input("Question> ").strip()
        search_qa_demo(cfg, retriever, query, question, top_k=args.top_k)

    elif args.type == "trake":
        events = args.events
        if not events:
            print("Nhập các sự kiện (Enter trống để kết thúc):")
            events = []
            i = 1
            while True:
                e = input(f"  Event {i}> ").strip()
                if not e:
                    break
                events.append(e)
                i += 1
        search_trake_demo(retriever, events, top_k=args.top_k)

    retriever.close()


# ==========================================
# ENTRY POINT — giống code gốc của user
# ==========================================
if __name__ == "__main__":
    # Nếu muốn chạy nhanh như code gốc, bỏ comment và chỉnh query_text:
    #
    # import yaml, sys, os
    # sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    # cfg = yaml.safe_load(open("config.yaml"))
    # retriever = MilvusRetriever(
    #     db_path=cfg["index"]["milvus_db_path"],
    #     collection_name=cfg["index"]["milvus_collection"]
    # )
    # query_text = "Night scene outdoors on a road with cargo container trucks, ..."
    # search_kis_simple(retriever, query_text, top_k=5)
    #
    # Hoặc dùng CLI:
    main()
