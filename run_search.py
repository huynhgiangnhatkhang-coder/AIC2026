"""
AIC 2026 Baseline — CLI Entry Point (UPDATED: SigLIP + Florence-2 + OCR)
======================================
Chạy search từ command line.

Usage:
  # Chạy single query (Bây giờ sẽ tự động dùng SigLIP + Florence-2 + OCR)
  python run_search.py --query "một người mở laptop" --type kis

  # Chạy từ file JSON
  python run_search.py --query-file queries.json --output submissions/

  # Q&A
  python run_search.py --query "cảnh bữa tiệc" --question "Váy của cô gái màu gì?" --type qa

  # TRAKE
  python run_search.py --events "giậm nhảy" "bay qua xà" "tiếp đất" --type trake
"""
import sys
import os
import json
import argparse
from pathlib import Path
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


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =======================================================================
# LỚP TÌM KIẾM MỚI (Tích hợp SigLIP + Florence-2 + OCR thay cho DINO)
# =======================================================================
class FlorenceKISSearcher:
    def __init__(self, db_path, collection_name, keyframes_dir, ocr_db_path, max_answers=100, batch_size=8):
        print("\n[Init] Khởi tạo hệ thống FlorenceKISSearcher (SigLIP + Florence-2 + OCR)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.keyframes_dir = keyframes_dir
        self.max_answers = max_answers
        self.batch_size = batch_size
        self.collection_name = collection_name

        # 1. Tải OCR Data
        self.ocr_data = {}
        if os.path.exists(ocr_db_path):
            with open(ocr_db_path, "r", encoding="utf-8") as f:
                self.ocr_data = json.load(f)
            print(f"  -> Đã tải {len(self.ocr_data)} bản ghi OCR!")
        else:
            print(f"  -> [CẢNH BÁO] Không tìm thấy {ocr_db_path}, tính năng OCR sẽ bị bỏ qua (0 điểm).")

        # 2. Kết nối Milvus cho SigLIP
        print(f"  -> Kết nối Milvus (SigLIP): {db_path} | {collection_name}")
        self.milvus_client = MilvusClient(db_path)
        self.milvus_client.load_collection(collection_name)

        # 3. Khởi tạo Model SigLIP
        print("  -> Đang tải model SigLIP...")
        self.siglip_model_name = "google/siglip-base-patch16-224"
        self.siglip_processor = AutoProcessor.from_pretrained(self.siglip_model_name)
        self.siglip_model = AutoModel.from_pretrained(self.siglip_model_name).to(self.device).eval()

        # 4. Khởi tạo Model Florence-2
        print("  -> Đang tải model Florence-2...")
        self.florence_model_name = "microsoft/Florence-2-base"
        self.florence_processor = AutoProcessor.from_pretrained(self.florence_model_name, trust_remote_code=True)
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            self.florence_model_name, 
            trust_remote_code=True,
            attn_implementation="sdpa"
        ).to(self.device).eval()

    def get_florence_scores_batch(self, image_paths, required_objects):
        if not required_objects or not image_paths:
            return [1.0] * len(image_paths) if not required_objects else [0.0] * len(image_paths)
            
        text_input = " and ".join(required_objects)
        prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {text_input}"
        all_scores = []
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="[Florence-2] Re-ranking", unit="batch", leave=False):
            batch_paths = image_paths[i:i+self.batch_size]
            images = []
            valid_indices = []
            
            for idx, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    valid_indices.append(idx)
                except Exception:
                    pass
                    
            if not images:
                all_scores.extend([0.0] * len(batch_paths))
                continue
                
            prompts = [prompt] * len(images)
            try:
                inputs = self.florence_processor(text=prompts, images=images, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    generated_ids = self.florence_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        num_beams=3
                    )
                generated_texts = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)
                batch_scores = [0.0] * len(batch_paths)
                
                for j, (gen_text, img) in enumerate(zip(generated_texts, images)):
                    parsed_answer = self.florence_processor.post_process_generation(
                        gen_text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(img.width, img.height)
                    )
                    results = parsed_answer.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
                    labels_found = results.get('labels', [])
                    if labels_found:
                        unique_labels_found = set(labels_found)
                        score = len(unique_labels_found) / len(required_objects)
                        batch_scores[valid_indices[j]] = min(score, 1.0)
                        
                all_scores.extend(batch_scores)
            except Exception as e:
                all_scores.extend([0.0] * len(batch_paths))
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return all_scores

    def get_ocr_score(self, image_rel_path, search_text):
        if not search_text or not self.ocr_data:
            return 0.0
        image_rel_path = image_rel_path.replace("\\", "/")
        ocr_text = ""
        for key, val in self.ocr_data.items():
            if image_rel_path.endswith(key.replace("\\", "/")):
                ocr_text = val
                break
        if not ocr_text:
            return 0.0
        search_keywords = search_text.lower().split()
        if not search_keywords:
            return 0.0
        matched = sum(1 for kw in search_keywords if kw in ocr_text)
        return matched / len(search_keywords)

    def search(self, raw_query):
        # 1. Dịch & Phân tích truy vấn
        parsed_query_data = analyze_query_offline_mt(raw_query)
        clip_query = parsed_query_data["clip_query"]
        required_objects = parsed_query_data["required_objects"]

        # 2. Mã hóa SigLIP
        text_inputs = self.siglip_processor(text=[clip_query], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.siglip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            text_vector = text_features[0].cpu().tolist()

        # 3. Truy xuất thô bằng Milvus (Top 50)
        search_results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[text_vector],
            limit=50, 
            output_fields=["video_id", "frame_id"],
            search_params={"metric_type": "IP"}
        )

        valid_hits = []
        valid_image_paths = []
        for hit in search_results[0]:
            v_id = hit["entity"]["video_id"]
            f_id = hit["entity"]["frame_id"]
            video_folder = v_id.replace(".mp4", "")
            batch_prefix = video_folder.split("_")[0]  
            batch_folder_name = f"Keyframes_{batch_prefix}" 
            
            possible_formats = [
                f"{f_id}.jpg", f"{f_id:03d}.jpg", f"{f_id:04d}.jpg", 
                f"{f_id:05d}.jpg", f"{f_id:06d}.jpg", f"{f_id}.png", f"{f_id:04d}.png" 
            ]
            image_path = None
            for fmt in possible_formats:
                temp_path = os.path.join(self.keyframes_dir, batch_folder_name, "keyframes", video_folder, fmt)
                if os.path.exists(temp_path):
                    image_path = temp_path
                    break
            
            if image_path:
                valid_hits.append(hit)
                valid_image_paths.append(image_path)

        # 4. Re-rank đồng loạt bằng Florence-2
        florence_scores = self.get_florence_scores_batch(valid_image_paths, required_objects)

        # 5. Tổng hợp điểm & Sắp xếp
        scored_results = []
        for hit, image_path, florence_score in zip(valid_hits, valid_image_paths, florence_scores):
            clip_score = hit["distance"]
            rel_image_path = os.path.relpath(image_path, self.keyframes_dir)
            ocr_score = self.get_ocr_score(rel_image_path, clip_query)

            final_score = (0.6 * clip_score) + (0.3 * florence_score) + (0.1 * ocr_score)
            if ocr_score > 0.5:
                final_score += 0.2
                
            scored_results.append({
                "video_id": hit["entity"]["video_id"],
                "frame_id": hit["entity"]["frame_id"],
                "score": final_score,
                "clip": clip_score, "flo": florence_score, "ocr": ocr_score
            })

        scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
        return scored_results[:self.max_answers]

    def format_submission(self, results):
        formatted = []
        for r in results:
            vid = r["video_id"].replace(".mp4", "")
            fid = r["frame_id"]
            formatted.append(f"{vid},{fid}")
        return formatted


def init_components(cfg: dict):
    backend = cfg.get("retrieval_backend", "milvus").lower()

    # GIỮ NGUYÊN CLIP CŨ ĐỂ CHẠY CÁC TÍNH NĂNG KHÁC (QA, TRAKE)
    if backend == "milvus":
        from src.retrieval import MilvusRetriever
        vector_retriever = MilvusRetriever(
            db_path=cfg["index"]["milvus_db_path"],
            collection_name=cfg["index"]["milvus_collection"],
            model_name=cfg["clip"]["model_name"],
            device=cfg["clip"]["device"]
        )
    else:
        from src.retrieval import CLIPRetriever
        vector_retriever = CLIPRetriever(
            index_path=cfg["index"]["faiss_index_path"],
            frame_map_path=cfg["index"]["frame_map_path"],
            model_name=cfg["clip"]["model_name"],
            device=cfg["clip"]["device"]
        )

    bm25_corpus_path = cfg["index"]["bm25_corpus_path"]
    frame_map_path   = cfg["index"]["frame_map_path"]

    if os.path.exists(bm25_corpus_path) and os.path.exists(frame_map_path):
        bm25_retriever = BM25Retriever(corpus_path=bm25_corpus_path, frame_map_path=frame_map_path)
    else:
        bm25_retriever = None

    hybrid = HybridRetriever(
        vector_retriever=vector_retriever, bm25_retriever=bm25_retriever,
        clip_weight=cfg["retrieval"]["clip_weight"], bm25_weight=cfg["retrieval"]["bm25_weight"]
    )

    # -------------------------------------------------------------
    # SỬ DỤNG MÔ HÌNH MỚI (FLORENCE KIS SEARCHER) CHO TEXTUAL KIS
    # -------------------------------------------------------------
    kis_searcher = FlorenceKISSearcher(
        db_path="aic_kis_database_siglip.db",
        collection_name="kis_keyframes_siglip",
        keyframes_dir=cfg["data"].get("keyframes_root", "DATASET"),
        ocr_db_path="ocr_database.json",
        max_answers=cfg["retrieval"]["final_top_k"],
        batch_size=8
    )
    
    # -------------------------------------------------------------
    qa_searcher = QASearcher(
        retriever=hybrid, vqa_model_name=cfg["vqa"]["model"], device=cfg["vqa"]["device"],
        top_k_frames_for_vqa=cfg["vqa"]["top_k_frames"], max_answers=cfg["retrieval"]["final_top_k"]
    )
    trake_searcher = TRAKESearcher(
        clip_retriever=vector_retriever, top_k_per_event=cfg["trake"]["top_k_per_event"],
        max_answers=cfg["retrieval"]["final_top_k"]
    )

    return {
        "vector": vector_retriever,
        "clip":   vector_retriever,
        "bm25":   bm25_retriever,
        "hybrid": hybrid,
        "kis":    kis_searcher, # Đã được cập nhật thành mô hình mới!
        "qa":     qa_searcher,
        "trake":  trake_searcher
    }


def run_single_query(components: dict, args, cfg: dict):
    qtype = args.type.lower()
    print(f"\n[Search] Type: {qtype}")

    if qtype in ("kis", "textual_kis"):
        print(f"  Query: '{args.query}'")
        results = components["kis"].search(args.query)
        
        # In chi tiết điểm ra console giống file code mẫu của bạn
        print("\n=== KẾT QUẢ TÌM KIẾM CHÍNH THỨC ===")
        for i, r in enumerate(results[:5]):
            print(f"▶ {r['video_id']} - {r['frame_id']} | Tổng điểm: {r['score']:.4f} (SigLIP: {r['clip']:.4f} | Florence: {r['flo']:.4f} | OCR: {r['ocr']:.4f})")
            
        formatted = components["kis"].format_submission(results)

    elif qtype in ("qa", "vqa"):
        results = components["qa"].search(query=args.query, question=args.question, use_vqa=not args.no_vqa)
        formatted = components["qa"].format_submission(results)

    elif qtype == "trake":
        results = components["trake"].search(args.events)
        formatted = components["trake"].format_submission(results)

    else:
        print(f"[ERROR] Unknown query type: {qtype}")
        return

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
    print(f"[Batch] Loading queries from: {args.query_file}")
    with open(args.query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    manager = SubmissionManager(
        output_dir=args.output or cfg["submission"]["output_dir"],
        max_answers=cfg["retrieval"]["final_top_k"]
    )

    all_submissions = []
    all_query_results = [] 

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

            sub = manager.build_query_submission({"query_id": qid, "query_type": qtype}, results)
            all_submissions.append(sub)

            if args.evaluate and "ground_truth" in q:
                all_query_results.append({
                    "query_id": qid, "query_type": qtype,
                    "answers": results, "ground_truth": q["ground_truth"]
                })

        except Exception as e:
            print(f"  [ERROR] {e}")
            all_submissions.append({"query_id": qid, "query_type": qtype, "answers": [], "error": str(e)})

    manager.save_all(all_submissions)

    if args.evaluate and all_query_results:
        print("\n[Evaluate] Computing scores...")
        eval_result = evaluate_dataset(all_query_results)
        print_evaluation_report(eval_result)
        report_path = Path(args.output or cfg["submission"]["output_dir"]) / "eval_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            summary = {k: v for k, v in eval_result.items() if k != "per_query"}
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="AIC2026 Baseline — Search CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", type=str, help="Query text (single query)")
    group.add_argument("--query-file", type=str, help="JSON file chứa nhiều queries")

    parser.add_argument("--type", default="kis", choices=["kis", "textual_kis", "qa", "vqa", "trake"])
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--events", nargs="+")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--no-vqa", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--top-k", type=int, default=100)

    args = parser.parse_args()
    cfg = load_config(args.config)

    components = init_components(cfg)
    print("\n[Ready] Tất cả components đã load xong!\n")

    if args.query_file:
        run_batch_queries(components, args, cfg)
    elif args.query:
        run_single_query(components, args, cfg)
    else:
        print("[Interactive Mode] Nhập query (Ctrl+C để thoát)")
        while True:
            try:
                query = input("\nQuery> ").strip()
                if not query: continue
                args.query = query
                if args.type in ("qa", "vqa") and not args.question:
                    args.question = input("Question> ").strip()
                run_single_query(components, args, cfg)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
