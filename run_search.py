"""
AIC 2026 Baseline — CLI Entry Point (UPDATED: SigLIP + Florence-2 + OCR)
======================================
Chạy search từ command line.

Usage:
    # Chạy single query (Bây giờ sẽ tự động dùng SigLIP + Florence-2 + OCR)
    python run_search.py --query "cặp khỉ sinh đôi, khỉ, sinh đôi" --type kis --show-images --show-k 10

    # Chạy từ file JSON
    # python run_search.py --query-file queries.json --output submissions/

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


class FlorenceKISSearcher:
    def __init__(self, db_path, collection_name, keyframes_dir, ocr_db_path, max_answers=100, batch_size=8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.keyframes_dir = keyframes_dir
        self.max_answers = max_answers
        self.batch_size = batch_size
        self.collection_name = collection_name

        self.ocr_data = {}
        if os.path.exists(ocr_db_path):
            with open(ocr_db_path, "r", encoding="utf-8") as f:
                self.ocr_data = json.load(f)

        self.milvus_client = MilvusClient(db_path)
        self.milvus_client.load_collection(collection_name)

        self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        self.siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(self.device).eval()

        self.florence_processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_NAME, trust_remote_code=True)
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_NAME, 
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
                        unique_labels_found = set([lbl.lower().strip() for lbl in labels_found])
                        
                        matched_count = 0
                        for req_obj in required_objects:
                            req_obj_lower = req_obj.lower().strip()
                            if any(req_obj_lower in lbl or lbl in req_obj_lower for lbl in unique_labels_found):
                                matched_count += 1
                                
                        score = matched_count / max(1, len(required_objects))
                        batch_scores[valid_indices[j]] = min(score, 1.0)
                        
                all_scores.extend(batch_scores)
            except Exception:
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
            
        import unicodedata
        # Bỏ dấu tiếng Việt để match tốt hơn vì ocr_text đôi khi không dấu hoặc bị nhiễu
        search_text_no_accents = ''.join(c for c in unicodedata.normalize('NFD', search_text.lower()) if unicodedata.category(c) != 'Mn')
        search_text_no_accents = search_text_no_accents.replace("đ", "d")
        
        search_keywords = search_text_no_accents.split()
        if not search_keywords:
            return 0.0
        matched = sum(1 for kw in search_keywords if kw in ocr_text)
        return matched / len(search_keywords)

    def _ocr_search(self, raw_query, existing_keys):
        """Quét toàn bộ OCR database tìm frame có chữ khớp với query (độc lập với SigLIP)."""
        import unicodedata
        if not self.ocr_data or not raw_query:
            return []

        search_text = ''.join(c for c in unicodedata.normalize('NFD', raw_query.lower()) if unicodedata.category(c) != 'Mn')
        search_text = search_text.replace("đ", "d")
        keywords = search_text.split()
        if not keywords:
            return []

        ocr_hits = []
        for key, ocr_text in self.ocr_data.items():
            # Bỏ qua nếu đã có trong danh sách SigLIP
            if key in existing_keys:
                continue
            matched = sum(1 for kw in keywords if kw in ocr_text.lower())
            ocr_score = matched / len(keywords)
            if ocr_score >= 0.3:  # Tối thiểu 30% từ khóa khớp
                # Parse video_id và frame_id từ key, ví dụ: Keyframes_L22/keyframes/L22_V002/209.jpg
                parts = key.replace("\\", "/").split("/")
                if len(parts) >= 2:
                    video_folder = parts[-2]  # L22_V002
                    frame_file = parts[-1]     # 209.jpg
                    frame_id = int(os.path.splitext(frame_file)[0])
                    video_id = f"{video_folder}.mp4"
                    
                    # Xây dựng đường dẫn ảnh
                    image_path = os.path.join(self.keyframes_dir, key)
                    if os.path.exists(image_path):
                        ocr_hits.append({
                            "video_id": video_id,
                            "frame_id": frame_id,
                            "image_path": image_path,
                            "ocr_score": ocr_score,
                            "clip_score": 0.0,  # Không có SigLIP score
                        })

        # Sắp xếp theo ocr_score giảm dần
        ocr_hits.sort(key=lambda x: x["ocr_score"], reverse=True)
        return ocr_hits[:200]  # Giới hạn 200 kết quả OCR

    def search(self, raw_query):
        parsed_query_data = analyze_query_offline_mt(raw_query)
        clip_query = parsed_query_data["clip_query"]
        required_objects = parsed_query_data["required_objects"]

        text_inputs = self.siglip_processor(text=[clip_query], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.siglip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            text_vector = text_features[0].cpu().tolist()

        search_results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[text_vector],
            limit=1000,
            output_fields=["video_id", "frame_id"],
            search_params={"metric_type": "IP", "params": {"ef": 128}}
        )

        # --- Kênh 1: SigLIP candidates ---
        siglip_candidates = []
        existing_ocr_keys = set()
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
                rel_path = os.path.relpath(image_path, self.keyframes_dir)
                existing_ocr_keys.add(rel_path)
                ocr_score = self.get_ocr_score(rel_path, raw_query)
                siglip_candidates.append({
                    "video_id": v_id,
                    "frame_id": f_id,
                    "image_path": image_path,
                    "clip_score": hit["distance"],
                    "ocr_score": ocr_score,
                })

        # --- Kênh 2: OCR search độc lập (quét toàn bộ OCR DB) ---
        ocr_only_hits = self._ocr_search(raw_query, existing_ocr_keys)
        if ocr_only_hits:
            print(f"[OCR Search] Tìm thêm {len(ocr_only_hits)} frame từ OCR database")

        # --- Merge 2 kênh ---
        all_candidates = siglip_candidates + ocr_only_hits

        # Pre-ranking: SigLIP + OCR
        for c in all_candidates:
            c["pre_score"] = c["clip_score"] + (c["ocr_score"] * 0.5)
        all_candidates.sort(key=lambda x: x["pre_score"], reverse=True)
        top_candidates = all_candidates[:self.max_answers]

        # Florence-2 re-ranking
        cand_paths = [c["image_path"] for c in top_candidates]
        florence_scores = self.get_florence_scores_batch(cand_paths, required_objects)

        # Final scoring
        scored_results = []
        for cand, florence_score in zip(top_candidates, florence_scores):
            clip_score = cand["clip_score"]
            ocr_score = cand["ocr_score"]
            
            # Base score = weighted sum of 3 models
            final_score = (0.6 * clip_score) + (0.3 * florence_score) + (0.1 * ocr_score)
            
            # OCR override: chữ trên màn hình là bằng chứng mạnh nhất
            # Khi OCR score rất cao, đặt sàn điểm (floor) để đảm bảo
            # frame có text luôn lên top, bất kể visual model có nhận diện hay không.
            if ocr_score >= 0.6:
                # Sàn = 0.95 + bonus theo mức OCR (tối đa ~1.05)
                floor = 0.90 + ocr_score * 0.15
                final_score = max(final_score, floor)
            elif ocr_score >= 0.5:
                floor = 0.80 + ocr_score * 0.10
                final_score = max(final_score, floor)
            elif ocr_score >= 0.3:
                final_score += 0.2
                
            scored_results.append({
                "video_id": cand["video_id"],
                "frame_id": cand["frame_id"],
                "image_path": cand["image_path"],
                "score": final_score,
                "clip": clip_score, "flo": florence_score, "ocr": ocr_score
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:self.max_answers]

    def format_submission(self, results):
        formatted = []
        for r in results:
            vid = r["video_id"].replace(".mp4", "")
            fid = r["frame_id"]
            formatted.append(f"{vid},{fid}")
        return formatted


def show_top_k_images(results, k=5, query_text=""):
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
        img_path = r.get("image_path", "")
        if img_path and os.path.exists(img_path):
            img = PILImage.open(img_path).convert("RGB")
            images.append(img)
            labels.append(
                f"#{i+1} {r['video_id']} - {r['frame_id']}\n"
                f"Score: {r['score']:.4f} | SigLIP: {r['clip']:.4f} | Flo: {r['flo']:.2f} | OCR: {r['ocr']:.2f}"
            )
        else:
            # Placeholder
            img = PILImage.new("RGB", (480, 270), color=(40, 40, 40))
            images.append(img)
            labels.append(f"#{i+1} {r['video_id']} - {r['frame_id']} (không tìm thấy ảnh)")

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
    output_path = "search_results_preview.jpg"
    grid.save(output_path, quality=90)
    print(f"\n[Preview] Đã lưu ảnh kết quả: {output_path}")
    
    # Thử mở ảnh tự động
    try:
        import subprocess
        subprocess.Popen(["xdg-open", output_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        bm25_retriever = BM25Retriever(corpus_path=bm25_corpus_path, frame_map_path=frame_map_path) if os.path.exists(bm25_corpus_path) and os.path.exists(frame_map_path) else None

        hybrid = HybridRetriever(
            vector_retriever=vector_retriever, bm25_retriever=bm25_retriever,
            clip_weight=cfg["retrieval"]["clip_weight"], bm25_weight=cfg["retrieval"]["bm25_weight"]
        )

        qa_searcher = QASearcher(
            retriever=hybrid, vqa_model_name=cfg["vqa"]["model"], device=cfg["vqa"]["device"],
            top_k_frames_for_vqa=cfg["vqa"]["top_k_frames"], max_answers=cfg["retrieval"]["final_top_k"]
        )

        trake_searcher = TRAKESearcher(
            clip_retriever=vector_retriever, top_k_per_event=cfg["trake"]["top_k_per_event"],
            max_answers=cfg["retrieval"]["final_top_k"]
        )
    except Exception as e:
        print(f"[WARN] Không thể khởi tạo CLIP/QA/TRAKE retriever: {e}")
        print("  → KIS search (SigLIP + Florence-2) vẫn hoạt động bình thường.")

    # --- KIS searcher (SigLIP + Florence-2) — luôn khởi tạo ---
    kis_searcher = FlorenceKISSearcher(
        db_path=SIGLIP_DB_PATH,
        collection_name=SIGLIP_COLLECTION,
        keyframes_dir=cfg["data"].get("keyframes_root", "DATASET"),
        ocr_db_path=OCR_DB_PATH,
        max_answers=cfg["retrieval"]["final_top_k"],
        batch_size=8
    )

    return {
        "vector": vector_retriever,
        "clip":   vector_retriever,
        "bm25":   bm25_retriever,
        "hybrid": hybrid,
        "kis":    kis_searcher,
        "qa":     qa_searcher,
        "trake":  trake_searcher
    }


def run_single_query(components: dict, args, cfg: dict):
    qtype = args.type.lower()
    
    if qtype in ("kis", "textual_kis"):
        results = components["kis"].search(args.query)
        print("\n=== KẾT QUẢ TÌM KIẾM CHÍNH THỨC ===")
        for i, r in enumerate(results[:5]):
            print(f"▶ {r['video_id']} - {r['frame_id']} | Tổng: {r['score']:.4f} (SigLIP: {r['clip']:.4f} | Florence: {r['flo']:.4f} | OCR: {r['ocr']:.4f})")
        formatted = components["kis"].format_submission(results)
        
        # Hiển thị ảnh nếu --show-images
        if getattr(args, 'show_images', False):
            show_k = getattr(args, 'show_k', 5)
            show_top_k_images(results, k=show_k, query_text=args.query)

    elif qtype in ("qa", "vqa"):
        results = components["qa"].search(query=args.query, question=args.question, use_vqa=not args.no_vqa)
        formatted = components["qa"].format_submission(results)

    elif qtype == "trake":
        results = components["trake"].search(args.events)
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
        max_answers=cfg["retrieval"]["final_top_k"]
    )

    all_submissions, all_query_results = [], []

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

            all_submissions.append(manager.build_query_submission({"query_id": qid, "query_type": qtype}, results))

            if args.evaluate and "ground_truth" in q:
                all_query_results.append({
                    "query_id": qid, "query_type": qtype,
                    "answers": results, "ground_truth": q["ground_truth"]
                })
        except Exception as e:
            all_submissions.append({"query_id": qid, "query_type": qtype, "answers": [], "error": str(e)})

    manager.save_all(all_submissions)

    if args.evaluate and all_query_results:
        eval_result = evaluate_dataset(all_query_results)
        print_evaluation_report(eval_result)
        report_path = Path(args.output or cfg["submission"]["output_dir"]) / "eval_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in eval_result.items() if k != "per_query"}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="AIC2026 Baseline")
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
    parser.add_argument("--show-images", action="store_true", help="Hiển thị top-k ảnh kết quả sau khi search")
    parser.add_argument("--show-k", type=int, default=5, help="Số ảnh hiển thị (mặc định 5)")

    args = parser.parse_args()
    cfg = load_config(args.config)
    components = init_components(cfg)

    if args.query_file:
        run_batch_queries(components, args, cfg)
    elif args.query:
        run_single_query(components, args, cfg)
    else:
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

