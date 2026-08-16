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


class FlorenceKISSearcher:
    def __init__(
        self,
        db_path,
        collection_name,
        keyframes_dir,
        ocr_db_path,
        max_answers=100,
        batch_size=8,
    ):
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
        print(f"Debug: Milvus db_path = {db_path}")
        self.milvus_client.load_collection(collection_name)

        self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        self.siglip_model = (
            AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(self.device).eval()
        )

        self.florence_processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_NAME, trust_remote_code=True
        )
        self.florence_model = (
            AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_NAME, trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )

    def get_florence_scores_batch(self, image_paths, required_objects):
        if not required_objects or not image_paths:
            return (
                [1.0] * len(image_paths)
                if not required_objects
                else [0.0] * len(image_paths)
            )

        text_input = " and ".join(required_objects)
        prompt = f"<CAPTION_TO_PHRASE_GROUNDING> {text_input}"
        all_scores = []

        for i in tqdm(
            range(0, len(image_paths), self.batch_size),
            desc="[Florence-2] Re-ranking",
            unit="batch",
            leave=False,
        ):
            batch_paths = image_paths[i : i + self.batch_size]
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
                inputs = self.florence_processor(
                    text=prompts, images=images, return_tensors="pt", padding=True
                ).to(self.device)
                with torch.no_grad():
                    generated_ids = self.florence_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        num_beams=3,
                    )
                generated_texts = self.florence_processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )
                batch_scores = [0.0] * len(batch_paths)

                for j, (gen_text, img) in enumerate(zip(generated_texts, images)):
                    parsed_answer = self.florence_processor.post_process_generation(
                        gen_text,
                        task="<CAPTION_TO_PHRASE_GROUNDING>",
                        image_size=(img.width, img.height),
                    )
                    results = parsed_answer.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
                    labels_found = results.get("labels", [])
                    if labels_found:
                        unique_labels_found = set(
                            [lbl.lower().strip() for lbl in labels_found]
                        )

                        matched_count = 0
                        for req_obj in required_objects:
                            req_obj_lower = req_obj.lower().strip()
                            if any(
                                req_obj_lower in lbl or lbl in req_obj_lower
                                for lbl in unique_labels_found
                            ):
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

        search_text_no_accents = "".join(
            c for c in unicodedata.normalize("NFD", search_text.lower()) if unicodedata.category(c) != "Mn"
        ).replace("đ", "d")
        search_phrase = re.sub(r'[^\w\s]', ' ', search_text_no_accents)
        search_phrase = re.sub(r'\s+', ' ', search_phrase).strip()
        if not search_phrase:
            return 0.0

        ocr_text_no_accents = "".join(
            c for c in unicodedata.normalize("NFD", ocr_text.lower()) if unicodedata.category(c) != "Mn"
        ).replace("đ", "d")
        ocr_norm = re.sub(r'[^\w\s]', ' ', ocr_text_no_accents)
        ocr_norm = re.sub(r'\s+', ' ', ocr_norm).strip()

        # 1) Khớp nguyên cụm từ đầy đủ
        if re.search(rf'\b{re.escape(search_phrase)}\b', ocr_norm) or \
           re.search(rf'\b{re.escape(search_phrase.replace(" ", ""))}\b', ocr_norm):
            return 1.0

        keywords = search_phrase.split()
        n = len(keywords)

        # 2) Sliding window: tìm cụm từ CON liên tiếp dài nhất có trong OCR
        #    Ví dụ: query = "ban tin ve tai nan giao thong tai dak lak"
        #           OCR chứa "tai nan giao thong tai dak lak" → khớp 7/10 từ liên tiếp
        best_sub_ratio = 0.0
        for win_len in range(n - 1, 1, -1):  # Từ dài đến ngắn (tối thiểu 2 từ)
            for start in range(n - win_len + 1):
                sub_phrase = " ".join(keywords[start:start + win_len])
                if re.search(rf'\b{re.escape(sub_phrase)}\b', ocr_norm) or \
                   re.search(rf'\b{re.escape(sub_phrase.replace(" ", ""))}\b', ocr_norm):
                    best_sub_ratio = win_len / n
                    break
            if best_sub_ratio > 0:
                break

        if best_sub_ratio >= 0.5:
            # Cụm con liên tiếp dài ≥ 50% query → điểm cao (0.6 ~ 0.95)
            return 0.5 + best_sub_ratio * 0.5

        # 3) Fallback: đếm từ khóa rời rạc, giảm 50% điểm (tối đa 0.5)
        matched = sum(1 for kw in keywords if re.search(rf'\b{re.escape(kw)}\b', ocr_norm))
        return (matched / n) * 0.5

    def _ocr_search(self, raw_query, existing_keys):
        """Quét toàn bộ OCR database tìm frame có chữ khớp với query (độc lập với SigLIP)."""


        if not self.ocr_data or not raw_query:
            return []

        search_text = "".join(
            c
            for c in unicodedata.normalize("NFD", raw_query.lower())
            if unicodedata.category(c) != "Mn"
        )
        search_text = search_text.replace("đ", "d")
        search_phrase = re.sub(r'[^\w\s]', ' ', search_text)
        search_phrase = re.sub(r'\s+', ' ', search_phrase).strip()
        if not search_phrase:
            return []
            
        keywords = search_phrase.split()

        ocr_hits = []
        for key, ocr_text in self.ocr_data.items():
            if key in existing_keys:
                continue
                
            ocr_text_no_accents = "".join(
                c for c in unicodedata.normalize("NFD", ocr_text.lower()) if unicodedata.category(c) != "Mn"
            ).replace("đ", "d")
            ocr_norm = re.sub(r'[^\w\s]', ' ', ocr_text_no_accents)
            ocr_norm = re.sub(r'\s+', ' ', ocr_norm).strip()
            n = len(keywords)

            # Ưu tiên khớp nguyên cụm
            if re.search(rf'\b{re.escape(search_phrase)}\b', ocr_norm) or \
               re.search(rf'\b{re.escape(search_phrase.replace(" ", ""))}\b', ocr_norm):
                ocr_score = 1.0
            else:
                # Sliding window: tìm cụm từ CON liên tiếp dài nhất có trong OCR
                best_sub_ratio = 0.0
                for win_len in range(n - 1, 1, -1):
                    found = False
                    for start in range(n - win_len + 1):
                        sub_phrase = " ".join(keywords[start:start + win_len])
                        if re.search(rf'\b{re.escape(sub_phrase)}\b', ocr_norm) or \
                           re.search(rf'\b{re.escape(sub_phrase.replace(" ", ""))}\b', ocr_norm):
                            best_sub_ratio = win_len / n
                            found = True
                            break
                    if found:
                        break

                if best_sub_ratio >= 0.5:
                    ocr_score = 0.5 + best_sub_ratio * 0.5
                else:
                    matched = sum(1 for kw in keywords if re.search(rf'\b{re.escape(kw)}\b', ocr_norm))
                    ocr_score = (matched / n) * 0.5

            if ocr_score >= 0.3:  # Hạ threshold để không bỏ lọt partial match mạnh
                # Parse video_id và frame_id từ key, ví dụ: Keyframes_L22/keyframes/L22_V002/209.jpg
                parts = key.replace("\\", "/").split("/")
                if len(parts) >= 2:
                    video_folder = parts[-2]  # L22_V002
                    frame_file = parts[-1]  # 209.jpg
                    frame_id = int(os.path.splitext(frame_file)[0])
                    video_id = f"{video_folder}.mp4"

                    # Xây dựng đường dẫn ảnh
                    image_path = os.path.join(self.keyframes_dir, key)
                    if os.path.exists(image_path):
                        ocr_hits.append(
                            {
                                "video_id": video_id,
                                "frame_id": frame_id,
                                "image_path": image_path,
                                "ocr_score": ocr_score,
                                "clip_score": 0.0,  # Không có SigLIP score
                            }
                        )

        # Sắp xếp theo ocr_score giảm dần
        ocr_hits.sort(key=lambda x: x["ocr_score"], reverse=True)
        return ocr_hits[:200]  # Giới hạn 200 kết quả OCR

    def search(self, raw_query, search_mode="hybrid"):
        if search_mode == "text":
            print("\n[Search Mode] Thuần Chữ (OCR-only). Bỏ qua SigLIP/Florence.")
            ocr_only_hits = self._ocr_search(raw_query, set())
            if ocr_only_hits:
                print(f"[OCR Search] Tìm thấy {len(ocr_only_hits)} frame khớp chữ.")
            for c in ocr_only_hits:
                c["score"] = c["ocr_score"]
                c["clip"] = 0.0
                c["flo"] = 0.0
                c["ocr"] = c["ocr_score"]
            return ocr_only_hits[:self.max_answers]

        parsed_query_data = analyze_query_offline_mt(raw_query)
        clip_query = parsed_query_data["clip_query"]
        required_objects = parsed_query_data["required_objects"]
        proper_nouns = parsed_query_data.get("proper_nouns", [])
        has_proper_nouns = len(proper_nouns) > 0

        # Tạo query "thuần hình ảnh" cho SigLIP bằng cách loại bỏ danh từ riêng
        # SigLIP không thể nhìn hình mà biết đó là "Tam Kì" hay "Quảng Nam"
        # nên giữ lại chỉ làm nhiễu, ví dụ: "Bicycle racers at Tam Kì, Quảng Nam" -> "Bicycle racers"
        siglip_query = clip_query
        if has_proper_nouns:
            for pn in proper_nouns:
                pn_clean = pn.rstrip(',')
                siglip_query = re.sub(r'(?i)\b' + re.escape(pn_clean) + r'\b[,]?', '', siglip_query)
            # Xóa giới từ thừa sau khi loại danh từ riêng
            siglip_query = re.sub(r'\b(at|in|on|from|near|of)\s*$', '', siglip_query.strip())
            siglip_query = re.sub(r'\b(at|in|on|from|near|of)\s*,', ',', siglip_query)
            siglip_query = re.sub(r',\s*,', ',', siglip_query)
            siglip_query = re.sub(r',\s*$', '', siglip_query)
            siglip_query = re.sub(r'\s+', ' ', siglip_query).strip()
            if siglip_query:
                print(f"-> SigLIP visual query: '{siglip_query}'")
            else:
                siglip_query = clip_query  # Fallback

        text_inputs = self.siglip_processor(
            text=[siglip_query], padding="max_length", return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            text_features = self.siglip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )
            text_vector = text_features[0].cpu().tolist()

        search_results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[text_vector],
            limit=1000,
            output_fields=["video_id", "frame_id"],
            search_params={"metric_type": "IP", "params": {"ef": 128}},
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
                f"{f_id}.jpg",
                f"{f_id:03d}.jpg",
                f"{f_id:04d}.jpg",
                f"{f_id:05d}.jpg",
                f"{f_id:06d}.jpg",
                f"{f_id}.png",
                f"{f_id:04d}.png",
            ]
            image_path = None
            for fmt in possible_formats:
                temp_path = os.path.join(
                    self.keyframes_dir,
                    batch_folder_name,
                    "keyframes",
                    video_folder,
                    fmt,
                )
                if os.path.exists(temp_path):
                    image_path = temp_path
                    break

            if image_path:
                rel_path = os.path.relpath(image_path, self.keyframes_dir)
                existing_ocr_keys.add(rel_path)
                ocr_score = 0.0 if search_mode == "visual" else self.get_ocr_score(rel_path, raw_query)
                siglip_candidates.append(
                    {
                        "video_id": v_id,
                        "frame_id": f_id,
                        "image_path": image_path,
                        "clip_score": hit["distance"],
                        "ocr_score": ocr_score,
                    }
                )

        # --- Kênh 2: OCR search độc lập (quét toàn bộ OCR DB) ---
        ocr_only_hits = []
        if search_mode != "visual":
            ocr_only_hits = self._ocr_search(raw_query, existing_ocr_keys)
            if ocr_only_hits:
                print(f"[OCR Search] Tìm thêm {len(ocr_only_hits)} frame từ OCR database")
        else:
            print("[Search Mode] Thuần Hình Ảnh (Visual-only). Bỏ qua OCR.")

        # --- Merge 2 kênh ---
        all_candidates = siglip_candidates + ocr_only_hits

        raw_q_norm = ''.join(c for c in unicodedata.normalize('NFD', raw_query.lower()) if unicodedata.category(c) != 'Mn')
        raw_q_norm = raw_q_norm.replace("đ", "d")
        raw_q_norm = re.sub(r'[^\w\s]', '', raw_q_norm)
        
        text_phrases = [" co chu ", " ban tin ", " thong bao ", " tieu de ", " van ban ", " hien chu ", " dong chu ", " chu ", " bien so ", " bien bao "]
        is_text_heavy = any(p in f" {raw_q_norm} " for p in text_phrases)

        # Pre-ranking: SigLIP + OCR
        # Nếu query tìm chữ hoặc có danh từ riêng (địa danh), cho OCR can thiệp vào pre-ranking
        if search_mode == "visual":
            ocr_weight = 0.0
        elif is_text_heavy:
            ocr_weight = 0.5
        elif has_proper_nouns:
            ocr_weight = 0.3  # Danh từ riêng cần OCR xác minh nhưng ít hơn query tìm chữ thuần
        else:
            ocr_weight = 0.0
        
        for c in all_candidates:
            c["pre_score"] = c["clip_score"] + (c["ocr_score"] * ocr_weight)
        all_candidates.sort(key=lambda x: x["pre_score"], reverse=True)
        top_candidates = all_candidates[: self.max_answers]

        # Florence-2 re-ranking
        # Lọc bỏ danh từ riêng khỏi danh sách object gửi cho Florence
        # vì Florence không thể nhìn hình mà biết đó là "Đắk Lắk" hay "Hà Nội"
        # Danh từ riêng sẽ do OCR chịu trách nhiệm xác minh
        if has_proper_nouns:
            proper_nouns_lower = [p.lower() for p in proper_nouns]
            florence_objects = [obj for obj in required_objects if not any(pn in obj for pn in proper_nouns_lower)]
            if not florence_objects:
                florence_objects = required_objects  # Fallback nếu lọc hết
        else:
            florence_objects = required_objects
        
        cand_paths = [c["image_path"] for c in top_candidates]
        florence_scores = self.get_florence_scores_batch(cand_paths, florence_objects)

        # Final scoring
        scored_results = []
        for cand, florence_score in zip(top_candidates, florence_scores):
            clip_score = cand["clip_score"]
            ocr_score = cand["ocr_score"]

            # OCR-only candidates (SigLIP=0): Florence hay ảo tưởng vì không có bằng chứng hình ảnh
            # Giảm trọng số Florence xuống 50% cho những frame này
            effective_florence = florence_score * 0.5 if clip_score == 0.0 else florence_score

            # Base score = weighted sum of 3 models (cố định cho mọi loại query)
            if search_mode == "visual":
                final_score = (0.7 * clip_score) + (0.3 * effective_florence)
            else:
                final_score = (0.6 * clip_score) + (0.3 * effective_florence) + (0.1 * ocr_score)

                # OCR bonus: phân loại query 3 tầng
                if is_text_heavy:
                    # Query tìm chữ: OCR là bằng chứng mạnh nhất, cho phép override
                    if ocr_score >= 0.6:
                        floor = 0.90 + ocr_score * 0.15
                        final_score = max(final_score, floor)
                    elif ocr_score >= 0.5:
                        floor = 0.80 + ocr_score * 0.10
                        final_score = max(final_score, floor)
                    elif ocr_score >= 0.4:
                        final_score += 0.3
                elif has_proper_nouns:
                    # Query có danh từ riêng: thưởng điểm CHỈ KHI có cả bằng chứng hình ảnh VÀ text
                    # Tránh frame rác (SigLIP=0) chiếm top chỉ nhờ text match
                    if clip_score > 0 and ocr_score >= 0.3:
                        # Bonus tỉ lệ thuận với OCR score — frame có text đúng sẽ được đẩy lên
                        final_score += ocr_score * 0.5
                else:
                    # Query thuần hình ảnh: OCR chỉ là điểm thưởng nhỏ
                    if ocr_score >= 0.8:
                        final_score += 0.10
                    elif ocr_score >= 0.5:
                        final_score += 0.05
                    elif ocr_score >= 0.3:
                        final_score += 0.02
                
            scored_results.append({
                "video_id": cand["video_id"],
                "frame_id": cand["frame_id"],
                "image_path": cand["image_path"],
                "score": final_score,
                "clip": clip_score, "flo": florence_score, "ocr": ocr_score
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[: self.max_answers]

    def format_submission(self, results):
        formatted = []
        for r in results:
            vid = r["video_id"].replace(".mp4", "")
            fid = r["frame_id"]
            formatted.append(f"{vid},{fid}")
        return formatted


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
        img_path = r.get("image_path", "")
        if img_path and os.path.exists(img_path):
            img = PILImage.open(img_path).convert("RGB")
            images.append(img)
            labels.append(
                f"#{i + 1} {r['video_id']} - {r['frame_id']}\n"
                f"Score: {r['score']:.4f} | SigLIP: {r['clip']:.4f} | Flo: {r['flo']:.2f} | OCR: {r['ocr']:.2f}"
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
        db_path=SIGLIP_DB_PATH,
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
