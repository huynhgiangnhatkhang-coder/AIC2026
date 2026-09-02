import sys
import os
import re
import json
import unicodedata
import torch
from PIL import Image
from pymilvus import MilvusClient
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel
from src.query.translate import analyze_query_offline_mt
from tqdm import tqdm

SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
FLORENCE_MODEL_NAME = "microsoft/Florence-2-base"

class FlorenceKISSearcher:
    def __init__(
        self,
        vector_retriever,
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
        self.vector_retriever = vector_retriever
        self.milvus_client = vector_retriever._client if vector_retriever else None

        self.ocr_data = {}
        self._ocr_lookup = {}  # video_folder/frame_file -> ocr_text
        self._ocr_norm_cache = {}  # key -> normalized ocr text (pre-computed)
        self._ocr_norm_by_lookup = {}  # "video_folder/frame_file" -> normalized ocr text
        if os.path.exists(ocr_db_path):
            with open(ocr_db_path, "r", encoding="utf-8") as f:
                self.ocr_data = json.load(f)
            # Build O(1) lookup and pre-normalize all OCR text
            for key, val in self.ocr_data.items():
                parts = key.replace("\\", "/").split("/")
                lookup_key = None
                if len(parts) >= 2:
                    lookup_key = f"{parts[-2]}/{parts[-1]}"
                    self._ocr_lookup[lookup_key] = val
                # Pre-normalize OCR text (biggest perf win)
                norm = "".join(
                    c for c in unicodedata.normalize("NFD", val.lower()) if unicodedata.category(c) != "Mn"
                ).replace("đ", "d")
                norm = re.sub(r'[^\w\s]', ' ', norm)
                norm = re.sub(r'\s+', ' ', norm).strip()
                self._ocr_norm_cache[key] = norm
                if lookup_key:
                    self._ocr_norm_by_lookup[lookup_key] = norm
            print(f"[OCR] Built lookup index: {len(self._ocr_lookup)} entries, pre-normalized: {len(self._ocr_norm_cache)}")

        # Build file existence cache for all keyframe dirs
        self._frame_path_cache = {}  # (video_folder, frame_id) -> absolute_path
        self._build_frame_cache(keyframes_dir)

        print(f"Debug: Milvus collection = {collection_name}")
        if self.milvus_client:
            self.milvus_client.load_collection(collection_name)
        else:
            print("[WARN] Milvus client is None. SigLIP visual search will be disabled.")

        self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        self.siglip_model = (
            AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(self.device).eval()
        )

        self.florence_processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_NAME, trust_remote_code=True
        )
        self.florence_model = (
            AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )

    def _build_frame_cache(self, keyframes_dir):
        """Pre-scan all keyframe directories to build a path cache."""
        import glob
        dataset_dir = keyframes_dir
        pattern = os.path.join(dataset_dir, "Keyframes_*", "keyframes", "*", "*.*")
        count = 0
        for filepath in glob.iglob(pattern):
            parts = filepath.replace("\\", "/").split("/")
            video_folder = parts[-2]
            frame_file = parts[-1]
            frame_base = os.path.splitext(frame_file)[0]
            try:
                frame_id = int(frame_base)
                self._frame_path_cache[(video_folder, frame_id)] = filepath
                count += 1
            except ValueError:
                pass
        print(f"[Cache] Built frame path cache: {count} files")

    @staticmethod
    def _extract_lookup_key(image_rel_path):
        """Extract 'video_folder/frame_file' from a relative path."""
        parts = image_rel_path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        return image_rel_path

    def _get_ocr_text(self, image_rel_path):
        """O(1) OCR text lookup."""
        lookup_key = self._extract_lookup_key(image_rel_path)
        return self._ocr_lookup.get(lookup_key, "")

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
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
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
            except Exception as e:
                print(f"[Florence] Lỗi khi inference batch: {e}")
                all_scores.extend([0.0] * len(batch_paths))
                
            try:
                del inputs
                del generated_ids
                del generated_texts
            except:
                pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_scores

    def get_ocr_score(self, image_rel_path, search_text):
        if not search_text or not self._ocr_norm_by_lookup:
            return 0.0
        
        # O(1) lookup for pre-normalized OCR text
        lookup_key = self._extract_lookup_key(image_rel_path)
        ocr_norm = self._ocr_norm_by_lookup.get(lookup_key, "")
        if not ocr_norm:
            return 0.0

        search_text_no_accents = "".join(
            c for c in unicodedata.normalize("NFD", search_text.lower()) if unicodedata.category(c) != "Mn"
        ).replace("đ", "d")
        search_phrase = re.sub(r'[^\w\s]', ' ', search_text_no_accents)
        search_phrase = re.sub(r'\s+', ' ', search_phrase).strip()
        if not search_phrase:
            return 0.0

        # 1) Full phrase match
        if search_phrase in ocr_norm or search_phrase.replace(" ", "") in ocr_norm:
            return 1.0

        keywords = search_phrase.split()
        n = len(keywords)

        # 2) Sliding window
        best_sub_ratio = 0.0
        for win_len in range(n - 1, 1, -1):
            for start in range(n - win_len + 1):
                sub_phrase = " ".join(keywords[start:start + win_len])
                if sub_phrase in ocr_norm or sub_phrase.replace(" ", "") in ocr_norm:
                    best_sub_ratio = win_len / n
                    break
            if best_sub_ratio > 0:
                break

        if best_sub_ratio >= 0.5:
            return 0.5 + best_sub_ratio * 0.5

        # 3) Fallback
        matched = sum(1 for kw in keywords if kw in ocr_norm)
        return (matched / n) * 0.5

    def _ocr_search(self, raw_query, existing_keys):
        """Quét toàn bộ OCR database tìm frame có chữ khớp với query. (Optimized)"""
        if not self._ocr_norm_cache or not raw_query:
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
        n = len(keywords)

        ocr_hits = []
        for key, ocr_norm in self._ocr_norm_cache.items():
            if key in existing_keys:
                continue
            if not ocr_norm:
                continue

            # Fast pre-filter: any keyword present as substring?
            has_any_word = False
            for kw in keywords:
                if kw in ocr_norm:
                    has_any_word = True
                    break
            if not has_any_word:
                continue

            # Score using simple string `in` instead of regex
            # 1) Full phrase match
            if search_phrase in ocr_norm or search_phrase.replace(" ", "") in ocr_norm:
                ocr_score = 1.0
            else:
                best_sub_ratio = 0.0
                for win_len in range(n - 1, 1, -1):
                    found = False
                    for start in range(n - win_len + 1):
                        sub_phrase = " ".join(keywords[start:start + win_len])
                        if sub_phrase in ocr_norm or sub_phrase.replace(" ", "") in ocr_norm:
                            best_sub_ratio = win_len / n
                            found = True
                            break
                    if found:
                        break

                if best_sub_ratio >= 0.5:
                    ocr_score = 0.5 + best_sub_ratio * 0.5
                else:
                    matched = sum(1 for kw in keywords if kw in ocr_norm)
                    ocr_score = (matched / n) * 0.5

            if ocr_score >= 0.3:
                parts = key.replace("\\", "/").split("/")
                if len(parts) >= 2:
                    video_folder = parts[-2]
                    frame_file = parts[-1]
                    frame_id = int(os.path.splitext(frame_file)[0])
                    video_id = f"{video_folder}.mp4"

                    # Use frame cache instead of os.path.exists
                    image_path = self._frame_path_cache.get((video_folder, frame_id))
                    if not image_path:
                        image_path = os.path.join(self.keyframes_dir, key)
                        if not os.path.exists(image_path):
                            continue
                    
                    ocr_hits.append(
                        {
                            "video_id": video_id,
                            "frame_id": frame_id,
                            "image_path": image_path,
                            "ocr_score": ocr_score,
                            "clip_score": 0.0,
                        }
                    )

        ocr_hits.sort(key=lambda x: x["ocr_score"], reverse=True)
        return ocr_hits[:200]

    def search(self, raw_query, search_mode="hybrid", object_hints=None, top_k=None, **kwargs):
        actual_top_k = top_k if top_k is not None else self.max_answers
        print(f"DEBUG: florence_kis.search called with top_k={top_k}, actual_top_k={actual_top_k}")
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
            return ocr_only_hits[:actual_top_k]

        parsed_query_data = analyze_query_offline_mt(raw_query)
        clip_query = parsed_query_data["clip_query"]
        required_objects = parsed_query_data["required_objects"]
        proper_nouns = parsed_query_data.get("proper_nouns", [])
        has_proper_nouns = len(proper_nouns) > 0

        siglip_query = clip_query
        if has_proper_nouns:
            for pn in proper_nouns:
                pn_clean = pn.rstrip(',')
                siglip_query = re.sub(r'(?i)\b' + re.escape(pn_clean) + r'\b[,]?', '', siglip_query)
            siglip_query = re.sub(r'\b(at|in|on|from|near|of)\s*$', '', siglip_query.strip())
            siglip_query = re.sub(r'\b(at|in|on|from|near|of)\s*,', ',', siglip_query)
            siglip_query = re.sub(r',\s*,', ',', siglip_query)
            siglip_query = re.sub(r',\s*$', '', siglip_query)
            siglip_query = re.sub(r'\s+', ' ', siglip_query).strip()
            if siglip_query:
                print(f"-> SigLIP visual query: '{siglip_query}'")
            else:
                siglip_query = clip_query

        siglip_candidates = []
        existing_ocr_keys = set()
        
        if self.milvus_client:
            text_inputs = self.siglip_processor(
                text=[siglip_query], padding="max_length", return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                text_features = self.siglip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                text_vector = text_features[0].cpu().tolist()

            try:
                search_results = self.milvus_client.search(
                    collection_name=self.collection_name,
                    data=[text_vector],
                    limit=1000,
                    output_fields=["video_id", "frame_id"],
                    search_params={"metric_type": "IP", "params": {"ef": 1024}},
                )
            except Exception as e:
                if "metric type not match" in str(e).lower() or "expected=cosine" in str(e).lower():
                    search_results = self.milvus_client.search(
                        collection_name=self.collection_name,
                        data=[text_vector],
                        limit=1000,
                        output_fields=["video_id", "frame_id"],
                        search_params={"metric_type": "COSINE", "params": {"ef": 1024}},
                    )
                else:
                    raise e
            
            search_hits = search_results[0]
        else:
            search_hits = []

        for hit in search_hits:
            v_id = hit["entity"]["video_id"]
            f_id = hit["entity"]["frame_id"]
            video_folder = v_id.replace(".mp4", "")

            # O(1) cache lookup instead of 7x os.path.exists()
            image_path = self._frame_path_cache.get((video_folder, f_id))

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

        ocr_only_hits = []
        if search_mode != "visual":
            ocr_only_hits = self._ocr_search(raw_query, existing_ocr_keys)
            if ocr_only_hits:
                print(f"[OCR Search] Tìm thêm {len(ocr_only_hits)} frame từ OCR database")
        else:
            print("[Search Mode] Thuần Hình Ảnh (Visual-only). Bỏ qua OCR.")

        all_candidates = siglip_candidates + ocr_only_hits

        raw_q_norm = ''.join(c for c in unicodedata.normalize('NFD', raw_query.lower()) if unicodedata.category(c) != 'Mn')
        raw_q_norm = raw_q_norm.replace("đ", "d")
        raw_q_norm = re.sub(r'[^\w\s]', '', raw_q_norm)
        
        text_phrases = [" co chu ", " ban tin ", " thong bao ", " tieu de ", " van ban ", " hien chu ", " dong chu ", " chu ", " bien so ", " bien bao "]
        is_text_heavy = any(p in f" {raw_q_norm} " for p in text_phrases)

        if search_mode == "visual":
            ocr_weight = 0.0
        elif is_text_heavy:
            ocr_weight = 0.5
        elif has_proper_nouns:
            ocr_weight = 0.3
        else:
            ocr_weight = 0.0
        
        for c in all_candidates:
            c["pre_score"] = c["clip_score"] + (c["ocr_score"] * ocr_weight)
        all_candidates.sort(key=lambda x: x["pre_score"], reverse=True)
        top_candidates = all_candidates[: actual_top_k]

        if has_proper_nouns:
            proper_nouns_lower = [p.lower() for p in proper_nouns]
            florence_objects = [obj for obj in required_objects if not any(pn in obj for pn in proper_nouns_lower)]
            if not florence_objects:
                florence_objects = required_objects
        else:
            florence_objects = required_objects
        
        cand_paths = [c["image_path"] for c in top_candidates]
        florence_scores = self.get_florence_scores_batch(cand_paths, florence_objects)

        scored_results = []
        for cand, florence_score in zip(top_candidates, florence_scores):
            clip_score = cand["clip_score"]
            ocr_score = cand["ocr_score"]

            effective_florence = florence_score * 0.5 if clip_score == 0.0 else florence_score

            if search_mode == "visual":
                final_score = (0.7 * clip_score) + (0.3 * effective_florence)
            else:
                final_score = (0.6 * clip_score) + (0.3 * effective_florence) + (0.1 * ocr_score)

                if is_text_heavy:
                    if ocr_score >= 0.6:
                        floor = 0.90 + ocr_score * 0.15
                        final_score = max(final_score, floor)
                    elif ocr_score >= 0.5:
                        floor = 0.80 + ocr_score * 0.10
                        final_score = max(final_score, floor)
                    elif ocr_score >= 0.4:
                        final_score += 0.3
                elif has_proper_nouns:
                    if ocr_score >= 0.7:
                        floor = 0.80 + ocr_score * 0.10
                        final_score = max(final_score, floor)
                    elif clip_score > 0 and ocr_score >= 0.3:
                        final_score += ocr_score * 0.5
                else:
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
        return scored_results[: actual_top_k]

    def format_submission(self, answers):
        from src.frame_id_map import frame_id_from_index
        lines = []
        for ans in answers:
            frame_id = frame_id_from_index(ans["video_id"], ans["frame_id"])
            line = f"{ans['video_id']}, {frame_id}"
            lines.append(line)
        return lines
