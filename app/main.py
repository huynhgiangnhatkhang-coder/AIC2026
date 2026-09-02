"""
AIC 2026 Baseline — FastAPI REST API
======================================
Web service cho phép gửi query qua HTTP và nhận kết quả JSON.

Endpoints:
  POST /search/kis                — Textual KIS
  POST /search/qa                 — Q&A / VQA
  POST /search/trake              — TRAKE temporal
  GET  /frames/{video_id}/{frame_id}  — Phục vụ ảnh keyframe (image/jpeg|png)
  POST /frames/batch              — Lấy nhiều ảnh keyframe (base64 data URL)
  GET  /health                    — Health check

Usage:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import base64

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import yaml

from src.retrieval import CLIPRetriever, BM25Retriever, HybridRetriever
from src.query import FlorenceKISSearcher, QASearcher, TRAKESearcher
from src.submission import SubmissionManager
from src.frame_id_map import frame_id_from_index, frame_ids_from_indexes

# ── Load config ────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
    
# ── Init components (lazy — sẽ load khi nhận request đầu tiên) ─
_clip_retriever = None       # CLIPRetriever (FAISS backend)
_milvus_retriever = None     # MilvusRetriever (Milvus backend)
_bm25_retriever = None
_hybrid_retriever = None
_kis_searcher = None
_qa_searcher = None
_trake_searcher = None

_BACKEND = cfg.get("retrieval_backend", "milvus").lower()


def get_vector_retriever():
    """Trả về vector retriever (Milvus hoặc FAISS) tuỳ theo config."""
    global _milvus_retriever, _clip_retriever

    if _BACKEND == "milvus":
        if _milvus_retriever is None:
            from src.retrieval import MilvusRetriever
            _milvus_retriever = MilvusRetriever(
                db_path=cfg["index"]["milvus_db_path"],
                collection_name=cfg["index"]["milvus_collection"],
                model_name=cfg["clip"]["model_name"],
                device=cfg["clip"]["device"]
            )
        return _milvus_retriever
    else:
        if _clip_retriever is None:
            from src.retrieval import CLIPRetriever
            _clip_retriever = CLIPRetriever(
                index_path=cfg["index"]["faiss_index_path"],
                frame_map_path=cfg["index"]["frame_map_path"],
                model_name=cfg["clip"]["model_name"],
                device=cfg["clip"]["device"]
            )
        return _clip_retriever

# keep old name for backward compat
def get_clip_retriever():
    return get_vector_retriever()


def get_bm25_retriever() -> Optional[BM25Retriever]:
    global _bm25_retriever
    if _bm25_retriever is None:
        bm25_corpus_path = cfg["index"]["bm25_corpus_path"]
        frame_map_path = cfg["index"]["frame_map_path"]
        if os.path.exists(bm25_corpus_path) and os.path.exists(frame_map_path):
            _bm25_retriever = BM25Retriever(
                corpus_path=bm25_corpus_path,
                frame_map_path=frame_map_path
            )
    return _bm25_retriever


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever(
            vector_retriever=get_vector_retriever(),
            bm25_retriever=get_bm25_retriever(),
            clip_weight=cfg["retrieval"]["clip_weight"],
            bm25_weight=cfg["retrieval"]["bm25_weight"]
        )
    return _hybrid_retriever


def get_kis_searcher() -> FlorenceKISSearcher:
    global _kis_searcher
    if _kis_searcher is None:
        _kis_searcher = FlorenceKISSearcher(
            vector_retriever=get_vector_retriever(),
            collection_name=cfg["index"]["milvus_collection"],
            keyframes_dir=str(cfg["data"].get("keyframes_root", "DATASET")),
            ocr_db_path="ocr_database.json",
            max_answers=cfg["retrieval"]["final_top_k"],
            batch_size=16
        )
    return _kis_searcher


def get_qa_searcher() -> QASearcher:
    global _qa_searcher
    if _qa_searcher is None:
        _qa_searcher = QASearcher(
            kis_searcher=get_kis_searcher(),
            vqa_model_name=cfg["vqa"]["model"],
            api_url=cfg["vqa"].get("api_url", "http://aicpc.sytes.net:1234/v1/chat/completions"),
            device=cfg["vqa"]["device"],
            top_k_frames_for_vqa=cfg["vqa"]["top_k_frames"],
            max_answers=cfg["retrieval"]["final_top_k"],
            keyframes_dir=str(cfg["data"].get("keyframes_root", "DATASET"))
        )
    return _qa_searcher


def get_trake_searcher() -> TRAKESearcher:
    global _trake_searcher
    if _trake_searcher is None:
        _trake_searcher = TRAKESearcher(
            kis_searcher=get_kis_searcher(),
            top_k_per_event=cfg["trake"]["top_k_per_event"],
            max_answers=cfg["retrieval"]["final_top_k"]
        )
    return _trake_searcher


def _resolve_frame_file(video_id: str, frame_id: int) -> Optional[str]:
    """
    Giải quyết (video_id, frame_id) → đường dẫn file ảnh keyframe.

    Ưu tiên tra cứu từ các index đã build (frame_map.json, ocr_database.json,
    Milvus) trước, sau đó mới dùng heuristic trên filesystem.
    """
    from src.frame_paths import resolve_frame_path

    path = resolve_frame_path(video_id, frame_id, cfg)
    if path is not None:
        return path

    # Fallback: lấy filename chính xác từ index vector đã build (Milvus)
    try:
        retriever = get_vector_retriever()
        get_fn = getattr(retriever, "get_frame_filename", None)
        if get_fn is not None:
            fn = get_fn(video_id, frame_id)
            if fn:
                return resolve_frame_path(video_id, frame_id, cfg, exact_filename=fn)
    except Exception:
        pass

    return None


# ── FastAPI App ────────────────────────────────────────────
app = FastAPI(
    title="AIC 2026 Baseline API",
    description="Video Retrieval baseline cho AIC 2026 Vòng Sơ Tuyển",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ──────────────────────────────

class KISRequest(BaseModel):
    query: str = Field(..., description="Mô tả văn bản cần tìm", example="Một người đang mở laptop trong phòng họp")
    object_hints: Optional[List[str]] = Field(None, description="Các object keywords để boost", example=["laptop", "person"])
    top_k: int = Field(100, ge=1, le=100)
    search_mode: str = Field("hybrid", description="Chế độ tìm kiếm: hybrid, text, visual")


class QARequest(BaseModel):
    retrieval_query: str = Field(..., description="Mô tả cảnh cần tìm (cho retrieval)")
    question: str = Field(..., description="Câu hỏi VQA", example="Người phụ nữ mặc váy màu gì?")
    use_vqa: bool = Field(True, description="Có chạy VQA model không")
    top_k: int = Field(100, ge=1, le=100)


class TRAKERequest(BaseModel):
    events: List[str] = Field(..., description="Danh sách N mô tả sự kiện theo thứ tự",
                               example=["vận động viên giậm nhảy", "bay qua xà ngang", "tiếp đất"])
    top_k: int = Field(100, ge=1, le=100)


class AnswerItem(BaseModel):
    rank: int
    video_id: str
    frame_id: Optional[int] = None
    frame_ids: Optional[List[int]] = None
    answer: Optional[str] = None
    score: float
    formatted: str  # chuỗi nộp bài
    image_url: Optional[str] = Field(None, description="URL lấy ảnh keyframe, vd. /frames/L21_V001/1")


class SearchResponse(BaseModel):
    query_type: str
    num_results: int
    answers: List[AnswerItem]


class FrameRequest(BaseModel):
    video_id: str = Field(..., description="Mã video, ví dụ: L21_V001 hoặc L21_V001.mp4")
    frame_id: int = Field(..., ge=1, description="Số thứ tự frame")


class FrameBatchRequest(BaseModel):
    frames: List[FrameRequest] = Field(..., min_length=1)


class FrameBatchItem(BaseModel):
    video_id: str
    frame_id: int
    found: bool
    data_url: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "AIC2026 Baseline API",
        "version": "1.0.0"
    }


@app.get("/info")
def info():
    """Thông tin về index và model đang dùng."""
    import os
    return {
        "clip_model": cfg["clip"]["model_name"],
        "clip_device": cfg["clip"]["device"],
        "faiss_index": cfg["index"]["faiss_index_path"],
        "faiss_exists": os.path.exists(cfg["index"]["faiss_index_path"]),
        "bm25_corpus": cfg["index"]["bm25_corpus_path"],
        "bm25_exists": os.path.exists(cfg["index"]["bm25_corpus_path"]),
        "vqa_model": cfg["vqa"]["model"],
    }


@app.post("/search/kis", response_model=SearchResponse)
def search_kis(req: KISRequest):
    """
    **Textual KIS** — Tìm kiếm theo mô tả văn bản.
    
    Trả về top-100 cặp (video_id, frame_id).
    """
    try:
        searcher = get_kis_searcher()
        print(f"DEBUG: /search/kis called with req.top_k = {req.top_k}")
        results = searcher.search(
            raw_query=req.query,
            object_hints=req.object_hints,
            search_mode=req.search_mode,
            top_k=req.top_k
        )
        print(f"DEBUG: searcher.search returned {len(results)} results")

        answers = []
        for i, r in enumerate(results[:req.top_k], 1):
            idx = r["frame_id"]
            fid = frame_id_from_index(r["video_id"], idx)
            answers.append(AnswerItem(
                rank=r.get("rank", i),
                video_id=r["video_id"],
                frame_id=fid,
                score=r.get("score", 0.0),
                formatted=f"{r['video_id']}, {fid}",
                image_url=f"/frames/{r['video_id']}/{idx}"
            ))

        return SearchResponse(
            query_type="textual_kis",
            num_results=len(answers),
            answers=answers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/qa", response_model=SearchResponse)
def search_qa(req: QARequest):
    """
    **Q&A (VQA)** — Tìm kiếm và trả lời câu hỏi trực quan.
    
    Trả về top-100 bộ (video_id, frame_id, answer).
    """
    try:
        searcher = get_qa_searcher()
        results = searcher.search(
            query=req.retrieval_query,
            question=req.question,
            use_vqa=req.use_vqa
        )

        answers = []
        for r in results[:req.top_k]:
            idx = r["frame_id"]
            fid = frame_id_from_index(r["video_id"], idx)
            answers.append(AnswerItem(
                rank=r["rank"],
                video_id=r["video_id"],
                frame_id=fid,
                answer=r.get("answer", ""),
                score=r.get("score", 0.0),
                formatted=f"{r['video_id']}, {fid}, {r.get('answer', '')}",
                image_url=f"/frames/{r['video_id']}/{idx}"
            ))

        return SearchResponse(
            query_type="qa",
            num_results=len(answers),
            answers=answers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/trake", response_model=SearchResponse)
def search_trake(req: TRAKERequest):
    """
    **TRAKE** — Tìm chuỗi sự kiện theo thứ tự thời gian.
    
    Trả về top-100 bộ (video_id, frame_id_1, ..., frame_id_N).
    """
    if len(req.events) == 0:
        raise HTTPException(status_code=400, detail="Cần ít nhất 1 sự kiện")
    if len(req.events) > 10:
        raise HTTPException(status_code=400, detail="Tối đa 10 sự kiện")

    try:
        searcher = get_trake_searcher()
        results = searcher.search(req.events)

        answers = []
        for r in results[:req.top_k]:
            fids = r.get("frame_ids", [])
            true_fids = frame_ids_from_indexes(r["video_id"], fids)
            # TRAKE: frame_ids = [before, current, after, ...]; middle là frame hiện tại
            cur = None
            if fids:
                mid = fids[len(fids) // 2]
                cur = mid if mid is not None else next((x for x in fids if x is not None), None)
            answers.append(AnswerItem(
                rank=r["rank"],
                video_id=r["video_id"],
                frame_ids=true_fids,
                score=r.get("total_score", 0.0),
                formatted=f"{r['video_id']}, " + ", ".join(str(f) for f in true_fids),
                image_url=(f"/frames/{r['video_id']}/{cur}" if cur is not None else None)
            ))

        return SearchResponse(
            query_type="trake",
            num_results=len(answers),
            answers=answers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/frames/{video_id}/{frame_id}", summary="Phục vụ ảnh keyframe")
def get_frame(video_id: str, frame_id: int):
    """
    Trả về ảnh keyframe cho (video_id, frame_id) — dùng trực tiếp làm `<img src>`.

    Ví dụ:
      GET /frames/L21_V001/1
      GET /frames/L21_V001.mp4/1
    """
    path = _resolve_frame_file(video_id, frame_id)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy frame: {video_id}, {frame_id}"
        )

    ext = os.path.splitext(path)[1].lower()
    media_type = (
        "image/jpeg" if ext in (".jpg", ".jpeg")
        else "image/png" if ext == ".png"
        else "application/octet-stream"
    )
    return FileResponse(
        path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/frames/batch", summary="Lấy nhiều ảnh keyframe (base64 data URL)")
def get_frames_batch(req: FrameBatchRequest):
    """
    Nhận danh sách (video_id, frame_id), trả về từng ảnh dưới dạng data URL.

    Body:
      {"frames": [{"video_id": "L21_V001", "frame_id": 1}, ...]}
    """
    items = []
    for f in req.frames:
        path = _resolve_frame_file(f.video_id, f.frame_id)
        if path is None:
            items.append(FrameBatchItem(
                video_id=f.video_id, frame_id=f.frame_id, found=False
            ))
            continue

        ext = os.path.splitext(path)[1].lower()
        media_type = (
            "image/jpeg" if ext in (".jpg", ".jpeg")
            else "image/png" if ext == ".png"
            else "application/octet-stream"
        )
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")

        items.append(FrameBatchItem(
            video_id=f.video_id,
            frame_id=f.frame_id,
            found=True,
            data_url=f"data:{media_type};base64,{b64}",
        ))

    return {
        "num_requests": len(req.frames),
        "num_found": sum(1 for i in items if i.found),
        "frames": items,
    }


@app.post("/submit")
def submit_batch(queries: List[Dict[str, Any]]):
    """
    Batch submission: nhận danh sách queries, trả về toàn bộ formatted answers.
    
    Input format:
    [
      {"query_id": "q1", "query_type": "textual_kis", "query_text": "..."},
      {"query_id": "q2", "query_type": "qa", "retrieval_query": "...", "question": "..."},
      {"query_id": "q3", "query_type": "trake", "events": ["...", "..."]}
    ]
    """
    manager = SubmissionManager(output_dir=cfg["submission"]["output_dir"])
    all_submissions = []

    for q in queries:
        qtype = q.get("query_type", "textual_kis").lower()
        qid = q.get("query_id", "unknown")

        try:
            if qtype in ("textual_kis", "kis"):
                searcher = get_kis_searcher()
                results = searcher.search(q.get("query_text", ""))
                formatted = searcher.format_submission(results)
            elif qtype in ("qa", "vqa"):
                searcher = get_qa_searcher()
                results = searcher.search(
                    query=q.get("retrieval_query", q.get("query_text", "")),
                    question=q.get("question", q.get("query_text", "")),
                    use_vqa=q.get("use_vqa", True)
                )
                formatted = searcher.format_submission(results)
            elif qtype == "trake":
                searcher = get_trake_searcher()
                results = searcher.search(q.get("events", []))
                formatted = searcher.format_submission(results)
            else:
                formatted = []

            sub = manager.build_query_submission(
                {"query_id": qid, "query_type": qtype},
                results if "results" in dir() else []
            )
            all_submissions.append(sub)

        except Exception as e:
            all_submissions.append({
                "query_id": qid,
                "query_type": qtype,
                "error": str(e),
                "answers": []
            })

    # Lưu file
    paths = manager.save_all(all_submissions)
    return {"submissions": all_submissions, "saved_to": paths}