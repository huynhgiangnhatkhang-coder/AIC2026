/**
 * Type definitions mirroring the AIC2026 backend schemas (app/main.py).
 * Keep these in sync with the FastAPI models.
 */

/* ---------- raw AIC2026 API DTOs ---------- */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface BackendInfo {
  clip_model: string;
  clip_device: string;
  faiss_index: string;
  faiss_exists: boolean;
  bm25_corpus: string;
  bm25_exists: boolean;
  vqa_model: string;
}

export interface KISSearchRequest {
  query: string;
  object_hints: string[] | null;
  top_k: number;
  search_mode?: "hybrid" | "text" | "visual";
}

export interface QASearchRequest {
  retrieval_query: string;
  question: string;
  use_vqa: boolean;
  top_k: number;
}

export interface TrakeSearchRequest {
  events: string[];
  top_k: number;
}

export interface AnswerItem {
  rank: number;
  video_id: string;
  frame_id: number | null;
  frame_ids: number[] | null;
  answer: string | null;
  score: number;
  formatted: string;
  image_url: string | null;
}

export interface SearchResponse {
  query_type: string;
  num_results: number;
  answers: AnswerItem[];
}

/* ---------- display models ---------- */

export interface FrameResult {
  rank: number;
  id: string;
  score: number;
  frame_name: string | null;
  video_name: string | null;
  timestamp_ms: number | null;
  frame_url: string | null;
  video_url: string | null;
  fps: number | null;
  snippet: string | null;
  /** Raw keyframe id — needed to build `video_id, frame_id` submission lines. */
  frame_id: number | null;
  /** VQA answer text (Q&A mode) or null for plain KIS. */
  answer: string | null;
}

export interface KISSearchResponse {
  query: string;
  top_k: number;
  count: number;
  results: FrameResult[];
}

export interface TemporalEventResult {
  event_index: number;
  id: string;
  timestamp_ms: number;
  frame_name: string | null;
  frame_url: string | null;
  video_url: string | null;
  fps: number | null;
}

export interface TemporalSequence {
  total_score: number;
  events: TemporalEventResult[];
}

export interface TemporalVideo {
  video_name: string;
  best_sequence: TemporalSequence;
  /** Ordered keyframe ids of the matched event sequence — TRAKE submission line. */
  frame_ids: number[];
}

export interface TemporalSearchResponse {
  top_k: number;
  videos: TemporalVideo[];
}
