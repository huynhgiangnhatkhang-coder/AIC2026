import type {
  AnswerItem,
  BackendInfo,
  FrameResult,
  HealthResponse,
  KISSearchRequest,
  KISSearchResponse,
  QASearchRequest,
  SearchResponse,
  TemporalEventResult,
  TemporalSearchResponse,
  TemporalVideo,
  TrakeSearchRequest,
} from "./types";

import { mockKisResponse, mockQaResponse, mockTrakeResponse } from "../mocks/mockSearch";

/**
 * Error taxonomy for the Lifelog Search frontend.
 * - NetworkError: backend unreachable / connection refused
 * - TimeoutError: request exceeded the deadline
 * - HttpError: backend answered with a non-2xx status
 * - ParseError: response body was not the JSON we expected
 */
export class ApiError extends Error {
  readonly kind: "network" | "timeout" | "http" | "parse" | "abort";
  readonly status: number | undefined;
  readonly detail: string | undefined;

  constructor(
    kind: ApiError["kind"],
    message: string,
    opts: { status?: number; detail?: string | undefined } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = opts.status;
    this.detail = opts.detail;
  }
}

export function isApiError(e: unknown): e is ApiError {
  return e instanceof ApiError;
}

const DEFAULT_TIMEOUT_MS = 600_000;

function resolveBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL;
  if (typeof configured === "string" && configured.trim().length > 0) {
    return configured.trim().replace(/\/+$/, "");
  }
  // No env var → rely on the Vite dev proxy (/health, /info, /search, /submit → :8000).
  return "";
}

const BASE_URL = resolveBaseUrl();

function endpoint(path: string): string {
  return BASE_URL ? `${BASE_URL}${path}` : path;
}

/** Turn a relative backend path (e.g. /frames/L21_V001/1) into an absolute URL. */
function absolutePath(path: string): string {
  if (/^[a-z][a-z\d+\-.]*:\/\//i.test(path)) return path;
  return `${BASE_URL}${path}`;
}

interface RequestOptions {
  signal?: AbortSignal | undefined;
  timeoutMs?: number;
}

interface ApiRequestError {
  detail?: unknown;
  [key: string]: unknown;
}

/** Normalize a FastAPI `detail` payload (string OR array of error objects). */
function describeDetail(detail: unknown): string | undefined {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = detail.flatMap((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object" && "msg" in item) {
        const msg = (item as { msg?: unknown }).msg;
        return typeof msg === "string" ? msg : undefined;
      }
      return [];
    });
    return msgs.length > 0 ? msgs.join("; ") : undefined;
  }
  return undefined;
}

async function parseDetail(resp: Response): Promise<ApiRequestError> {
  try {
    const body = (await resp.json()) as unknown;
    if (body && typeof body === "object") return body as ApiRequestError;
  } catch {
    /* not JSON — ignore */
  }
  return {};
}

async function request<T>(path: string, init: RequestInit, opts?: RequestOptions): Promise<T> {
  const outer = new AbortController();
  const internalTime = opts?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timer = setTimeout(
    () => outer.abort(new DOMException("Request timed out", "TimeoutError")),
    internalTime,
  );

  let abortedEarly = false;
  const onOuterAbort = () => {
    outer.abort();
  };
  if (opts?.signal) {
    if (opts.signal.aborted) {
      abortedEarly = true;
      outer.abort();
    } else {
      opts.signal.addEventListener("abort", onOuterAbort, { once: true });
    }
  }

  let resp: Response;
  try {
    if (abortedEarly) outer.abort();
    resp = await fetch(endpoint(path), {
      ...init,
      signal: outer.signal,
      headers: { Accept: "application/json", ...(init.headers ?? {}) },
    });
  } catch (err) {
    if (opts?.signal?.aborted) {
      throw new ApiError("abort", "Request was cancelled");
    }
    if (err instanceof DOMException && err.name === "TimeoutError") {
      throw new ApiError("timeout", `The request timed out after ${internalTime / 1000}s`);
    }
    throw new ApiError("network", "Cannot reach the backend. Is it running?", {
      detail: err instanceof Error ? err.message : undefined,
    });
  } finally {
    clearTimeout(timer);
    opts?.signal?.removeEventListener("abort", onOuterAbort);
  }

  if (!resp.ok) {
    const body = await parseDetail(resp);
    const detail = describeDetail(body.detail);
    throw new ApiError("http", detail ?? `Backend error (HTTP ${resp.status})`, {
      status: resp.status,
      detail,
    });
  }

  try {
    return (await resp.json()) as T;
  } catch {
    throw new ApiError("parse", "The backend returned an unreadable response body");
  }
}

/* ---------- response mappers (AIC2026 AnswerItem → display models) ---------- */

function mapAnswersToFrames(answers: AnswerItem[]): FrameResult[] {
  return answers.map((a, i) => {
    const frameKey = a.frame_id ?? a.frame_ids?.[0] ?? i + 1;
    const frameId = a.frame_id ?? a.frame_ids?.[0] ?? null;
    return {
      rank: a.rank,
      id: `${a.video_id}:${frameKey}`,
      score: a.score,
      frame_name: frameId !== null && frameId !== undefined ? `${a.video_id}/${frameId}.jpg` : null,
      video_name: a.video_id,
      timestamp_ms: null,
      frame_url: a.image_url ? absolutePath(a.image_url) : null,
      video_url: null,
      fps: null,
      snippet: a.answer ?? a.formatted,
      frame_id: frameId,
      answer: a.answer,
    };
  });
}

function mapAnswersToVideos(answers: AnswerItem[]): TemporalVideo[] {
  return answers.map((a) => {
    const fids = a.frame_ids ?? (a.frame_id !== null && a.frame_id !== undefined ? [a.frame_id] : []);
    const events: TemporalEventResult[] = fids.map((fid, i) => ({
      event_index: i,
      id: `${a.video_id}:${fid}`,
      timestamp_ms: 0,
      frame_name: fid != null ? String(fid) : null,
      frame_url: fid != null ? absolutePath(`/frames/${a.video_id}/${fid}`) : null,
      video_url: null,
      fps: null,
    }));
    return {
      video_name: a.video_id,
      best_sequence: { total_score: a.score, events },
      frame_ids: fids,
    };
  });
}

/* ---------- API client ---------- */

export interface SearchKisParams {
  query: string;
  top_k?: number | undefined;
  object_hints?: string[] | null | undefined;
  search_mode?: "hybrid" | "text" | "visual" | undefined;
  signal?: AbortSignal | undefined;
}

export interface SearchQaParams {
  retrieval_query: string;
  question: string;
  use_vqa?: boolean | undefined;
  top_k?: number | undefined;
  signal?: AbortSignal | undefined;
}

export interface SearchTrakeParams {
  events: string[];
  top_k?: number | undefined;
  signal?: AbortSignal | undefined;
}

export const api = {
  baseUrl: BASE_URL,

  health: (signal?: AbortSignal) =>
    request<HealthResponse>("/health", { signal: signal ?? null }, { signal }),

  info: (signal?: AbortSignal) =>
    request<BackendInfo>("/info", { signal: signal ?? null }, { signal }),

  searchKis: (params: SearchKisParams, opts?: RequestOptions) => {
    const body: KISSearchRequest = {
      query: params.query,
      object_hints: params.object_hints ?? null,
      top_k: params.top_k ?? 100,
      search_mode: params.search_mode ?? "hybrid",
    };
    return request<SearchResponse>(
      "/search/kis",
      {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Content-Type": "application/json" },
      },
      { ...opts, signal: opts?.signal ?? params.signal },
    ).then((res): KISSearchResponse => ({
      query: params.query,
      top_k: body.top_k,
      count: res.num_results,
      results: mapAnswersToFrames(res.answers),
    }));
  },

  searchQa: (params: SearchQaParams, opts?: RequestOptions) => {
    const body: QASearchRequest = {
      retrieval_query: params.retrieval_query,
      question: params.question,
      use_vqa: params.use_vqa ?? true,
      top_k: params.top_k ?? 50,
    };
    return request<SearchResponse>(
      "/search/qa",
      {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Content-Type": "application/json" },
      },
      { ...opts, signal: opts?.signal ?? params.signal },
    ).then((res): KISSearchResponse => ({
      query: params.retrieval_query,
      top_k: body.top_k,
      count: res.num_results,
      results: mapAnswersToFrames(res.answers),
    }));
  },

  searchTrake: (params: SearchTrakeParams, opts?: RequestOptions) => {
    const body: TrakeSearchRequest = {
      events: params.events,
      top_k: params.top_k ?? 50,
    };
    return request<SearchResponse>(
      "/search/trake",
      {
        method: "POST",
        body: JSON.stringify(body),
        headers: { "Content-Type": "application/json" },
      },
      { ...opts, signal: opts?.signal ?? params.signal },
    ).then((res): TemporalSearchResponse => ({
      top_k: body.top_k,
      videos: mapAnswersToVideos(res.answers),
    }));
  },
};

/** Rewrite host/port of a frame_url when VITE_MEDIA_BASE_URL is set. */
export function resolveMediaUrl(frameUrl: string | null | undefined): string | null {
  if (!frameUrl) return null;
  const override = import.meta.env.VITE_MEDIA_BASE_URL;
  if (typeof override !== "string" || override.trim().length === 0) return frameUrl;
  try {
    const target = new URL(frameUrl);
    const replaced = new URL(override.trim());
    target.protocol = replaced.protocol;
    target.host = replaced.host;
    return target.toString();
  } catch {
    return frameUrl;
  }
}

/* ---------- offline fallback (backend needs GPU, may be unavailable) ---------- */

export interface FallbackResult<T> {
  data: T;
  /** True when the backend was unreachable and offline demo data was served. */
  isMock: boolean;
}

/**
 * Runs a backend request; on a failure caused by the GPU backend being
 * unavailable (connection refused / timeout, or a 5xx from the dev proxy when
 * the backend is down) it transparently returns demo data shaped exactly like
 * the real response, so callers can keep using it without code changes.
 */
export async function withFallback<T>(
  fetcher: () => Promise<T>,
  fallback: () => T,
): Promise<FallbackResult<T>> {
  try {
    const data = await fetcher();
    return { data, isMock: false };
  } catch (err) {
    const unavailable =
      (isApiError(err) && (err.kind === "network" || err.kind === "timeout")) ||
      (isApiError(err) && err.kind === "http" && (err.status ?? 0) >= 500);
    if (unavailable) {
      return { data: fallback(), isMock: true };
    }
    throw err;
  }
}

export const apiWithFallback = {
  searchKis: (params: SearchKisParams, opts?: RequestOptions) =>
    withFallback(() => api.searchKis(params, opts), () =>
      mockKisResponse(params.query, params.top_k ?? 100),
    ),

  searchQa: (params: SearchQaParams, opts?: RequestOptions) =>
    withFallback(() => api.searchQa(params, opts), () =>
      mockQaResponse(params.retrieval_query, params.question, params.top_k ?? 50),
    ),

  searchTrake: (params: SearchTrakeParams, opts?: RequestOptions) =>
    withFallback(() => api.searchTrake(params, opts), () =>
      mockTrakeResponse(params.events, params.top_k ?? 50),
    ),
};
