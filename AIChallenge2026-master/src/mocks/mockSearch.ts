import type {
  FrameResult,
  HealthResponse,
  KISSearchResponse,
  TemporalEventResult,
  TemporalSearchResponse,
  TemporalVideo,
} from "../api/types";

/**
 * Offline demo data — served when the GPU-backed backend is unreachable.
 *
 * Every shape here mirrors the backend response models exactly
 * (src/api/types.ts + app/main.py), so mock results can be swapped in/out
 * with real backend results without touching the rest of the UI.
 */

/* ---------- deterministic PRNG ---------- */

function hashString(input: string): number {
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let s = seed;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- placeholder keyframe images ---------- */

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Generate an inline SVG data-URL "keyframe" so offline demo results render
 * as real thumbnails — the same role a backend-served /frames/... image plays.
 */
export function placeholderFrameUrl(videoId: string, frameId: number | null): string {
  const label = escapeXml(videoId);
  const sub = frameId !== null && frameId !== undefined ? escapeXml(`frame ${frameId}`) : "frame";
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="480" height="270" viewBox="0 0 480 270">` +
    `<rect width="480" height="270" fill="#0e1420"/>` +
    `<rect x="1" y="1" width="478" height="268" fill="none" stroke="#24303e" stroke-width="2"/>` +
    `<circle cx="240" cy="118" r="34" fill="none" stroke="#34d399" stroke-width="3"/>` +
    `<text x="240" y="120" font-family="monospace" font-size="30" fill="#7dd3fc" text-anchor="middle">▶</text>` +
    `<text x="240" y="186" font-family="monospace" font-size="24" fill="#e2e8f0" text-anchor="middle">${label}</text>` +
    `<text x="240" y="214" font-family="monospace" font-size="16" fill="#64748b" text-anchor="middle">${sub}</text>` +
    `<text x="240" y="252" font-family="monospace" font-size="12" fill="#33404f" text-anchor="middle">offline demo keyframe</text>` +
    `</svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

/* ---------- frame/video pools ---------- */

const VIDEO_POOL = Array.from({ length: 20 }, (_, i) => `L21_V${String(i + 1).padStart(3, "0")}`);

const TAG_POOL = [
  "person, laptop, meeting room",
  "red car, street, traffic light",
  "woman, party, dress",
  "athlete, stadium, running",
  "screenshot, news, headline",
  "child, playground, swing",
  "kitchen, cooking, stove",
  "beach, ocean, sunset",
  "office, desk, document",
  "train, platform, crowd",
  "bicycle, race, finish line",
  "concert, stage, lights",
];

const ANSWER_POOL = [
  "a red dress",
  "a laptop",
  "a white cup",
  "a green signboard",
  "a black backpack",
  "two people talking",
  "a blue car",
  "a yellow umbrella",
  "a glass of juice",
  "a smartphone",
];

function pick<T>(rng: () => number, pool: readonly T[]): T {
  const idx = Math.floor(rng() * pool.length) % pool.length;
  return pool[idx]!;
}

/* ---------- frame-level result builders (KIS / Q&A) ---------- */

function buildFrame(
  rng: () => number,
  rank: number,
  opts: { query: string; answer?: string | null; score?: number },
): FrameResult {
  const video = pick(rng, VIDEO_POOL);
  const frameId = 1 + Math.floor(rng() * 5000);
  const score = opts.score ?? Math.max(0.2, 0.95 - rank * 0.028 - rng() * 0.01);
  const answer = opts.answer ?? null;
  return {
    rank,
    id: `${video}:${frameId}`,
    score,
    frame_name: `${video}/${frameId}.jpg`,
    video_name: video,
    timestamp_ms: frameId * 1000,
    frame_url: placeholderFrameUrl(video, frameId),
    video_url: null,
    fps: null,
    snippet: answer ?? pick(rng, TAG_POOL),
    frame_id: frameId,
    answer,
  };
}

/* ---------- mock responses ---------- */

export function mockHealth(): HealthResponse {
  return { status: "ok", service: "AIC2026 Baseline API (offline demo)", version: "1.0.0-mock" };
}

export function mockKisResponse(query: string, topK: number): KISSearchResponse {
  const rng = mulberry32(hashString(`kis:${query}:${topK}`));
  const results: FrameResult[] = Array.from({ length: Math.min(25, topK) }, (_, i) =>
    buildFrame(rng, i + 1, { query }),
  );
  return { query, top_k: topK, count: results.length, results };
}

export function mockQaResponse(retrievalQuery: string, question: string, topK: number): KISSearchResponse {
  const rng = mulberry32(hashString(`qa:${retrievalQuery}:${question}:${topK}`));
  const results: FrameResult[] = Array.from({ length: Math.min(25, topK) }, (_, i) =>
    buildFrame(rng, i + 1, { query: retrievalQuery, answer: pick(rng, ANSWER_POOL) }),
  );
  return { query: retrievalQuery, top_k: topK, count: results.length, results };
}

export function mockTrakeResponse(events: string[], topK: number): TemporalSearchResponse {
  const rng = mulberry32(hashString(`trake:${events.join("|")}:${topK}`));
  const nVideos = Math.min(8, topK);
  const videos: TemporalVideo[] = Array.from({ length: nVideos }, (_, i) => {
    const video = pick(rng, VIDEO_POOL);
    const totalScore = Math.max(0.3, 0.92 - i * 0.06 - rng() * 0.02);
    const start = 50 + Math.floor(rng() * 3000);
    const frameIds = Array.from({ length: Math.max(2, events.length) }, (_, j) => start + j * 25 + Math.floor(rng() * 8));
    const evs: TemporalEventResult[] = frameIds.map((fid, j) => ({
      event_index: j,
      id: `${video}:${fid}`,
      timestamp_ms: fid * 1000,
      frame_name: String(fid),
      frame_url: placeholderFrameUrl(video, fid),
      video_url: null,
      fps: null,
    }));
    return { video_name: video, best_sequence: { total_score: totalScore, events: evs }, frame_ids: frameIds };
  });
  return { top_k: topK, videos };
}
