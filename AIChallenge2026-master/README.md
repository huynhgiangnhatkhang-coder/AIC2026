# Lifelog Search AI — Frontend

Frontend for the AIC2026 baseline backend (`../AIC2026`): text-only retrieval over
AIC 2026 lifelog keyframes. Built with **React 19 + TypeScript + Vite**, plain CSS
modules, no UI framework dependencies.

## What it does

- **KIS search** — textual keyframe retrieval (`POST /search/kis`) with `top_k`
  control and optional comma-separated object hints.
- **Temporal search (TRAKE)** — ordered event-sequence retrieval (`POST /search/trake`):
  describe 1–3 moments in chronological order and get videos whose frames contain
  the whole sequence.
- **Q&A (VQA)** — scene retrieval + visual question answering (`POST /search/qa`),
  with an optional VQA-model toggle.
- **Live backend health** pill (`GET /health`).

Image search, OCR and ASR are **out of scope**: the challenge is text-input only.

## Quick start

```bash
# 1. Start the backend (from the AIC2026 folder)
cd AIC2026 && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 2. Start the frontend
cd frontend
npm install
cp .env.example .env   # optional — defaults work out of the box
npm run dev            # http://localhost:5173
```

The Vite dev server proxies `/health`, `/info`, `/search/*` and `/submit` to
`http://localhost:8000`, so the UI works with no env config. The backend allows
all origins in CORS.

## Configuration (`.env`)

| Variable              | Default                 | Purpose                                                             |
| --------------------- | ----------------------- | ------------------------------------------------------------------- |
| `VITE_API_BASE_URL`   | _(unset → dev proxy)_   | Direct backend URL, e.g. `http://localhost:8000`                    |
| `VITE_MEDIA_BASE_URL` | _(unset → use backend)_ | Rewrite keyframe thumbnails to another host (e.g. MinIO at `:9000`) |
| `VITE_PROXY_TARGET`   | `http://localhost:8000` | Dev-proxy target (ignored when `VITE_API_BASE_URL` is set)          |

## Scripts

```bash
npm run dev          # dev server on :5173
npm run build        # type-check (tsc) + production build
npm run typecheck    # TypeScript only
npm run preview      # serve the production build
```

## Robustness notes

- Typed API layer with an explicit error taxonomy: `network`, `timeout`
  (30 s, aborted), `http` (FastAPI `detail` normalized for both string and
  array forms), `parse`, `abort`.
- Every search aborts the previous in-flight request; results are only ever
  committed for the most recent run (stale-result protection).
- Health check polls every 15 s (5 s when the backend is down, no error spam).
- Broken/missing keyframe images render a themed placeholder instead of a
  broken `<img>`, so a down media server never breaks the layout.
- Input validation mirrors the backend: query 1–500 chars, `top_k` 1–100,
  temporal events 1–3.

## Project layout

```
src/
  api/        types.ts (AIC2026 schemas) + client.ts (fetch wrapper + mappers)
  hooks/      useHealth, useKISSearch, useTrakeSearch, useQASearch, useAsyncSearch
  components/ SearchBar, KISSearch, TrakeSearch, QASearch, ResultsGrid, FrameCard,
              FrameImage, StatusPill, plus shared UI primitives
  lib/        formatting helpers + constants
  styles/     global.css (design tokens)
```
