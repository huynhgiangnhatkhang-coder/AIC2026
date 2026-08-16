import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// Dev proxy so the UI works even without a VITE_API_BASE_URL:
// AIC2026 backend endpoints (/health, /info, /search/*, /submit) are
// forwarded to the FastAPI backend on :8000.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_PROXY_TARGET || "http://localhost:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/health": { target, changeOrigin: true, timeout: 60_000 },
        "/info": { target, changeOrigin: true, timeout: 60_000 },
        "/search": { target, changeOrigin: true, timeout: 60_000 },
        "/frames": { target, changeOrigin: true, timeout: 60_000 },
        "/submit": { target, changeOrigin: true, timeout: 60_000 },
      },
    },
  };
});