import { useEffect, useState } from "react";

import { api, isApiError } from "../api/client";
import type { HealthResponse } from "../api/types";

export type HealthState =
  | { status: "checking" }
  | { status: "ok"; info: HealthResponse }
  | { status: "error"; message: string };

const POLL_OK_MS = 15_000;
const POLL_ERROR_MS = 5_000;

export function useHealth() {
  const [state, setState] = useState<HealthState>({ status: "checking" });
  const [checks, setChecks] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let lastOk = false;
    let controller: AbortController | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      controller = new AbortController();
      try {
        const info = await api.health(controller.signal);
        if (!cancelled) {
          lastOk = true;
          setState({ status: "ok", info });
        }
      } catch (err) {
        console.log("Error");
        if (!cancelled && isApiError(err) && err.kind === "abort") {
          return;
        }
        if (!cancelled) {
          lastOk = false;
          setState({
            status: "error",
            message: isApiError(err) ? err.message : "Health check failed",
          });
        }
      } finally {
        if (!cancelled) {
          setChecks((c) => c + 1);
          const delay = lastOk ? POLL_OK_MS : POLL_ERROR_MS;
          timer = setTimeout(tick, delay);
        }
      }
    };

    void tick();
    return () => {
      cancelled = true;
      controller?.abort();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, []);

  return { state, checks };
}
