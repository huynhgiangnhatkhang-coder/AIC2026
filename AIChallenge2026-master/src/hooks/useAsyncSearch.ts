import { useCallback, useRef, useState } from "react";

import { ApiError, isApiError } from "../api/client";
import type { FallbackResult } from "../api/client";

export type AsyncSearchStatus = "idle" | "loading" | "success" | "error";

export interface AsyncSearchState<T> {
  status: AsyncSearchStatus;
  data: T | null;
  error: string | null;
  errorDetail: string | null;
  /** True when the result came from the offline demo fallback (backend down). */
  isMock: boolean;
  /** Monotonically increasing run id — guards against stale results. */
  requestId: number;
}

const IDLE: AsyncSearchState<never> = {
  status: "idle",
  data: null,
  error: null,
  errorDetail: null,
  isMock: false,
  requestId: 0,
};

/**
 * Runs an async fetch as a "search action": cancels the previous in-flight
 * request, tracks loading/error/success, and only ever commits the result of
 * the most recent invocation (stale-result protection).
 */
export function useAsyncSearch<T>() {
  const [state, setState] = useState<AsyncSearchState<T>>(IDLE);
  const controllerRef = useRef<AbortController | null>(null);
  const idRef = useRef(0);

  const run = useCallback((fetcher: (signal: AbortSignal) => Promise<FallbackResult<T>>) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const id = ++idRef.current;

    setState((prev) => ({ ...prev, status: "loading", error: null, errorDetail: null }));

    void (async () => {
      try {
        const res = await fetcher(controller.signal);
        if (id === idRef.current)
          setState({ status: "success", data: res.data, isMock: res.isMock, error: null, errorDetail: null, requestId: id });
      } catch (err) {
        if (id !== idRef.current) return;
        if (isApiError(err) && err.kind === "abort") return;
        if (err instanceof ApiError) {
          setState({
            status: "error",
            data: null,
            isMock: false,
            error: err.message,
            errorDetail: err.detail ?? null,
            requestId: id,
          });
        } else {
          setState({
            status: "error",
            data: null,
            isMock: false,
            error: err instanceof Error ? err.message : "Unknown error",
            errorDetail: null,
            requestId: id,
          });
        }
      }
    })();
  }, []);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    idRef.current += 1;
    setState({ status: "idle", data: null, isMock: false, error: null, errorDetail: null, requestId: idRef.current });
  }, []);

  return { state, run, reset };
}