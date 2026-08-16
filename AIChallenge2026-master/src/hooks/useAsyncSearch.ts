import { useCallback, useRef, useState } from "react";

import { ApiError, isApiError } from "../api/client";

export type AsyncSearchStatus = "idle" | "loading" | "success" | "error";

export interface AsyncSearchState<T> {
  status: AsyncSearchStatus;
  data: T | null;
  error: string | null;
  errorDetail: string | null;
  /** Monotonically increasing run id — guards against stale results. */
  requestId: number;
}

const IDLE: AsyncSearchState<never> = {
  status: "idle",
  data: null,
  error: null,
  errorDetail: null,
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

  const run = useCallback((fetcher: (signal: AbortSignal) => Promise<T>) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const id = ++idRef.current;

    setState((prev) => ({ ...prev, status: "loading", error: null, errorDetail: null }));

    void (async () => {
      try {
        const data = await fetcher(controller.signal);
        if (id === idRef.current) setState({ status: "success", data, error: null, errorDetail: null, requestId: id });
      } catch (err) {
        if (id !== idRef.current) return;
        if (isApiError(err) && err.kind === "abort") return;
        if (err instanceof ApiError) {
          setState({
            status: "error",
            data: null,
            error: err.message,
            errorDetail: err.detail ?? null,
            requestId: id,
          });
        } else {
          setState({
            status: "error",
            data: null,
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
    setState({ status: "idle", data: null, error: null, errorDetail: null, requestId: idRef.current });
  }, []);

  return { state, run, reset };
}