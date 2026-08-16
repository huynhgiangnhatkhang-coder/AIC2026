import { api } from "../api/client";
import type { TemporalSearchResponse } from "../api/types";
import { useAsyncSearch } from "./useAsyncSearch";

export interface TrakeRunParams {
  events: string[];
  topK: number;
}

export function useTrakeSearch() {
  const { state, run, reset } = useAsyncSearch<TemporalSearchResponse>();
  const search = (params: TrakeRunParams) => {
    void run((signal) =>
      api.searchTrake({
        events: params.events,
        top_k: params.topK,
        signal,
      }),
    );
  };
  return { state, search, reset };
}
