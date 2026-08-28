import { apiWithFallback } from "../api/client";
import type { KISSearchResponse } from "../api/types";
import { useAsyncSearch } from "./useAsyncSearch";

export interface KISSearchRunParams {
  query: string;
  topK: number;
  objectHints?: string[] | undefined;
  searchMode?: "hybrid" | "text" | "visual";
}

export function useKISSearch() {
  const { state, run, reset } = useAsyncSearch<KISSearchResponse>();
  const search = (params: KISSearchRunParams) => {
    void run((signal) =>
      apiWithFallback.searchKis({
        query: params.query,
        top_k: params.topK,
        object_hints: params.objectHints,
        search_mode: params.searchMode,
        signal,
      }),
    );
  };
  return { state, search, reset };
}
