import { api } from "../api/client";
import type { KISSearchResponse } from "../api/types";
import { useAsyncSearch } from "./useAsyncSearch";

export interface QASearchRunParams {
  retrievalQuery: string;
  question: string;
  useVqa: boolean;
  topK: number;
}

export function useQASearch() {
  const { state, run, reset } = useAsyncSearch<KISSearchResponse>();
  const search = (params: QASearchRunParams) => {
    void run((signal) =>
      api.searchQa({
        retrieval_query: params.retrievalQuery,
        question: params.question,
        use_vqa: params.useVqa,
        top_k: params.topK,
        signal,
      }),
    );
  };
  return { state, search, reset };
}
