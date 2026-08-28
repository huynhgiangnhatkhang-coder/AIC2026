import { useState } from "react";

import { DEFAULT_TOP_K, QUERY_MAX_LENGTH, TOP_K_MAX, TOP_K_MIN } from "../lib/constants";
import { defaultQueryId } from "../lib/constants";
import { useQASearch } from "../hooks/useQASearch";
import { EmptyState } from "./EmptyState";
import { ErrorBanner } from "./ErrorBanner";
import { MockNotice } from "./MockNotice";
import { QueryIdField } from "./QueryIdField";
import { ResultsGrid } from "./ResultsGrid";
import { Spinner } from "./Spinner";
import styles from "./QASearch.module.css";

export function QASearch() {
  const { state, search, reset } = useQASearch();
  const [retrievalQuery, setRetrievalQuery] = useState("");
  const [question, setQuestion] = useState("");
  const [useVqa, setUseVqa] = useState(true);
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [queryId, setQueryId] = useState(() => defaultQueryId("qa"));

  const loading = state.status === "loading";
  const valid = retrievalQuery.trim().length >= 1 && question.trim().length >= 1;

  const submit = () => {
    if (!valid || loading) return;
    search({
      retrievalQuery: retrievalQuery.trim(),
      question: question.trim(),
      useVqa,
      topK,
    });
  };

  return (
    <section className={styles.section}>
      <div className={styles.leftCol}>
        <div className={`panel ${styles.panel}`}>
          <p className={styles.intro}>
            Describe a <strong>scene</strong> to retrieve relevant frames, then ask a{" "}
            <strong>visual question</strong> about those frames — e.g.{" "}
            <em>“a person at a party”</em> and <em>“What color is her dress?”</em>
          </p>

          <div className={styles.field}>
            <label className="field-label" htmlFor="qa-retrieval">
              Scene description (used to retrieve frames)
            </label>
            <textarea
              id="qa-retrieval"
              className={styles.query}
              value={retrievalQuery}
              maxLength={QUERY_MAX_LENGTH}
              rows={2}
              placeholder="e.g. a person at a party"
              disabled={loading}
              onChange={(e) => setRetrievalQuery(e.target.value)}
            />
          </div>

          <div className={styles.field}>
            <label className="field-label" htmlFor="qa-question">
              Question
            </label>
            <input
              id="qa-question"
              className="input"
              type="text"
              value={question}
              maxLength={QUERY_MAX_LENGTH}
              placeholder="e.g. What color is her dress?"
              disabled={loading}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  submit();
                }
              }}
            />
          </div>

          <div className={styles.controls}>
            <label className={styles.checkLabel}>
              <input type="checkbox" checked={useVqa} disabled={loading} onChange={(e) => setUseVqa(e.target.checked)} />
              Use VQA model (BLIP-2)
            </label>

            <div className={styles.opt}>
              <label className="field-label" htmlFor="qa-topk">
                Top‑K <span className={styles.optValue}>{topK}</span>
              </label>
              <input
                id="qa-topk"
                type="range"
                min={TOP_K_MIN}
                max={TOP_K_MAX}
                step={1}
                value={topK}
                disabled={loading}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </div>

            <button type="button" className="btn btn--primary" onClick={submit} disabled={!valid || loading}>
              {loading ? "Searching…" : "Ask"}
            </button>
          </div>

          <QueryIdField value={queryId} type="qa" disabled={loading} onChange={setQueryId} />
        </div>
      </div>

      <div className={styles.rightCol}>
        {state.status === "error" ? (
          <ErrorBanner title="Q&A search failed" detail={state.errorDetail ?? state.error} actionLabel="Reset" onAction={reset} />
        ) : null}
        {state.status === "loading" ? <Spinner label="Retrieving frames and answering…" /> : null}
        {state.isMock && state.status === "success" ? <MockNotice queryType="Q&A" /> : null}
        {state.status === "idle" ? (
          <EmptyState title="No question asked yet" hint="Enter a scene description and a question to retrieve matching frames." />
        ) : null}
        {state.status === "success" && state.data ? (
          state.data.count === 0 ? (
            <EmptyState title={`No frames matched “${state.data.query}”`} hint="Try a different scene description or wording." />
          ) : (
            <ResultsGrid results={state.data.results} total={state.data.count} queryLabel={state.data.query} queryId={queryId} queryType="qa" />
          )
        ) : null}
      </div>
    </section>
  );
}
