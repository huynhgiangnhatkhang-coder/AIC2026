import { useState } from "react";

import { useKISSearch } from "../hooks/useKISSearch";
import { defaultQueryId } from "../lib/constants";
import { EmptyState } from "./EmptyState";
import { ErrorBanner } from "./ErrorBanner";
import { MockNotice } from "./MockNotice";
import { QueryIdField } from "./QueryIdField";
import { ResultsGrid } from "./ResultsGrid";
import { SearchBar, type SearchParams } from "./SearchBar";
import { Spinner } from "./Spinner";
import styles from "./KISSearch.module.css";

export function KISSearch() {
  const { state, search, reset } = useKISSearch();
  const [objectHints, setObjectHints] = useState("");
  const [searchMode, setSearchMode] = useState<"hybrid" | "text" | "visual">("hybrid");
  const [queryId, setQueryId] = useState(() => defaultQueryId("kis"));
  const loading = state.status === "loading";

  const handleSearch = (params: SearchParams) => {
    const hints = objectHints
      .split(",")
      .map((h) => h.trim())
      .filter(Boolean);
    search({
      query: params.query,
      topK: params.topK,
      objectHints: hints.length > 0 ? hints : undefined,
      searchMode,
    });
  };

  return (
    <section className={styles.section}>
      <div className={styles.leftCol}>
        <div className={`panel ${styles.searchPanel}`}>
          <SearchBar loading={loading} onSearch={handleSearch} />
          <div className={styles.hints}>
            <label className="field-label" htmlFor="kis-search-mode">
              Search mode
            </label>
            <select
              id="kis-search-mode"
              className="input"
              value={searchMode}
              disabled={loading}
              onChange={(e) => setSearchMode(e.target.value as "hybrid" | "text" | "visual")}
              style={{ marginBottom: "1rem" }}
            >
              <option value="hybrid">Hybrid (CLIP + OCR + Florence)</option>
              <option value="text">Text Only (OCR)</option>
              <option value="visual">Visual Only (CLIP + Florence)</option>
            </select>

            <label className="field-label" htmlFor="kis-object-hints">
              Object hints (comma-separated, optional)
            </label>
            <input
              id="kis-object-hints"
              className="input"
              type="text"
              value={objectHints}
              placeholder="e.g. laptop, person"
              disabled={loading}
              onChange={(e) => setObjectHints(e.target.value)}
            />
          </div>
          <QueryIdField value={queryId} type="kis" disabled={loading} onChange={setQueryId} />
        </div>
      </div>

      <div className={styles.rightCol}>
        {state.status === "error" ? (
          <ErrorBanner
            title="Search failed"
            detail={state.errorDetail ?? state.error}
            actionLabel="Reset"
            onAction={reset}
          />
        ) : null}

        {state.status === "loading" ? <Spinner label="Searching keyframes…" /> : null}

        {state.isMock && state.status === "success" ? <MockNotice queryType="KIS" /> : null}

        {state.status === "idle" ? (
          <EmptyState
            title="Describe a moment to find matching keyframes"
            hint="Search is powered by a CLIP embedding model over the AIC 2026 lifelog keyframes."
          />
        ) : null}

        {state.status === "success" && state.data ? (
          state.data.count === 0 ? (
            <EmptyState
              title={`No frames matched “${state.data.query}”`}
              hint="Try different wording or increase top-K."
            />
          ) : (
            <ResultsGrid
              results={state.data.results}
              total={state.data.count}
              queryLabel={state.data.query}
              queryId={queryId}
              queryType="textual_kis"
            />
          )
        ) : null}
      </div>
    </section>
  );
}
