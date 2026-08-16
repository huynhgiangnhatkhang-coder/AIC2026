import { useState } from "react";

import { DEFAULT_TOP_K, QUERY_MAX_LENGTH, TOP_K_MAX, TOP_K_MIN } from "../lib/constants";
import styles from "./SearchBar.module.css";

export interface SearchParams {
  query: string;
  topK: number;
}

interface SearchBarProps {
  loading: boolean;
  onSearch: (params: SearchParams) => void;
}

export function SearchBar({ loading, onSearch }: SearchBarProps) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(DEFAULT_TOP_K);

  const trimmed = query.trim();
  const valid = trimmed.length >= 1 && trimmed.length <= QUERY_MAX_LENGTH;

  const submit = () => {
    if (!valid || loading) return;
    onSearch({ query: trimmed, topK });
  };

  return (
    <form
      className={styles.form}
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <div className={styles.queryWrap}>
        <label className="field-label" htmlFor="text-query">
          Describe what you are looking for
        </label>
        <textarea
          id="text-query"
          className={styles.query}
          value={query}
          maxLength={QUERY_MAX_LENGTH}
          rows={2}
          placeholder="e.g. a person crossing the street near a red car"
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        <div className={styles.queryFoot}>
          <span className={styles.validate}>enter to search · 1–{QUERY_MAX_LENGTH} characters</span>
          <span
            className={query.length > QUERY_MAX_LENGTH * 0.9 ? styles.countWarn : styles.countOk}
          >
            {query.length}/{QUERY_MAX_LENGTH}
          </span>
        </div>
      </div>

      <div className={styles.row}>
        <div className={styles.options}>
          <div className={styles.opt}>
            <label className="field-label" htmlFor="text-topk">
              Top‑K <span className={styles.optValue}>{topK}</span>
            </label>
            <input
              id="text-topk"
              type="range"
              min={TOP_K_MIN}
              max={TOP_K_MAX}
              step={1}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </div>
        </div>

        <div className={styles.actions}>
          <button type="submit" className="btn btn--primary" disabled={!valid || loading}>
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
      </div>
    </form>
  );
}
