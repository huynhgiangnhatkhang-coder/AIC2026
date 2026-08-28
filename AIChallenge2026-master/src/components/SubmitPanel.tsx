import { useState } from "react";

import { useSelection } from "../context/SelectionContext";
import {
  buildSubmissions,
  downloadSubmissionFile,
  type SubmissionFormat,
} from "../lib/submission";
import styles from "./SubmitPanel.module.css";

const FORMAT_OPTIONS: { id: SubmissionFormat; label: string }[] = [
  { id: "txt", label: "TXT (video_id, frame_id …)" },
  { id: "csv", label: "CSV (query_id, rank, answer)" },
  { id: "json", label: "JSON (query_id → answers[])" },
];

/**
 * Floating submission bar: collects the selected results, serializes them in
 * the AIC2026 submission format and lets the user save the file to storage.
 */
export function SubmitPanel() {
  const { selected, count, clear } = useSelection();
  const [format, setFormat] = useState<SubmissionFormat>("txt");

  const submit = () => {
    if (count === 0) return;
    const submissions = buildSubmissions(selected);
    downloadSubmissionFile(submissions, format);
  };

  return (
    <div className={styles.panel} aria-label="Submission panel">
      <div className={styles.header}>
        <span className={styles.count}>
          <strong>{count}</strong> result{count === 1 ? "" : "s"} selected
        </span>
        <button
          type="button"
          className="btn btn--ghost"
          onClick={clear}
          disabled={count === 0}
          style={{ padding: "5px 10px", fontSize: "12.5px" }}
        >
          Clear all
        </button>
      </div>
      <div className={styles.row}>
        <div className={styles.field}>
          <label className="field-label" htmlFor="submit-format">
            Format
          </label>
          <select
            id="submit-format"
            className="select"
            value={format}
            onChange={(e) => setFormat(e.target.value as SubmissionFormat)}
          >
            {FORMAT_OPTIONS.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
        </div>
        <button
          type="button"
          className="btn btn--primary"
          onClick={submit}
          disabled={count === 0}
        >
          Submit &amp; save file
        </button>
      </div>
      {count > 0 ? (
        <div className={styles.groups}>
          {buildSubmissions(selected).map((g) => (
            <span key={`${g.query_id}|${g.query_type}`} className={styles.group}>
              {g.query_id} · {g.query_type} · {g.answers.length} answer{g.answers.length === 1 ? "" : "s"}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
