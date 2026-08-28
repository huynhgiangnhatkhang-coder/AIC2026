import styles from "./MockNotice.module.css";

interface MockNoticeProps {
  queryType: string;
}

/**
 * Banner shown when results came from the offline demo fallback because the
 * GPU backend is unreachable.
 */
export function MockNotice({ queryType }: MockNoticeProps) {
  return (
    <div className={styles.notice} role="status">
      <span className={styles.glyph} aria-hidden="true">
        ◌
      </span>
      <div>
        <strong>Offline demo data</strong> — the backend (GPU) is unreachable, so this{" "}
        {queryType} result is placeholder mock data with the same shape as real results.
      </div>
    </div>
  );
}
