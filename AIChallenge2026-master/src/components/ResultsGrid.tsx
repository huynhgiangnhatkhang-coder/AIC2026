import type { FrameResult } from "../api/types";
import { FrameCard } from "./FrameCard";
import styles from "./ResultsGrid.module.css";

interface ResultsGridProps {
  results: FrameResult[];
  total: number;
  queryLabel: string;
}

export function ResultsGrid({ results, total, queryLabel }: ResultsGridProps) {
  return (
    <section aria-label="Search results" className={styles.section}>
      <header className={styles.header}>
        <h2 className={styles.title}>Results for “{queryLabel}”</h2>
        <span className={styles.count}>
          {results.length} of {total} shown
        </span>
      </header>
      <div className={styles.grid}>
        {results.map((frame, i) => (
          <FrameCard key={frame.id} frame={frame} index={i} />
        ))}
      </div>
    </section>
  );
}