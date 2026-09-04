import type { FrameResult } from "../api/types";
import { useSelection, type QueryType, type SelectedItem } from "../context/SelectionContext";
import { formatScore, formatTimestampMs } from "../lib/format";
import { FrameImage } from "./FrameImage";
import styles from "./FrameCard.module.css";

interface FrameCardProps {
  frame: FrameResult;
  index: number;
  queryId: string;
  queryType: QueryType;
}

const COLLAPSED_TAGS = 4;

export function FrameCard({ frame, index, queryId, queryType }: FrameCardProps) {
  const video = frame.video_name ?? "unknown";
  const displayId = frame.keyframe_id ?? frame.frame_id ?? "";
  const tags = [video, String(displayId)];
  if (frame.answer) tags.push(frame.answer);
  const scorePct = Math.round(((frame.score + 1) / 2) * 100);

  const { isSelected, toggle } = useSelection();

  const itemId = `${queryId}:${queryType}:${frame.video_name ?? "unknown"}:${frame.frame_id ?? index + 1}`;
  const selected = isSelected(itemId);

  const item: SelectedItem = {
    id: itemId,
    queryId,
    queryType,
    videoId: frame.video_name ?? "unknown",
    frameId: frame.frame_id,
    frameIds: [],
    answer: frame.answer,
    score: frame.score,
  };

  return (
    <article className={`${styles.card} ${selected ? styles.selected : ""}`} data-rank={frame.rank}>
      <div className={styles.thumb}>
        <FrameImage
          src={frame.frame_url}
          alt={`Frame ${formatTimestampMs(frame.timestamp_ms)} from ${video} (match #${frame.rank})`}
          fallback={`${video} / ${frame.frame_name ?? "frame"}`}
        />
        <span className={styles.rank} title={`Match rank ${frame.rank}`}>
          {index + 1}
        </span>
        <label className={styles.check} title={selected ? "Deselect this frame" : "Select this frame"}>
          <input
            type="checkbox"
            checked={selected}
            onChange={() => toggle(item)}
            aria-label={`Select ${video} frame ${frame.frame_id ?? ""}`}
          />
        </label>
      </div>

      <div className={styles.meta}>
        <div className={styles.metaRow}>
          <code className={styles.video} title={video}>
            {video}
          </code>
          <span className={styles.timestamp}>{formatTimestampMs(frame.timestamp_ms)}</span>
        </div>
        <div className={styles.scoreRow} title={`Similarity ${formatScore(frame.score)}`}>
          <span className={styles.scoreValue}>{formatScore(frame.score)}</span>
          <span className={styles.scoreTrack}>
            <span className={styles.scoreFill} style={{ width: `${scorePct}%` }} />
          </span>
        </div>
        {tags.length > 0 ? (
          <ul className={styles.tags}>
            {tags.slice(0, COLLAPSED_TAGS).map((tag, i) => (
              <li key={`${frame.id}-${tag}-${i}`}>{tag}</li>
            ))}
          </ul>
        ) : null}
        <div className={styles.frameId}>
          <code>{frame.frame_name ?? frame.id}</code>
          {frame.fps ? <span>{frame.fps} fps</span> : null}
        </div>
      </div>
    </article>
  );
}
