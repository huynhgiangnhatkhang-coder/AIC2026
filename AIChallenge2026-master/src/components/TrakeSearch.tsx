import { useState } from "react";

import type { TemporalSearchResponse, TemporalVideo } from "../api/types";
import {
  DEFAULT_TRAKE_TOP_K,
  TRAKE_TOP_K_MAX,
  TRAKE_TOP_K_MIN,
} from "../lib/constants";
import { formatScore, formatTimestampMs } from "../lib/format";
import { useTrakeSearch } from "../hooks/useTrakeSearch";
import { ErrorBanner } from "./ErrorBanner";
import { EmptyState } from "./EmptyState";
import { FrameImage } from "./FrameImage";
import { Spinner } from "./Spinner";
import styles from "./TrakeSearch.module.css";

function VideoRow({ video }: { video: TemporalVideo }) {
  const { best_sequence } = video;
  return (
    <article className={styles.videoCard}>
      <header className={styles.videoHeader}>
        <code className={styles.videoName}>{video.video_name}</code>
        <span className={styles.score} title="Sequence similarity score">
          {formatScore(best_sequence.total_score)}
        </span>
      </header>
      <div className={styles.sequence} aria-label="Best matched event sequence">
        {best_sequence.events.map((ev, i) => (
          <div className={styles.event} key={`${video.video_name}-${ev.id}-${i}`}>
            <span className={styles.eventTag}>
              <span className={styles.eventIndex}>{i + 1}</span>
              <time>{formatTimestampMs(ev.timestamp_ms)}</time>
            </span>
            <div className={styles.eventThumb}>
              <FrameImage
                src={ev.frame_url}
                alt={`Event ${i + 1} frame from ${video.video_name}`}
                fallback={`${video.video_name} / ${ev.frame_name ?? ev.id}`}
              />
            </div>
            <code className={styles.eventId}>{ev.frame_name ?? ev.id}</code>
          </div>
        ))}
        {best_sequence.events.length < 2 ? (
          <span className={styles.shortNote}>only 1 event in sequence</span>
        ) : null}
      </div>
    </article>
  );
}

function TrakeResults({ data }: { data: TemporalSearchResponse }) {
  if (data.videos.length === 0) {
    return (
      <EmptyState
        title="No videos matched the event sequence"
        hint="Try more generic event descriptions or a larger top-k."
      />
    );
  }
  return (
    <section aria-label="Temporal search results" className={styles.results}>
      <header className={styles.resultsHeader}>
        <h2 className={styles.resultsTitle}>
          {data.videos.length} video{data.videos.length > 1 ? "s" : ""} with the event sequence
        </h2>
        <span className={styles.resultsSub}>top-{data.top_k}</span>
      </header>
      <div className={styles.videos}>
        {data.videos.map((v) => (
          <VideoRow key={v.video_name} video={v} />
        ))}
      </div>
    </section>
  );
}

export function TrakeSearch() {
  const { state, search } = useTrakeSearch();
  const [events, setEvents] = useState<string[]>(["", "", ""]);
  const [topk, setTopk] = useState(DEFAULT_TRAKE_TOP_K);

  const loading = state.status === "loading";
  const firstValid = (events[1]?.trim().length ?? 0) > 0;

  const setEvent = (i: number, value: string) =>
    setEvents((prev) => prev.map((e, idx) => (idx === i ? value : e)));

  const submit = () => {
    if (!firstValid || loading) return;
    let searchString = events.map((e) => e.trim());
    console.log(searchString);

    search({
      events: searchString,
      topK: topk,
    });
  };

  const tagMap: { [id: number]: string } = {
    0: "Before",
    1: "Current",
    2: "After",
  };
  return (
    <section className={styles.section}>
      <div className={styles.leftCol}>
        <div className={`panel ${styles.panel}`}>
          <p className={styles.intro}>
            Describe a <strong>sequence of events</strong> (in chronological order) that should happen
            in one video — e.g. <em>“a person enters a building”</em> then{" "}
            <em>“they exit with a bag”</em>. Videos containing the whole sequence are ranked first.
          </p>

          <div className={styles.events}>
            {events.map((ev, i) => (
              <div className={styles.eventInput} key={i}>
                <span className={styles.eventNum} aria-hidden="true">
                  {i + 1}
                </span>
                <input
                  className="input"
                  type="text"
                  value={ev}
                  maxLength={500}
                  placeholder={tagMap[i]}
                  aria-label={`Event ${i + 1} description`}
                  disabled={loading}
                  onChange={(e) => setEvent(i, e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      submit();
                    }
                  }}
                />
              </div>
            ))}
          </div>

          <div className={styles.controls}>
            <div className={styles.opt}>
              <label className="field-label" htmlFor="trake-topk">
                Max results <span className={styles.optValue}>{topk}</span>
              </label>
              <input
                id="trake-topk"
                type="range"
                min={TRAKE_TOP_K_MIN}
                max={TRAKE_TOP_K_MAX}
                step={1}
                value={topk}
                disabled={loading}
                onChange={(e) => setTopk(Number(e.target.value))}
              />
            </div>

            <button
              type="button"
              className="btn btn--primary"
              onClick={submit}
              disabled={!firstValid || loading}
            >
              {loading ? "Searching…" : "Search sequence"}
            </button>
          </div>
        </div>
      </div>

      <div className={styles.rightCol}>
        {state.status === "error" ? (
          <ErrorBanner title="Temporal search failed" detail={state.errorDetail ?? state.error} />
        ) : null}
        {state.status === "loading" ? (
          <Spinner label="Finding videos with this event sequence…" />
        ) : null}
        {state.status === "idle" ? (
          <EmptyState
            title="No search yet"
            hint="Enter at least one event description to find videos containing the sequence."
          />
        ) : null}
        {state.status === "success" && state.data ? <TrakeResults data={state.data} /> : null}
      </div>
    </section>
  );
}
