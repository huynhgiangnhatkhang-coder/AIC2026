import type { SelectedItem } from "../context/SelectionContext";

export const MAX_ANSWERS_PER_QUERY = 100;

export type SubmissionFormat = "json" | "csv" | "txt";

export interface SubmissionEntry {
  query_id: string;
  query_type: string;
  answers: string[];
}

export const FORMAT_EXTENSION: Record<SubmissionFormat, string> = {
  json: "json",
  csv: "csv",
  txt: "txt",
};

export const FORMAT_MIME: Record<SubmissionFormat, string> = {
  json: "application/json",
  csv: "text/csv",
  txt: "text/plain",
};

/**
 * Build the AIC2026 submission line for one selected result:
 *   textual_kis -> `video_id, frame_id`
 *   qa          -> `video_id, frame_id, answer`
 *   trake       -> `video_id, frame_id_1, ..., frame_id_N`
 */
export function formatAnswer(item: SelectedItem): string | null {
  const { videoId, queryType } = item;

  if (queryType === "qa") {
    if (item.frameId === null || item.frameId === undefined) return null;
    return `${videoId}, ${item.frameId}, ${item.answer ?? ""}`;
  }

  if (queryType === "trake") {
    const fids = item.frameIds.length > 0 ? item.frameIds : [item.frameId];
    if (fids.length === 0 || fids.some((f) => f === null || f === undefined)) return null;
    return `${videoId}, ${fids.join(", ")}`;
  }

  // textual_kis
  if (item.frameId === null || item.frameId === undefined) return null;
  return `${videoId}, ${item.frameId}`;
}

/**
 * Group selected items by (query_id, query_type) and produce the submission
 * entries, capped at MAX_ANSWERS_PER_QUERY answers each.
 */
export function buildSubmissions(selected: SelectedItem[]): SubmissionEntry[] {
  const groups = new Map<string, SubmissionEntry>();

  for (const item of selected) {
    const line = formatAnswer(item);
    if (!line) continue;
    const key = `${item.queryId}|${item.queryType}`;
    let entry = groups.get(key);
    if (!entry) {
      entry = { query_id: item.queryId, query_type: item.queryType, answers: [] };
      groups.set(key, entry);
    }
    if (entry.answers.length >= MAX_ANSWERS_PER_QUERY) continue;
    entry.answers.push(line);
  }

  return Array.from(groups.values());
}

export function serializeJson(submissions: SubmissionEntry[]): string {
  return `${JSON.stringify(submissions, null, 2)}\n`;
}

export function serializeCsv(submissions: SubmissionEntry[]): string {
  const lines: string[] = [];
  for (const sub of submissions) {
    for (const answer of sub.answers) {
      lines.push(answer);
    }
  }
  return `${lines.join("\n")}\n`;
}

export function serializeTxt(submissions: SubmissionEntry[]): string {
  const lines: string[] = [];
  for (const sub of submissions) {
    for (const answer of sub.answers) {
      lines.push(`${sub.query_id}\t${answer}`);
    }
  }
  return `${lines.join("\n")}\n`;
}

export function serializeSubmission(
  submissions: SubmissionEntry[],
  format: SubmissionFormat,
): string {
  switch (format) {
    case "json":
      return serializeJson(submissions);
    case "csv":
      return serializeCsv(submissions);
    case "txt":
      return serializeTxt(submissions);
  }
}

/** Trigger a browser "Save to storage" dialog for the generated file. */
export function downloadSubmissionFile(
  submissions: SubmissionEntry[],
  format: SubmissionFormat,
): void {
  const content = serializeSubmission(submissions, format);
  const ext = FORMAT_EXTENSION[format];
  const stamp = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
  const filename = `submission_${stamp}.${ext}`;
  const blob = new Blob([content], { type: FORMAT_MIME[format] });
  const url = URL.createObjectURL(blob);

  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
