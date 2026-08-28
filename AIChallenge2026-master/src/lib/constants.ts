export const DEFAULT_TOP_K = 25;
export const TOP_K_MIN = 1;
export const TOP_K_MAX = 100;

export const TRAKE_EVENTS_MAX = 3;
export const TRAKE_TOP_K_MIN = 1;
export const TRAKE_TOP_K_MAX = 100;
export const DEFAULT_TRAKE_TOP_K = 50;

export const QUERY_MIN_LENGTH = 1;
export const QUERY_MAX_LENGTH = 500;

/** Generate a default query id for the submission file, e.g. `kis_20260828_153012`. */
export function defaultQueryId(type: string): string {
  const stamp = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
  return `${type}_${stamp}`;
}
