import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";

export type QueryType = "textual_kis" | "qa" | "trake";

export interface SelectedItem {
  /** Stable unique key used for checkbox state (also the "registered" id). */
  id: string;
  queryId: string;
  queryType: QueryType;
  videoId: string;
  /** Single keyframe id (KIS / Q&A). */
  frameId: number | null;
  /** Ordered keyframe ids (TRAKE sequence). */
  frameIds: number[];
  /** VQA answer (Q&A). */
  answer: string | null;
  score: number;
}

interface SelectionContextValue {
  selected: SelectedItem[];
  selectedIds: ReadonlySet<string>;
  count: number;
  isSelected: (id: string) => boolean;
  toggle: (item: SelectedItem) => void;
  clear: () => void;
}

const SelectionContext = createContext<SelectionContextValue | null>(null);

export function SelectionProvider({ children }: { children: ReactNode }) {
  const [selected, setSelected] = useState<SelectedItem[]>([]);

  const isSelected = useCallback(
    (id: string) => selected.some((item) => item.id === id),
    [selected],
  );

  const toggle = useCallback((item: SelectedItem) => {
    setSelected((prev) => {
      const exists = prev.some((existing) => existing.id === item.id);
      return exists ? prev.filter((existing) => existing.id !== item.id) : [...prev, item];
    });
  }, []);

  const clear = useCallback(() => setSelected([]), []);

  const selectedIds = useMemo(() => new Set(selected.map((s) => s.id)), [selected]);

  const value = useMemo<SelectionContextValue>(
    () => ({ selected, selectedIds, count: selected.length, isSelected, toggle, clear }),
    [selected, selectedIds, isSelected, toggle, clear],
  );

  return <SelectionContext.Provider value={value}>{children}</SelectionContext.Provider>;
}

export function useSelection(): SelectionContextValue {
  const ctx = useContext(SelectionContext);
  if (!ctx) throw new Error("useSelection must be used within a SelectionProvider");
  return ctx;
}
