import { useState } from "react";

import { useHealth } from "./hooks/useHealth";
import { SelectionProvider } from "./context/SelectionContext";
import { StatusPill } from "./components/StatusPill";
import { SubmitPanel } from "./components/SubmitPanel";
import { TrakeSearch } from "./components/TrakeSearch";
import { KISSearch } from "./components/KISSearch";
import { QASearch } from "./components/QASearch";
import styles from "./App.module.css";

type Tab = "search" | "trake" | "qa";

const TABS: { id: Tab; label: string }[] = [
  { id: "search", label: "KIS search" },
  { id: "trake", label: "Temporal search" },
  { id: "qa", label: "Q&A" },
];

export default function App() {
  const health = useHealth();
  const [tab, setTab] = useState<Tab>("search");

  return (
    <SelectionProvider>
      <div className={styles.app}>
        <header className={styles.header}>
          <div className={styles.brand}>
            <h1 className={styles.title}>HCM AI Challenge</h1>
            <span className={styles.subtitle}>
              AIC 2026 · text-to-video retrieval over lifelog keyframes
            </span>
          </div>
          <StatusPill health={health.state} />
        </header>
        <div className={styles.body}>
          <aside className={styles.tabRail}>
            <nav className={styles.tabs} aria-label="Search modes">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  className={`${styles.tab} ${tab === t.id ? styles.tabActive : ""}`}
                  onClick={() => setTab(t.id)}
                  aria-current={tab === t.id ? "page" : undefined}
                >
                  {t.label}
                </button>
              ))}
            </nav>
          </aside>
          <main className={styles.main}>
            {tab === "search" ? <KISSearch /> : null}
            {tab === "trake" ? <TrakeSearch /> : null}
            {tab === "qa" ? <QASearch /> : null}
          </main>
        </div>
        <footer className={styles.footer}>
          <span>Text-input only · OCR/ASR retrieval planned</span>
          <span>
            {health.state.status === "ok" && health.state.info
              ? "backend reachable"
              : "backend unreachable"}
          </span>
        </footer>
        <SubmitPanel />
      </div>
    </SelectionProvider>
  );
}
