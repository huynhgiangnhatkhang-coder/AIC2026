import type { ReactNode } from "react";

import { api } from "../api/client";
import type { HealthState } from "../hooks/useHealth";
import styles from "./StatusPill.module.css";

interface StatusPillProps {
  health: HealthState;
}

function Pill({ tone, title, children }: { tone: string; title: string; children: ReactNode }) {
  return (
    <div className={`${styles.pill} ${styles[tone]}`} role="status" title={title}>
      <span className={styles.dot} aria-hidden="true" />
      {children}
    </div>
  );
}

export function StatusPill({ health }: StatusPillProps) {
  if (health.status === "checking") {
    return (
      <Pill tone={"checking"} title="Checking backend connectivity…">
        <span>Backend …</span>
      </Pill>
    );
  }
  if (health.status === "error") {
    return (
      <Pill tone="error" title={health.message}>
        <span>Backend offline</span>
      </Pill>
    );
  }
  const { info } = health;
  return (
    <Pill tone="ok" title={`${info.service} v${info.version}`}>
      <span>Backend online</span>
      {api.baseUrl ? (
        <code className={styles.url}>{api.baseUrl.replace(/^https?:\/\//, "")}</code>
      ) : (
        <code className={styles.url}>dev proxy</code>
      )}
    </Pill>
  );
}
