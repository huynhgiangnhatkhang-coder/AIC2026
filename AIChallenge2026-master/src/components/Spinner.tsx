export function Spinner({ label }: { label?: string }) {
  return (
    <div className="spinner-holder" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      {label ? <span>{label}</span> : <span className="sr-only">Loading…</span>}
    </div>
  );
}