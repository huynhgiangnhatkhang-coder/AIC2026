interface EmptyStateProps {
  title: string;
  hint?: string;
}

export function EmptyState({ title, hint }: EmptyStateProps) {
  return (
    <div className="empty-state" role="status">
      <div className="empty-state__glyph" aria-hidden="true">
        {}
      </div>
      <h3 className="empty-state__title">{title}</h3>
      {hint ? <p className="empty-state__hint">{hint}</p> : null}
    </div>
  );
}