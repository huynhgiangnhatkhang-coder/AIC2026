import type { ReactNode } from "react";

interface ErrorBannerProps {
  title: string;
  detail?: string | null;
  actionLabel?: string;
  onAction?: () => void;
  children?: ReactNode;
}

export function ErrorBanner({ title, detail, actionLabel, onAction, children }: ErrorBannerProps) {
  return (
    <div className="banner banner--error" role="alert">
      <div className="banner__body">
        <strong>{title}</strong>
        {detail ? <p className="banner__detail">{detail}</p> : null}
        {children ? <div className="banner__children">{children}</div> : null}
      </div>
      {actionLabel && onAction ? (
        <button type="button" className="btn btn--ghost banner__action" onClick={onAction}>
          {actionLabel}
        </button>
      ) : null}
    </div>
  );
}