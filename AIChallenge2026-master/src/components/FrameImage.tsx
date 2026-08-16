import { useState } from "react";

import { resolveMediaUrl } from "../api/client";
import styles from "./FrameImage.module.css";

interface FrameImageProps {
  src: string | null;
  alt: string;
  /** Optional small caption shown inside the placeholder when the image is missing. */
  fallback?: string;
  className?: string;
}

/**
 * Keyframe thumbnail that degrades gracefully: if the media server is down or
 * the frame_url is missing, a themed placeholder is rendered instead so the
 * layout never breaks.
 */
export function FrameImage({ src, alt, fallback, className }: FrameImageProps) {
  const resolved = resolveMediaUrl(src);
  const [failed, setFailed] = useState(false);

  if (!resolved || failed) {
    return (
      <div className={`${styles.placeholder} ${className ?? ""}`} role="img" aria-label={alt}>
        <span className={styles.placeholder__glyph} aria-hidden="true">
          ◻
        </span>
        {fallback ? <span className={styles.placeholder__text}>{fallback}</span> : null}
      </div>
    );
  }

  return (
    <img
      className={`${styles.image} ${className ?? ""}`}
      src={resolved}
      alt={alt}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}