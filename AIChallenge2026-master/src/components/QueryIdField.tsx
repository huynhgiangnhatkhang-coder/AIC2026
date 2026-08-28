import { defaultQueryId } from "../lib/constants";
import styles from "./QueryIdField.module.css";

interface QueryIdFieldProps {
  value: string;
  type: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

/**
 * Editable query_id input used by every search mode. The value is embedded
 * into each selected result so the submitted file maps answers to the right
 * competition query.
 */
export function QueryIdField({ value, type, onChange, disabled }: QueryIdFieldProps) {
  return (
    <div className={styles.wrap}>
      <label className="field-label" htmlFor={`query-id-${type}`}>
        Query ID <span className={styles.hint}>(for submission)</span>
      </label>
      <div className={styles.row}>
        <input
          id={`query-id-${type}`}
          className="input"
          type="text"
          value={value}
          maxLength={100}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        />
        <button
          type="button"
          className="btn btn--ghost"
          disabled={disabled}
          title="Generate a new query id"
          onClick={() => onChange(defaultQueryId(type))}
        >
          ↻
        </button>
      </div>
    </div>
  );
}
