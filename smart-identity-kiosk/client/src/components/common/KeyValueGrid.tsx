import { Fragment } from "react";

export type KeyValueRow = { label: string; value: string };

type Props = {
  rows: KeyValueRow[];
  className?: string;
};

export function KeyValueGrid({ rows, className = "" }: Props) {
  return (
    <div className={`kv ${className}`.trim()}>
      {rows.map((row) => (
        <Fragment key={row.label}>
          <div className="k">{row.label}</div>
          <div className="v">{row.value}</div>
        </Fragment>
      ))}
    </div>
  );
}
