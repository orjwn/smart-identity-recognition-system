import { KeyValueGrid, type KeyValueRow } from "../common/KeyValueGrid";
import { t, type Locale } from "../../i18n";
import type { PassportRecord } from "../../types/kiosk";

type Props = { locale: Locale; passport: PassportRecord | null };

export function PassportCard({ locale, passport }: Props) {
  const rows: KeyValueRow[] = [
    { label: t(locale, "passportNo"), value: String(passport?.passport_number ?? "") },
    { label: t(locale, "nationality"), value: String(passport?.nationality ?? "") },
    { label: t(locale, "dob"), value: String(passport?.date_of_birth ?? "") },
  ];

  return (
    <div className="card">
      <div className="cardTitle">{t(locale, "welcome")}</div>
      <div className="cardValue">{String(passport?.full_name ?? "")}</div>
      <KeyValueGrid rows={rows} />
    </div>
  );
}
