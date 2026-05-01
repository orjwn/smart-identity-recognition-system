import { KeyValueGrid, type KeyValueRow } from "../common/KeyValueGrid";
import { t, type Locale } from "../../i18n";
import type { BoardingPassRecord } from "../../types/kiosk";

type Props = { locale: Locale; boarding: BoardingPassRecord | null };

export function BoardingPassCard({ locale, boarding }: Props) {
  const rows: KeyValueRow[] = [
    { label: t(locale, "flight"), value: String(boarding?.flight_number ?? "") },
    { label: t(locale, "seat"), value: String(boarding?.seat ?? "") },
    { label: t(locale, "class"), value: String(boarding?.cabin_class ?? "") },
    { label: t(locale, "group"), value: String(boarding?.boarding_group ?? "") },
    { label: t(locale, "checkIn"), value: String(boarding?.check_in_status ?? "") },
    { label: t(locale, "security"), value: String(boarding?.security_status ?? "") },
  ];

  return (
    <div className="card">
      <div className="cardTitle">{t(locale, "boardingPass")}</div>
      <KeyValueGrid rows={rows} />
    </div>
  );
}
