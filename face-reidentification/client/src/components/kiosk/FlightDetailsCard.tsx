import { LanguageSelect } from "../common/LanguageSelect";
import { KeyValueGrid, type KeyValueRow } from "../common/KeyValueGrid";
import { formatTime, translateFlightStatus, t, type Locale } from "../../i18n";
import type { FlightRecord } from "../../types/kiosk";

type Props = {
  locale: Locale;
  flight: FlightRecord | null;
  routeLabel: string;
  onLanguageChange: (code: string) => void;
  onToggleDirections: () => void;
};

export function FlightDetailsCard({
  locale,
  flight,
  routeLabel,
  onLanguageChange,
  onToggleDirections,
}: Props) {
  const destination =
    flight?.destination_city != null
      ? `${flight.destination_city}${
          flight.destination_country ? `, ${flight.destination_country}` : ""
        }`
      : "";

  const rows: KeyValueRow[] = [
    { label: t(locale, "airline"), value: flight?.airline ?? "-" },
    { label: t(locale, "route"), value: routeLabel },
    { label: t(locale, "destination"), value: destination },
    { label: t(locale, "terminal"), value: flight?.terminal ?? "-" },
    { label: t(locale, "gate"), value: flight?.gate ?? "-" },
    { label: t(locale, "boarding"), value: formatTime(flight?.boarding_starts, locale) },
    { label: t(locale, "gateCloses"), value: formatTime(flight?.gate_closes, locale) },
    { label: t(locale, "departure"), value: formatTime(flight?.scheduled_departure, locale) },
    { label: t(locale, "arrival"), value: formatTime(flight?.scheduled_arrival, locale) },
  ];

  return (
    <div className="card flight-details-card">
      <div className="flight-details-heading">{t(locale, "yourFlightDetails")}</div>
      <div className="flight-hero">
        <span className="flight-hero__no">{flight?.flight_number}</span>
        <span className="flight-hero__status">{translateFlightStatus(locale, flight?.status)}</span>
      </div>
      <KeyValueGrid rows={rows} className="flight-kv" />

      <LanguageSelect locale={locale} onChange={onLanguageChange} />
      <p className="guidance-note">{t(locale, "guidanceNote")}</p>

      <div className="actions">
        <button type="button" className="btn btn--primary" onClick={onToggleDirections}>
          {t(locale, "showDirections")}
        </button>
      </div>
    </div>
  );
}
