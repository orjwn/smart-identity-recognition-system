import { AirportMap } from "../map/AirportMap";
import type { Locale } from "../../i18n";
import type { BoardingPassRecord, FlightRecord, PassportRecord } from "../../types/kiosk";
import { BoardingPassCard } from "./BoardingPassCard";
import { FlightDetailsCard } from "./FlightDetailsCard";
import { LiveCameraCard } from "./LiveCameraCard";
import { PassportCard } from "./PassportCard";

type Props = {
  locale: Locale;
  passport: PassportRecord | null;
  boarding: BoardingPassRecord | null;
  flight: FlightRecord | null;
  similarity: number | null;
  routeLabel: string;
  directionsFocus: boolean;
  onLanguageChange: (code: string) => void;
  onToggleDirections: () => void;
};

export function TravellerDashboard({
  locale,
  passport,
  boarding,
  flight,
  similarity,
  routeLabel,
  directionsFocus,
  onLanguageChange,
  onToggleDirections,
}: Props) {
  return (
    <div className="kiosk-flow">
      <LiveCameraCard locale={locale} similarity={similarity} />

      <div className="grid-two">
        <PassportCard locale={locale} passport={passport} />
        <BoardingPassCard locale={locale} boarding={boarding} />
      </div>

      <div className="flight-map-split">
        <FlightDetailsCard
          locale={locale}
          flight={flight}
          routeLabel={routeLabel}
          onLanguageChange={onLanguageChange}
          onToggleDirections={onToggleDirections}
        />

        <div className="card map-card">
          <AirportMap
            terminal={flight?.terminal}
            gate={flight?.gate}
            locale={locale}
            emphasize={directionsFocus}
          />
        </div>
      </div>
    </div>
  );
}
