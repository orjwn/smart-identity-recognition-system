import { useState } from "react";
import { KioskHeader } from "./components/kiosk/KioskHeader";
import { ScanningView } from "./components/kiosk/ScanningView";
import { TravellerDashboard } from "./components/kiosk/TravellerDashboard";
import { useDocumentLocale } from "./hooks/useDocumentLocale";
import { useKioskEventSource } from "./hooks/useKioskEventSource";
import { useLocalePreference } from "./hooks/useLocalePreference";
import type { BoardingPassRecord, PassportRecord } from "./types/kiosk";
import { asFlightRecord, buildRouteLabel } from "./utils/flight";
import "./App.css";

export default function App() {
  const { state, connection, lastError, resetServerState } = useKioskEventSource();
  const [directionsFocus, setDirectionsFocus] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);

  const passport = (state?.passport ?? null) as PassportRecord | null;
  const boarding = (state?.boarding_pass ?? null) as BoardingPassRecord | null;
  const flight = asFlightRecord(state?.flight);
  const routeLabel = buildRouteLabel(flight);
  const recognitionSimilarity = state?.recognized?.similarity ?? null;

  const { locale, selectLocale, clearManualAndReset } = useLocalePreference(passport);
  useDocumentLocale(locale);

  async function resetKiosk() {
    clearManualAndReset();
    setDirectionsFocus(false);
    setResetError(null);
    try {
      await resetServerState();
    } catch (error) {
      setResetError(error instanceof Error ? error.message : "Reset failed.");
    }
  }

  const scanning = !state?.recognized || !!state?.error;
  const statusError = state?.error ?? lastError ?? resetError;

  return (
    <div className={`page ${locale === "ar" ? "page--rtl" : ""}`}>
      <KioskHeader locale={locale} connection={connection} onReset={resetKiosk} />

      {scanning ? (
        <ScanningView
          locale={locale}
          recognized={state?.recognized ?? null}
          error={statusError}
        />
      ) : (
        <TravellerDashboard
          locale={locale}
          passport={passport}
          boarding={boarding}
          flight={flight}
          similarity={recognitionSimilarity}
          routeLabel={routeLabel}
          directionsFocus={directionsFocus}
          onLanguageChange={selectLocale}
          onToggleDirections={() => setDirectionsFocus((v) => !v)}
        />
      )}
    </div>
  );
}
