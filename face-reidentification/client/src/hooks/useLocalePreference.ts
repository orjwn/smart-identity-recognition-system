import { useEffect, useRef, useState } from "react";
import { normalizeLocale, type Locale } from "../i18n";
import type { PassportRecord } from "../types/kiosk";

export function useLocalePreference(passport: PassportRecord | null) {
  const manualOverride = useRef(false);
  const [locale, setLocale] = useState<Locale>("en");

  useEffect(
    () => {
      if (!passport || manualOverride.current) return;
      setLocale(normalizeLocale(passport.preferred_language));
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- passport object identity changes every SSE tick
    [passport?.passport_number, passport?.preferred_language],
  );

  const selectLocale = (code: string) => {
    manualOverride.current = true;
    setLocale(normalizeLocale(code));
  };

  const clearManualAndReset = () => {
    manualOverride.current = false;
    setLocale("en");
  };

  return { locale, selectLocale, clearManualAndReset };
}
