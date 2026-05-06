import { useEffect } from "react";
import type { Locale } from "../i18n";

export function useDocumentLocale(locale: Locale) {
  useEffect(() => {
    document.documentElement.lang = locale;
    document.documentElement.dir = locale === "ar" ? "rtl" : "ltr";
  }, [locale]);
}
