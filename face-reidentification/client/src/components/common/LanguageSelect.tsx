import { LOCALE_DISPLAY_NAMES } from "../../constants/localeDisplayNames";
import { SUPPORTED_LOCALES, t, type Locale } from "../../i18n";

type Props = {
  locale: Locale;
  onChange: (code: string) => void;
};

export function LanguageSelect({ locale, onChange }: Props) {
  return (
    <label className="lang-row">
      <span className="lang-row__label">{t(locale, "changeLanguage")}</span>
      <select
        className="lang-select"
        value={locale}
        onChange={(e) => onChange(e.target.value)}
        aria-label={t(locale, "changeLanguage")}
      >
        {SUPPORTED_LOCALES.map((code) => (
          <option key={code} value={code}>
            {LOCALE_DISPLAY_NAMES[code]}
          </option>
        ))}
      </select>
    </label>
  );
}
