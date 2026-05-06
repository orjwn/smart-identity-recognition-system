import { t, type Locale } from "../../i18n";
import type { KioskConnection } from "../../types/kiosk";

type Props = {
  locale: Locale;
  connection: KioskConnection;
  onReset: () => void;
};

export function KioskHeader({ locale, connection, onReset }: Props) {
  const connectionText = t(locale, `connection_${connection}`);

  return (
    <header className="topbar">
      <div>
        <div className="title">{t(locale, "appTitle")}</div>
        <div className="subtitle">
          {t(locale, "sse")}: {connectionText}
        </div>
      </div>
      <button type="button" className="btn" onClick={onReset}>
        {t(locale, "reset")}
      </button>
    </header>
  );
}
