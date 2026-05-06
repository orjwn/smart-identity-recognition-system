import { formatKioskError, t, type Locale } from "../../i18n";
import type { RecognizedPayload } from "../../types/kiosk";
import { CameraFeed } from "./CameraFeed";

type Props = {
  locale: Locale;
  recognized: RecognizedPayload | null;
  error: string | null;
};

export function ScanningView({ locale, recognized, error }: Props) {
  return (
    <div className="panel">
      <div className="big">{t(locale, "scanning")}</div>
      <div className="muted">{t(locale, "lookAtCamera")}</div>

      <CameraFeed locale={locale} />

      {(recognized || error) && (
        <div className={`note ${error ? "note--warning" : ""}`}>
          <div>
            {recognized
              ? (
                  <>
                    {t(locale, "recognizedAs")}: <b>{recognized.name}</b>
                  </>
                )
              : t(locale, "waitingForBackend")}
          </div>
          {typeof recognized?.similarity === "number" && (
            <div>
              {t(locale, "similarity")}: {recognized.similarity.toFixed(3)}
            </div>
          )}
          {error && (
            <div className="error">
              {t(locale, "reason")}: {formatKioskError(locale, error)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
