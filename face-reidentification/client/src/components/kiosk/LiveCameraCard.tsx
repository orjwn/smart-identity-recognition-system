import { t, type Locale } from "../../i18n";
import { CameraFeed } from "./CameraFeed";

type Props = { locale: Locale };

export function LiveCameraCard({ locale }: Props) {
  return (
    <div className="card camera-card">
      <div className="camera-card__bar">
        <div className="cardTitle">{t(locale, "liveCamera")}</div>
      </div>
      <CameraFeed locale={locale} embedded />
    </div>
  );
}
