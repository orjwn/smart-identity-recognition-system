import { t, type Locale } from "../../i18n";
import { CameraFeed } from "./CameraFeed";

type Props = {
  locale: Locale;
  similarity?: number | null;
};

export function LiveCameraCard({ locale, similarity = null }: Props) {
  const hasSimilarity = typeof similarity === "number" && Number.isFinite(similarity);

  return (
    <div className="card camera-card">
      <div className="camera-card__bar">
        <div className="cardTitle">{t(locale, "liveCamera")}</div>
        <div className="similarity-badge" aria-label="Recognition similarity">
          {t(locale, "similarity")}: {hasSimilarity ? `${(similarity * 100).toFixed(1)}%` : t(locale, "unavailable")}
        </div>
      </div>
      <CameraFeed locale={locale} embedded />
    </div>
  );
}
