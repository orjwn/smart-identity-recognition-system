import { useState } from "react";
import { t, type Locale } from "../../i18n";

type Props = {
  locale: Locale;
  embedded?: boolean;
};

export function CameraFeed({ locale, embedded = false }: Props) {
  const [failed, setFailed] = useState(false);

  return (
    <div className={embedded ? "camera-shell camera-shell--embed" : "camera-shell"}>
      <img
        src="/kiosk/video"
        alt=""
        className={embedded ? "camera-feed camera-feed--embed" : "camera-feed"}
        onError={() => setFailed(true)}
        onLoad={() => setFailed(false)}
      />
      {failed && (
        <div className="camera-alert" role="status">
          {t(locale, "cameraStreamUnavailable")}
        </div>
      )}
    </div>
  );
}
