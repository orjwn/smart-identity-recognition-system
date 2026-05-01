import { useId } from "react";
import { t, type Locale } from "../../i18n";
import { computeMapLayout, concourseIndex } from "./mapLayout";

type Props = {
  terminal?: string;
  gate?: string;
  locale: Locale;
  emphasize?: boolean;
};

function ClockIcon({ className }: { className?: string }) {
  return (
    <svg className={className} width="22" height="22" viewBox="0 0 24 24" aria-hidden>
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="1.75" opacity="0.9" />
      <path
        fill="none"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        d="M12 7v5l3 2"
      />
    </svg>
  );
}

export function AirportMap({ terminal, gate, locale, emphasize }: Props) {
  const uid = useId().replace(/:/g, "");
  const gradSky = `sky-${uid}`;
  const gradVignette = `vig-${uid}`;
  const gradPath = `path-${uid}`;
  const patGrid = `grid-${uid}`;
  const glowId = `glow-${uid}`;
  const gateGlow = `gate-glow-${uid}`;

  const g = gate?.trim() || "?";
  const layout = computeMapLayout(terminal, g);
  const { start, end, pathD } = layout;
  const termLabel = terminal?.trim() || "—";

  return (
    <div className={`airport-map-wrap ${emphasize ? "airport-map-wrap--focus" : ""}`}>
      <div className="airport-map-toolbar">
        <div className="airport-map-header">
          {terminal ? `${t(locale, "terminal")} ${termLabel}` : t(locale, "mapTitle")}
        </div>
        <div className="airport-map-eta" aria-live="polite">
          <div className="airport-map-eta__glow" />
          <ClockIcon className="airport-map-eta__icon" />
          <div className="airport-map-eta__text">
            <span className="airport-map-eta__title">{t(locale, "etaBadgeTitle")}</span>
            <span className="airport-map-eta__row">
              <strong className="airport-map-eta__mins">{layout.etaMinutes}</strong>
              <span className="airport-map-eta__unit">min</span>
            </span>
            <span className="airport-map-eta__hint">{t(locale, "etaWalkHint")}</span>
          </div>
        </div>
      </div>

      <div className="airport-map-canvas">
        <svg
          className="airport-map-svg"
          viewBox="0 0 400 312"
          role="img"
          aria-label={t(locale, "directionsHint")}
        >
          <defs>
            <linearGradient id={gradSky} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#0f1020" />
              <stop offset="35%" stopColor="#152a48" />
              <stop offset="70%" stopColor="#0c3d52" />
              <stop offset="100%" stopColor="#0a1628" />
            </linearGradient>
            <radialGradient id={gradVignette} cx="50%" cy="45%" r="75%">
              <stop offset="0%" stopColor="rgba(90,232,255,0.07)" />
              <stop offset="55%" stopColor="rgba(15,23,42,0)" />
              <stop offset="100%" stopColor="rgba(5,10,25,0.65)" />
            </radialGradient>
            <linearGradient id={gradPath} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#22d3ee" />
              <stop offset="50%" stopColor="#5ae8ff" />
              <stop offset="100%" stopColor="#34d399" />
            </linearGradient>
            <pattern id={patGrid} width="20" height="20" patternUnits="userSpaceOnUse">
              <circle cx="1" cy="1" r="0.8" fill="rgba(148, 200, 255, 0.12)" />
            </pattern>
            <filter id={glowId} x="-35%" y="-35%" width="170%" height="170%">
              <feGaussianBlur stdDeviation="2.5" result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id={gateGlow} x="-80%" y="-80%" width="260%" height="260%">
              <feGaussianBlur stdDeviation="3" result="bg" />
              <feMerge>
                <feMergeNode in="bg" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <rect width="400" height="312" fill={`url(#${gradSky})`} rx="14" />
          <rect width="400" height="312" fill={`url(#${gradVignette})`} rx="14" />
          <rect width="400" height="312" fill={`url(#${patGrid})`} opacity="0.5" rx="14" />

          <g transform="translate(348, 18)" opacity="0.85">
            <circle r="18" fill="rgba(15,28,52,0.85)" stroke="rgba(120,200,255,0.35)" strokeWidth="1" />
            <text x="0" y="5" textAnchor="middle" fill="#7dd3fc" fontSize="14" fontWeight="800" fontFamily="system-ui">
              N
            </text>
          </g>

          <rect
            x="18"
            y="182"
            width={Math.min(95 + layout.terminalNum * 6, 165)}
            height="58"
            fill="rgba(21,42,72,0.92)"
            stroke="rgba(120,200,255,0.2)"
            strokeWidth="1"
            rx="10"
          />
          <text x="30" y="208" fill="#93c5fd" fontSize="11" fontWeight="700" fontFamily="system-ui, sans-serif">
            {termLabel}
          </text>

          {[0, 1, 2, 3, 4].map((zone) => {
            const ci = concourseIndex(layout.concourse);
            const active = Math.floor(ci / 5) === zone;
            const x0 = 95 + zone * 58;
            return (
              <rect
                key={zone}
                x={x0}
                y="44"
                width="52"
                height="168"
                fill={active ? "rgba(56, 189, 248, 0.12)" : "rgba(19, 37, 66, 0.38)"}
                stroke={active ? "rgba(94, 234, 212, 0.35)" : "rgba(255,255,255,0.04)"}
                strokeWidth={active ? 1.2 : 0.6}
                rx="8"
                opacity={active ? 1 : 0.65}
              />
            );
          })}

          <path
            d={pathD}
            fill="none"
            stroke="#0e7490"
            strokeWidth="8"
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity="0.25"
          />
          <path
            d={pathD}
            fill="none"
            stroke={`url(#${gradPath})`}
            strokeWidth="4.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity="0.55"
          />
          <path
            className="airport-map-path"
            d={pathD}
            fill="none"
            stroke={`url(#${gradPath})`}
            strokeWidth="2.75"
            strokeLinecap="round"
            strokeLinejoin="round"
            filter={`url(#${glowId})`}
          />

          <circle cx={start.x} cy={start.y} r="11" fill="#0c4a6e" stroke="#67e8f9" strokeWidth="2.5" />
          <circle cx={start.x} cy={start.y} r="4" fill="#e0f2fe" opacity="0.95" />
          <rect
            x={start.x - 54}
            y={start.y + 14}
            width="108"
            height="24"
            rx="8"
            fill="rgba(12, 24, 48, 0.92)"
            stroke="rgba(103, 232, 249, 0.45)"
            strokeWidth="1"
          />
          <text
            x={start.x}
            y={start.y + 30}
            textAnchor="middle"
            fill="#ecfeff"
            fontSize="10"
            fontWeight="700"
            fontFamily="system-ui, sans-serif"
          >
            {t(locale, "youAreHere")}
          </text>

          <circle
            cx={end.x}
            cy={end.y}
            r="18"
            fill="none"
            stroke="rgba(52, 211, 153, 0.35)"
            strokeWidth="2"
            className="airport-map-gate-ring"
          />
          <circle
            cx={end.x}
            cy={end.y}
            r="12"
            fill="#134e4a"
            stroke="#5eead4"
            strokeWidth="2"
            filter={`url(#${gateGlow})`}
          />
          <rect
            x={end.x - 46}
            y={end.y - 42}
            width="92"
            height="26"
            rx="8"
            fill="rgba(6, 78, 59, 0.92)"
            stroke="rgba(94, 234, 212, 0.5)"
            strokeWidth="1"
          />
          <text
            x={end.x}
            y={end.y - 24}
            textAnchor="middle"
            fill="#ecfdf5"
            fontSize="12"
            fontWeight="800"
            fontFamily="system-ui, sans-serif"
          >
            {t(locale, "gateLabel", { gate: g })}
          </text>
        </svg>
      </div>

      <p className="airport-map-hint">{t(locale, "directionsHint")}</p>
    </div>
  );
}
