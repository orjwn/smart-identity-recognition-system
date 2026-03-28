import { useEffect, useState } from "react";
import "./App.css";

type KioskState = {
  recognized: null | { name: string; similarity?: number | null; device_id?: string | null };
  passport: any | null;
  boarding_pass: any | null;
  flight: any | null;
  error: string | null;
  updated_at: string;
};

function hm(iso?: string) {
  if (!iso) return "";
  const t = iso.split("T")[1];
  return t ? t.slice(0, 5) : iso;
}

export default function App() {
  const [state, setState] = useState<KioskState | null>(null);
  const [conn, setConn] = useState<"connecting" | "open" | "error">("connecting");

  useEffect(() => {
    const es = new EventSource("/kiosk/events");

    const handleUpdate = (e: MessageEvent) => {
      try {
        setState(JSON.parse(e.data));
      } catch {}
    };

    es.addEventListener("init", handleUpdate);
    es.addEventListener("traveller_update", handleUpdate);

    es.onopen = () => setConn("open");
    es.onerror = () => setConn("error");

    return () => es.close();
  }, []);

  async function resetKiosk() {
    await fetch("/kiosk/reset", { method: "POST" });
  }

  const scanning = !state?.recognized || !!state?.error;

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <div className="title">Smart Identity Kiosk</div>
          <div className="subtitle">SSE: {conn}</div>
        </div>
        <button className="btn" onClick={resetKiosk}>
          Reset
        </button>
      </header>

      {scanning ? (
        <div className="panel">
          <div className="big">Scanning…</div>
          <div className="muted">Please look at the camera.</div>

          {/* ✅ Live webcam stream */}
          <img
            src="/kiosk/video"
            alt="camera"
            style={{
              width: "100%",
              maxWidth: "680px",
              borderRadius: "16px",
              marginTop: "16px",
              border: "1px solid rgba(255,255,255,0.08)"
            }}
          />

          {state?.recognized && (
            <div className="note">
              <div>
                Recognized: <b>{state.recognized.name}</b>
              </div>
              {typeof state.recognized.similarity === "number" && (
                <div>Similarity: {state.recognized.similarity.toFixed(3)}</div>
              )}
              {state.error && <div className="error">Reason: {state.error}</div>}
            </div>
          )}
        </div>
      ) : (
        <div className="grid">
          {/* ✅ Optional: small camera preview while showing details */}
          <div className="card wide" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ padding: 18 }}>
              <div className="cardTitle">Live Camera</div>
            </div>
            <img
              src="/kiosk/video"
              alt="camera"
              style={{
                width: "100%",
                display: "block"
              }}
            />
          </div>

          <div className="card">
            <div className="cardTitle">Welcome</div>
            <div className="cardValue">{state?.passport?.full_name}</div>
            <div className="kv">
              <div className="k">Passport No.</div>
              <div className="v">{state?.passport?.passport_number}</div>
              <div className="k">Nationality</div>
              <div className="v">{state?.passport?.nationality}</div>
              <div className="k">DOB</div>
              <div className="v">{state?.passport?.date_of_birth}</div>
            </div>
          </div>

          <div className="card">
            <div className="cardTitle">Boarding Pass</div>
            <div className="kv">
              <div className="k">Flight</div>
              <div className="v">{state?.boarding_pass?.flight_number}</div>
              <div className="k">Seat</div>
              <div className="v">{state?.boarding_pass?.seat}</div>
              <div className="k">Class</div>
              <div className="v">{state?.boarding_pass?.cabin_class}</div>
              <div className="k">Group</div>
              <div className="v">{state?.boarding_pass?.boarding_group}</div>
              <div className="k">Check-in</div>
              <div className="v">{state?.boarding_pass?.check_in_status}</div>
              <div className="k">Security</div>
              <div className="v">{state?.boarding_pass?.security_status}</div>
            </div>
          </div>

          <div className="card wide">
            <div className="cardTitle">Flight Information</div>
            <div className="flightTop">
              <div className="flightNo">{state?.flight?.flight_number}</div>
              <div className="status">{state?.flight?.status}</div>
            </div>

            <div className="kv">
              <div className="k">Destination</div>
              <div className="v">
                {state?.flight?.destination_city}, {state?.flight?.destination_country}
              </div>

              <div className="k">Terminal</div>
              <div className="v">{state?.flight?.terminal}</div>

              <div className="k">Gate</div>
              <div className="v">{state?.flight?.gate}</div>

              <div className="k">Boarding</div>
              <div className="v">{hm(state?.flight?.boarding_starts)}</div>

              <div className="k">Gate Closes</div>
              <div className="v">{hm(state?.flight?.gate_closes)}</div>

              <div className="k">Departure</div>
              <div className="v">{hm(state?.flight?.scheduled_departure)}</div>

              <div className="k">Arrival</div>
              <div className="v">{hm(state?.flight?.scheduled_arrival)}</div>
            </div>

            <div className="actions">
              <button className="btnSecondary">Show directions</button>
              <button className="btnSecondary">Change language</button>
              <button className="btnSecondary" onClick={resetKiosk}>
                Not me
              </button>
            </div>

            <div className="mapPlaceholder">
              Map placeholder: route to Terminal {state?.flight?.terminal}, Gate {state?.flight?.gate}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
