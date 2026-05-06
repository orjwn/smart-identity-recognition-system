import { useCallback, useEffect, useState } from "react";
import type { KioskConnection, KioskState } from "../types/kiosk";

const SSE_URL = "/kiosk/events";
const RESET_URL = "/kiosk/reset";

export function useKioskEventSource() {
  const [state, setState] = useState<KioskState | null>(null);
  const [connection, setConnection] = useState<KioskConnection>("connecting");
  const [lastError, setLastError] = useState<string | null>(null);

  useEffect(() => {
    const es = new EventSource(SSE_URL);

    const handleUpdate = (e: MessageEvent) => {
      try {
        setState(JSON.parse(e.data) as KioskState);
        setLastError(null);
      } catch {
        setLastError("The backend sent an unreadable kiosk update.");
      }
    };

    es.addEventListener("init", handleUpdate);
    es.addEventListener("traveller_update", handleUpdate);
    es.onopen = () => {
      setConnection("open");
      setLastError(null);
    };
    es.onerror = () => {
      setConnection("error");
      setLastError("Waiting for backend event stream at /kiosk/events.");
    };

    return () => es.close();
  }, []);

  const resetServerState = useCallback(async () => {
    const response = await fetch(RESET_URL, { method: "POST" });
    if (!response.ok) {
      throw new Error(`Reset failed with HTTP ${response.status}`);
    }
  }, []);

  return { state, connection, lastError, resetServerState };
}
