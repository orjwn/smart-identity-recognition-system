import type { FlightRecord } from "../types/kiosk";

export function buildRouteLabel(flight: FlightRecord | null | undefined): string {
  if (!flight) return "-";
  if (flight.origin_city && flight.destination_city) {
    return `${flight.origin_city} -> ${flight.destination_city}`;
  }
  if (flight.destination_city) {
    return flight.destination_country
      ? `${flight.destination_city}, ${flight.destination_country}`
      : flight.destination_city;
  }
  return "-";
}

export function asFlightRecord(raw: unknown): FlightRecord | null {
  if (!raw || typeof raw !== "object") return null;
  return raw as FlightRecord;
}
