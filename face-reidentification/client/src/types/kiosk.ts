export type KioskConnection = "connecting" | "open" | "error";

export type RecognizedPayload = {
  name: string;
  similarity?: number | null;
  device_id?: string | null;
  mode?: string | null;
  masked?: boolean | null;
  mask_score?: number | null;
};

export type KioskState = {
  recognized: RecognizedPayload | null;
  passport: Record<string, unknown> | null;
  boarding_pass: Record<string, unknown> | null;
  flight: Record<string, unknown> | null;
  error: string | null;
  updated_at: string;
  service?: Record<string, unknown>;
};

export type FlightRecord = {
  flight_number?: string;
  airline?: string;
  origin_city?: string;
  origin_country?: string;
  destination_city?: string;
  destination_country?: string;
  terminal?: string;
  gate?: string;
  scheduled_departure?: string;
  boarding_starts?: string;
  gate_closes?: string;
  scheduled_arrival?: string;
  status?: string;
};

export type PassportRecord = {
  full_name?: string;
  passport_number?: string;
  nationality?: string;
  date_of_birth?: string;
  preferred_language?: string;
};

export type BoardingPassRecord = {
  flight_number?: string;
  seat?: string;
  cabin_class?: string;
  boarding_group?: string;
  check_in_status?: string;
  security_status?: string;
};
