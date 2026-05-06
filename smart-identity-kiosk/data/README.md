# Mock Airport Data

This folder contains simulated airport records used by the kiosk backend.

- `mock_airport/passports.json` contains mock passport records.
- `mock_airport/boarding_passes.json` contains mock boarding pass records.
- `mock_airport/flights.json` contains mock flight records.

These files are local prototype data only. They are not real passenger, airline, airport, border-control or passport-system records.

The backend loads these records through `server/main.py` and joins them using:

1. recognised face label to passport full name;
2. passport number to boarding pass;
3. flight number to flight details.

Biometric FAISS databases are stored separately in `database/face_database/`. Offline evaluation datasets are stored separately in `datasets/`.
