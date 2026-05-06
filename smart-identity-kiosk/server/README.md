# Backend Server

This folder contains the FastAPI backend used by the kiosk demo.

- `main.py` exposes health, camera stream, Server-Sent Events, reset and recognition endpoints.
- The backend loads mock airport records from `data/mock_airport/`.
- Recognition uses model wrappers from `models/` and generated runtime galleries from `database/face_database/`.

The backend uses simulated passenger records only. It does not connect to real passport, airline, airport or border-control systems.

