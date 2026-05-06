# Smart Identity Kiosk Frontend

This folder contains the React, TypeScript and Vite kiosk UI for the Smart Identity Recognition System for Airports.

The frontend presents the passenger-facing kiosk screen. It does not perform face recognition in the browser; recognition, camera handling and traveller-data joining are handled by the FastAPI backend.

## Purpose

The kiosk UI connects to the backend, shows the live camera/scanning state, and displays recognised traveller information from the simulated airport records.

## Main Technologies

- React
- TypeScript
- Vite
- `@vitejs/plugin-react-swc` for the Vite React plugin

## Backend Connection

The Vite dev server proxies these paths to the FastAPI backend at `http://127.0.0.1:8000`:

- `/kiosk`
- `/admin`
- `/health`
- `/traveller`

The backend must be running before the frontend can receive recognition updates or display the live camera feed.

## Key Features

- Scanning screen for the waiting/recognition state.
- Live camera feed loaded from `/kiosk/video`.
- Server-Sent Events connection to `/kiosk/events`.
- Recognised traveller dashboard.
- Passport card.
- Boarding pass card.
- Flight details card.
- Language selector.
- Schematic airport map and highlighted route view.
- Reset / not-me flow through the backend reset endpoint.

## Run Locally

From this folder:

```powershell
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

Backend requirement:

```text
http://127.0.0.1:8000
```

## Localisation

Localisation is implemented through static project strings and the traveller/passport language preference, with a manual selector in the UI. It is not a full browser-language-detection implementation.

## Navigation

The airport map is a schematic prototype view for report/demo purposes. It is not a real indoor positioning or turn-by-turn routing engine.

## Template Note

This README replaces the default Vite template README. GitHub links from the original template do not imply that all optional Vite/ESLint plugins were used in this project.

