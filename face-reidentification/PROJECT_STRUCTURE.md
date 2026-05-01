# Project Structure

This is the cleaned structure for the Smart Identity Recognition System for Airports implementation.

## Core Runtime

- `server/` - FastAPI backend used by the final kiosk demo.
  - `server/main.py` exposes `/health`, `/docs`, `/kiosk/video`, `/kiosk/events`, `/kiosk/state`, `/kiosk/reset`, and `/kiosk/recognized`.
  - It owns webcam capture, recognition debounce, Server-Sent Events, and mock traveller data joining.
- `client/` - React/Vite frontend used by the final kiosk demo.
  - `client/src/components/kiosk/` contains the camera, scanning, traveller, passport, boarding pass, and flight display.
  - `client/src/components/map/` contains the prototype airport map/directions view.
  - `client/src/hooks/useKioskEventSource.ts` connects to the backend SSE stream.
- `models/` - stable project wrappers around the recognition/detection models.
  - `scrfd.py` detects faces and landmarks.
  - `arcface.py`, `adaface.py`, `focusface.py`, and `maskinv.py` produce embeddings.
- `database/` - FAISS database code and generated model-specific galleries.
  - `database/face_db.py` is the FAISS wrapper.
  - `database/face_database/<model>/` contains generated indexes and metadata.

## Data and Demo Assets

- `data/` - simulated airport records.
  - `passports.json` contains mock passport records.
  - `boarding_passes.json` contains mock boarding pass records.
  - `flights.json` contains mock flight records.
- `assets/faces/` - enrolled/demo face images used to build recognition databases.
- `weights/` - local model weights required for the demo. These are large assets and should normally stay out of Git.

## Evaluation and Tooling

- `evaluation/` - evaluation scripts, saved CSV result files, and generated plots.
  - The evaluation scripts build in-memory galleries from `datasets/gallery` and `datasets/test`.
  - They do not modify the runtime FAISS databases used by the kiosk.
- `tools/` - dataset preparation and helper scripts.
- `utils/` - shared Python helpers such as face alignment and logging.
- `datasets/` - processed and raw evaluation datasets. Treat this as a large local artefact unless the university asks for it.

## External Reference Repositories

Copied third-party or research repositories are grouped under `external/`:

- `external/adaface-test/` - AdaFace reference code. `models/adaface.py` imports its network definition from here.
- `external/FocusFace/` - FocusFace reference code. `models/focusface.py` imports the FocusFace model from here.
- `external/Masked-Face-Recognition-KD/` - Masked face recognition / knowledge-distillation reference code retained for traceability.

These folders are not the project-owned runtime architecture; the stable project entry points are the wrappers in `models/`.

## Generated or Archive-Like Material

- `client/dist/` - generated frontend build output.
- `client/node_modules/` - frontend dependencies.
- `venv/` - Python virtual environment.
- `__pycache__/` - Python bytecode cache.
- `app.log` and other `*.log` files - runtime/evaluation logs.
- `server/debug_faces/` - debug output.
- root `main.py` - older non-kiosk entry point retained for reference; the final backend command uses `server/main.py`.
- `../trash/` - project-level archive folder containing old experiments, datasets, checkpoints, logs, and generated environments. It was inspected but not changed during cleanup.

No folders were deleted during cleanup.
