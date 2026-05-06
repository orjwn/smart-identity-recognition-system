# Smart Identity Recognition System for Airports

Undergraduate Final Year Project implementation for BSc Computer Science with Artificial Intelligence.

This project demonstrates a local smart airport kiosk prototype that recognises an enrolled demo identity from a live camera feed, joins the recognised identity to simulated passport, boarding pass, and flight records, and displays the result in a React kiosk interface. It is not a production airport deployment and does not use real passenger, passport, airline, airport, or live flight-system data.

## Main Components

- `server/` - FastAPI kiosk backend, webcam capture, MJPEG video stream, SSE event stream, recognition worker, kiosk state, and mock traveller data joining.
- `client/` - React/Vite/TypeScript kiosk frontend.
- `models/` - wrappers for SCRFD, ArcFace, AdaFace, FocusFace, and MaskInv.
- `database/` - FAISS database wrapper and generated model-specific face database folders.
- `data/mock_airport/` - simulated passport, boarding pass, and flight JSON files.
- `evaluation/` - evaluation scripts, result CSVs, and generated plots used as report evidence.
- `tools/` - dataset preparation, alignment, split, and synthetic mask-generation tools.
- `utils/` - shared helper code for alignment, drawing, logging, and similarity support.
- `assets/faces/` - enrolment/demo face images used to build local recognition databases.
- `external/` - third-party/reference model repositories used by wrappers or retained for provenance.
- `weights/` - external model weights required for live recognition.
- `docs/` - project structure notes and archived audit/changelog notes.
- `report_assets/` - report/demo evidence assets.

## Important Scope Statement

The system is a local undergraduate implementation prototype. It demonstrates a software workflow for identity-linked passenger assistance, but it does not provide:

- production airport deployment;
- real passenger or passport data;
- real airline, airport, border-control, passport, or flight-status API integration;
- full privacy, security, GDPR, or biometric compliance;
- formal user-study evidence;
- formal demographic fairness evaluation;
- real indoor positioning or turn-by-turn airport navigation.

## Third-Party Code and Model Attribution

This project uses external face-recognition repositories as implementation references, model dependencies, and evaluation comparators. The project contribution is the integration of those components into the SmartIdentity kiosk pipeline, including backend state handling, webcam streaming, passenger/flight data joining, frontend kiosk UI, localisation controls, model routing, FAISS separation, and evaluation orchestration.

See:

- `THIRD_PARTY_ATTRIBUTION.md`
- `CODE_OWNERSHIP_MAP.md`
- `docs/PROJECT_STRUCTURE.md`

These files separate original integration work from copied, adapted, wrapper, external, generated, and dataset-derived material.

## Documentation Layout

- `README.md` - main run guide and submission overview.
- `THIRD_PARTY_ATTRIBUTION.md` - external source, licence, model, dataset, and redistribution notes.
- `CODE_OWNERSHIP_MAP.md` - ownership map for original, adapted, copied, wrapper, and external code.
- `docs/PROJECT_STRUCTURE.md` - folder structure explanation for supporting material.
- `docs/archive_notes/` - non-runtime audit, cleanup, changelog, and review notes kept as evidence but moved out of the project root.

## Backend Setup

From PowerShell:

```powershell
cd C:\Users\orjoa\OneDrive\Desktop\project\SmartIdentity\smart-identity-kiosk
.\venv\Scripts\Activate.ps1
cd server
python -m uvicorn main:app --port 8000
```

Useful backend URLs:

- Backend health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs
- Webcam stream: http://127.0.0.1:8000/kiosk/video
- SSE events: http://127.0.0.1:8000/kiosk/events
- Current kiosk state: http://127.0.0.1:8000/kiosk/state

The backend starts gracefully if the webcam, data files, model weights, or FAISS databases are unavailable. Check `/health` for exact readiness status.

## Frontend Setup

From PowerShell:

```powershell
cd C:\Users\orjoa\OneDrive\Desktop\project\SmartIdentity\smart-identity-kiosk\client
npm install
npm run dev
```

Open:

- Frontend: http://localhost:5173

The Vite dev server proxies `/kiosk`, `/health`, `/admin`, and `/traveller` requests to `http://127.0.0.1:8000`.

## Runtime Assets

The kiosk backend expects these external weights if live recognition is required:

- `weights/det_10g.onnx`
- `weights/w600k_mbf.onnx`
- `weights/adaface_ir18_webface4m.ckpt`
- `weights/w600k_r50.onnx`
- `weights/focus_face_w_pretrained.mdl`
- `weights/maskinv/maskinv_hg.onnx`
- `weights/maskinv/maskinv_hg.onnx.data`

The system also expects generated FAISS databases under:

- `database/face_database/arcface_mbf/`
- `database/face_database/adaface_ir18/`
- `database/face_database/arcface_r50/`
- `database/face_database/focusface/`
- `database/face_database/maskinv/`

If any model or database is missing, the backend remains online and reports the missing component through `/health`, but live recognition will be disabled until the asset is restored or rebuilt.

## Mock Data

The airport records are simulated and are stored in:

- `data/mock_airport/passports.json`
- `data/mock_airport/boarding_passes.json`
- `data/mock_airport/flights.json`

Joining logic:

1. Recognised face label is normalised and matched to `passports[*].full_name`.
2. The passport number is matched to `boarding_passes[*].passport_number`.
3. The boarding pass flight number is matched to `flights[*].flight_number`.

Do not replace these files with real passenger data for submission.

## Evaluation Material

Evaluation scripts and saved result CSVs live in `evaluation/`. The plotting script reads the saved CSVs and writes report-ready summaries/plots to `evaluation/plots/`.

```powershell
cd C:\Users\orjoa\OneDrive\Desktop\project\SmartIdentity\smart-identity-kiosk
.\venv\Scripts\Activate.ps1
python evaluation\plot_results.py
```

The evaluation code is separate from the live kiosk runtime and should not modify the runtime FAISS databases.

## Troubleshooting

- If `/health` returns `camera.ready: false`, check that a webcam is connected or set `CAMERA_INDEX`.
- If `/health` shows a model as missing, restore the corresponding file in `weights/`.
- If `/health` shows a database as missing, rebuild or restore the matching `database/face_database/<model>/` folder.
- If the frontend shows an event stream error, start the backend first and confirm `http://127.0.0.1:8000/kiosk/events` responds.
- If the camera image does not load in the browser, open `http://127.0.0.1:8000/kiosk/video` directly.
- If a traveller is recognised but no details appear, check spelling consistency between the face database label and `data/mock_airport/passports.json`.

## Recommended Submission Zip Contents

Include:

- `.gitignore`
- `README.md`
- `THIRD_PARTY_ATTRIBUTION.md`
- `CODE_OWNERSHIP_MAP.md`
- `requirements.txt`
- `download.sh`
- `main.py`
- `server/`
- `client/`, excluding `client/node_modules/` and `client/dist/`
- `models/`
- `database/`, including `face_database/` only if required and allowed
- `data/`
- `evaluation/`
- `tools/`
- `utils/`
- `assets/`, if allowed
- `docs/`
- `report_assets/`

Exclude unless specifically required and permitted by licence, file-size, and university rules:

- `venv/`
- `client/node_modules/`
- `client/dist/`
- `__pycache__/`
- `.pytest_cache/`
- `*.log`
- large datasets
- model weights
- generated FAISS databases
- external vendor folders

## External Repository Notes

The upstream README text from `yakhyo/face-reidentification` is not repeated here to avoid confusing upstream claims with this project. Its role is documented in `THIRD_PARTY_ATTRIBUTION.md` and `CODE_OWNERSHIP_MAP.md`.
