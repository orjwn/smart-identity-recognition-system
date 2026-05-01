# Code Review and Cleanup Notes

## Scope Reviewed

Reviewed only `SmartIdentity/face-reidentification`. No files in `FYP resrarchs` or `uni instructions` were edited.

## Controlled Move Summary

- Moved `adaface-test/` to `external/adaface-test/`.
- Moved `Masked-Face-Recognition-KD/` to `external/Masked-Face-Recognition-KD/`.
- Left `external/FocusFace/` in place because it was already correctly grouped as an external repository.
- Updated runtime imports/paths that depended on the old AdaFace location.

## Trash Folder Findings

The archive folder is located at `C:\Users\orjoa\OneDrive\Desktop\project\SmartIdentity\trash`. It was inspected at top level and sampled carefully, but nothing was restored, copied, moved, or deleted from `trash`.

Classification summary:

- `trash/arcface/` - useful only as archive/evidence or an old duplicate recognition database snapshot.
- `trash/archive/` - useful only as archive/evidence; appears to contain large old video/output material.
- `trash/processed/` and `trash/raw/` - useful only as archive/evidence or old dataset material.
- `trash/src/` - unclear/old experiment or source snapshot; keep archived unless a specific missing feature is needed.
- `trash/venv/` - duplicate generated Python environment; do not submit.
- Top-level `*.onnx` and `*.pth` files - model weights/checkpoints; preserve as archive assets, but do not restore blindly.
- Top-level `app.log` - old generated log; do not submit.
- Top-level AdaFace test scripts - useful only as archive/evidence of old experiments unless deliberately revived later.

Overall recommendation: keep `SmartIdentity/trash` out of the final submitted source package unless the university specifically asks for supporting archive evidence. It is very large and includes old datasets, generated environment files, logs, checkpoints, and duplicate/experimental material.

## Useful Folders and Files

- `server/main.py` - FastAPI kiosk backend, webcam MJPEG stream, SSE events, recognition worker, and traveller data joining.
- `client/src/` - React kiosk UI, language selection, live camera display, traveller dashboard, airport map, and SSE hook.
- `models/` - model wrappers for SCRFD, ArcFace, AdaFace, FocusFace, and MaskInv.
- `database/face_db.py` - FAISS database wrapper.
- `database/face_database/` - generated FAISS indexes and metadata for each model.
- `data/passports.json` - simulated passport records.
- `data/boarding_passes.json` - simulated boarding pass records.
- `data/flights.json` - simulated flight records.
- `evaluation/` - evaluation scripts, result CSVs, and plotting code for report evidence.
- `tools/` - dataset preparation, alignment, splitting, and APFF helper scripts.
- `external/` - copied third-party/reference repositories (`adaface-test`, `FocusFace`, `Masked-Face-Recognition-KD`).
- `weights/` - required model weights for the live demo.

## Files or Folders That Seem Duplicated, Old, or Research-Only

- `main.py` at project root appears to be an older/upstream non-kiosk entry point. The expected current backend command uses `server/main.py`.
- `client/dist/` is generated build output and can be rebuilt with `npm run build`.
- `client/public/vite.svg` and `client/src/assets/react.svg` are default Vite/React starter assets and do not appear central to the kiosk.
- `utils/utils.py` is currently empty.
- `server/debug_faces/` appears to contain debug output.
- `external/adaface-test/`, `external/FocusFace/`, and `external/Masked-Face-Recognition-KD/` are research/vendor/reference code rather than core kiosk submission code.
- `app.log` is a generated runtime log and is large.
- `__pycache__/` folders are generated Python cache folders.

No files were deleted.

## Files That Should Usually Not Be Submitted

- `venv/`
- `client/node_modules/`
- `client/dist/`
- `__pycache__/`
- `app.log` and other `*.log` files
- `weights/` unless the university explicitly asks for local runnable assets
- large raw/processed datasets unless required as supporting evidence
- generated FAISS databases unless the demo must run offline from the submitted package

## Git Exclusion Suggestions

The existing `.gitignore` already excludes many heavy/generated paths, including virtual environments, `node_modules`, `dist`, weights, databases, datasets, logs, external code, and evaluation outputs.

Recommended follow-up:

- Confirm whether `evaluation/results_*.csv` and `evaluation/plots/` should be submitted as evidence or excluded as generated artefacts.
- Confirm whether `database/face_database/` should be included only in the demo machine copy, not in Git.
- Review the current Git status carefully: there are many tracked deletions under `data/processed/test/` that were already present before this cleanup pass. I did not restore or remove those files.

## Data Join Review

The current join path is:

1. recognised face label -> `passports.full_name`
2. `passport_number` -> `boarding_passes.passport_number`
3. `flight_number` -> `flights.flight_number`

The mock data is consistent for the visible records: passport numbers and flight numbers line up. One mild schema gap remains: the frontend type supports `origin_city` and `origin_country`, but `flights.json` mainly stores outbound destination details. I did not invent additional origin data.

## Changes Actually Made

- Made the backend startup more resilient when webcam, model weights, data files, or FAISS databases are unavailable.
- Added backend service readiness details to `/health` and kiosk state snapshots.
- Kept `/kiosk/video` alive with a status frame when no camera frame is available.
- Kept SSE init/update payloads consistent with the latest kiosk state plus service status.
- Improved frontend SSE error reporting and reset failure handling.
- Added a reusable `CameraFeed` component with a visible camera stream failure message.
- Added friendly frontend messages for backend error codes such as missing camera, data, recognition assets, or unmatched traveller records.
- Added evaluation metric notes to `evaluation/plot_results.py`.
- Prepended project-specific FYP run/setup/supporting-material guidance to `README.md`.
- Created this review and cleanup notes file.
- Ran verification commands. `npm run build` refreshed ignored frontend build output in `client/dist/`, and Python compilation refreshed `__pycache__` files.
- Moved root-level external repositories into `external/adaface-test/` and `external/Masked-Face-Recognition-KD/`.
- Updated AdaFace wrapper import path to use `external/adaface-test/AdaFace`.
- Inspected the actual archive folder at `C:\Users\orjoa\OneDrive\Desktop\project\SmartIdentity\trash` and left all contents untouched.

## Code Improvements Made and Verification

| File | Improvement | Why it helps | Verification |
|---|---|---|---|
| `server/main.py` | Added focused docstrings/comments around service readiness, data joining, SSE broadcasting, camera fallback, recognition debounce, recognition worker, and model-specific FAISS loading. | Helps a second examiner understand the prototype design and demo resilience without changing runtime behaviour. | Python compile passed; backend `/health` smoke test passed. |
| `models/adaface.py` | Updated the external AdaFace repository path to `external/adaface-test/AdaFace` and documented the wrapper boundary. | Keeps imports working after moving the vendor repository and clarifies that project code uses a stable wrapper. | Python compile passed; backend `/health` loaded AdaFace successfully. |
| `models/focusface.py` | Added a wrapper docstring explaining the external FocusFace dependency. | Makes external model path handling clearer for final review. | Python compile passed; backend `/health` loaded FocusFace successfully. |
| `models/maskinv.py` | Added a short docstring for the embedding call. | Clarifies expected aligned input and normalized output. | Python compile passed; backend `/health` loaded MaskInv successfully. |
| `database/face_db.py` | Added class documentation and zero-norm embedding protection for FAISS storage/search. | Explains why databases are model-specific and avoids invalid vector normalization edge cases. | Python compile passed; backend `/health` loaded all FAISS databases successfully. |
| `tools/assets/add_mask.py` | Updated reference comment after moving the external MaskInv repository. | Removes stale path wording. | Python compile passed. |
| `evaluation/evaluate_*.py` and `evaluation/plot_results.py` | Added evaluation-methodology comments/docstrings stating that scripts use in-memory galleries or saved CSVs, not runtime FAISS databases. | Helps examiners distinguish evaluation evidence from live kiosk state. | Python compile passed. |
| `README.md`, `PROJECT_STRUCTURE.md`, `FINAL_SUBMISSION_CHECKLIST.md` | Documented cleaned structure, run commands, required assets, limitations, and submission guidance. | Improves GitHub/supervisor/viva readiness. | Manual review plus successful backend/frontend verification commands. |

## Recommended Follow-Up Changes Not Made

- Decide whether to keep or archive the old root `main.py` entry point. It may confuse markers because the kiosk backend uses `server/main.py`.
- Consider adding a small automated data integrity test for `passports.json`, `boarding_passes.json`, and `flights.json`.
- Consider adding a backend smoke test that imports `server/main.py` and calls `/health` with missing assets mocked.
- Consider replacing celebrity names/images in demo assets with clearly synthetic or consented mock identities if required by ethics guidance.
- Consider documenting the exact commands used to build each FAISS database.
- Consider a short `evaluation/README.md` describing which scripts generated each CSV.
