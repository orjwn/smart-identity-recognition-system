# Code Ownership Map

Project: Smart Identity Recognition System for Airports  
Student: Orjuwan Almotarafi  
Main code folder: `SmartIdentity/smart-identity-kiosk`

This map separates original SmartIdentity integration work from copied, lightly modified or adapted third-party source code, external vendor/reference folders, datasets and generated assets.

## My Original / Integration Code

| Area | Files / folders | Classification | Ownership note |
|---|---|---|---|
| FastAPI kiosk backend | `server/main.py`, supporting backend data files under `server/` | Original project integration code | Airport-style kiosk state, passenger/flight data joining, SSE updates and MJPEG webcam stream. Uses recognizer wrappers but is not itself copied from the listed repositories. |
| React kiosk UI | `client/src` | Original project integration code | Kiosk interface and localisation UI. No `i18next-browser-languageDetector` import/use was found in `client/src` or `client/package.json` during this audit. |
| Mock airport data joining | `server/` data/config files and backend logic | Original project integration code | Project-specific passenger/flight/passport-style joining used to present recognition outcomes in an airport context. |
| Model orchestration | Parts of `main.py`, `server/main.py`, `evaluation/*.py` | Mixed: adapted baseline plus original integration | `main.py` is adapted from the yakhyo baseline; server/evaluation orchestration is project-specific routing across ArcFace, MaskInv, AdaFace, FocusFace and hybrid comparison modes. |
| Evaluation orchestration | `evaluation/evaluate_*.py`, `evaluation/plot_results.py`, `evaluation/results_*.csv` | Original project evaluation orchestration | Generated local comparisons of recognizers on clean/masked aligned data. No copied external evaluation code was identified in this attribution pass. |
| Dataset preparation tooling | `tools/dataset_preparation/prepare_apff.py`, `tools/dataset_preparation/split_dataset.py`, `tools/dataset_preparation/align_by_identity.py` | Original project code inspired by dataset requirements | Local scripts for preparing, aligning, splitting and cleaning APFF-style evaluation data. |
| Documentation and submission notes | `PROJECT_STRUCTURE.md`, `PROJECT_FACTS_FROM_CODE.md`, `IMPLEMENTATION_JUSTIFICATION_FROM_SOURCES.md`, `THIRD_PARTY_ATTRIBUTION.md`, `CODE_OWNERSHIP_MAP.md` | Project documentation / provenance evidence | Documentation written for the SmartIdentity submission, with attribution to third-party sources. |

## Copied Unchanged External Code

| Local file / block | External source | What is unchanged | Student modification |
|---|---|---|---|
| `utils/helpers.py` ArcFace reference landmark values | yakhyo/face-reidentification | The five landmark coordinate values used for ArcFace alignment | None to the landmark values; retained for compatibility with the original alignment implementation. |

No whole local edited Python file in this audit was marked as copied unchanged. External vendor folders may contain copied repository code, but they require manual comparison before describing them as exact unchanged copies.

## Copied And Lightly Modified From External Repositories

| Local file / block | External source | What is lightly modified | Student modification |
|---|---|---|---|
| `models/arcface.py` | yakhyo/face-reidentification | ArcFace ONNX wrapper remains close to the upstream implementation | Minor edits for logging, validation/error handling, output-shape checks and SmartIdentity alignment/database integration. |
| `models/scrfd.py` | yakhyo/face-reidentification | SCRFD detector, anchor decoding flow and NMS structure remain close to the upstream implementation | Minor edits for imports, type hints, local demo/debug helpers and backend/evaluation integration. |
| `utils/helpers.py` alignment functions | yakhyo/face-reidentification | `estimate_norm` and `face_alignment` logic | Minor edits for typing, docstrings, comments and reuse across model wrappers/evaluation scripts. |
| `utils/helpers.py` SCRFD decoders | yakhyo/face-reidentification | `distance2bbox` and `distance2kps` decoding logic | Minor edits for local typing/style and reuse through the SCRFD wrapper. |
| `utils/helpers.py` visualisation helpers | yakhyo/face-reidentification | Cosine similarity and bounding-box overlay helpers | Minor edits for local visualisation while backend/kiosk logic adds separate passenger-state integration. |
| `models/adaface.py` `_to_input` block | mk-minchul/AdaFace | AdaFace normalization and tensor conversion pattern | Adapted from PIL/RGB input to aligned BGR crops produced by SmartIdentity. |
| `README.md` lower upstream README section | yakhyo/face-reidentification | Retained upstream baseline documentation | SmartIdentity documentation was added above; retained upstream block may differ due to local formatting/encoding. |

## Adapted From External Repositories

| Local file / block | External source | What is adapted | Student contribution |
|---|---|---|---|
| `database/face_db.py` | yakhyo/face-reidentification | FAISS identity gallery and similarity search pattern | Substantially modified with model-specific storage, thread-safe locking, batch/parallel search and cleanup/persistence safeguards. |
| `main.py` | yakhyo/face-reidentification | Live webcam/video recognition loop and CLI baseline | Extended with mode selection, MaskInv/AdaFace/FocusFace/hybrid routing, mock airport data joining, kiosk POST integration and evaluation-oriented database paths. |
| `tools/mask_generation/add_mask.py` mask mapping block | fdbtrs/Masked-Face-Recognition-KD | Synthetic-mask template landmark mapping and homography approach | Modified for fixed ArcFace-aligned 112x112 APFF images, local mask tinting and recursive dataset tree processing. |

## Project Wrapper / Integration Code Dependent On External Repositories

| Local file | External source | Classification | Student contribution |
|---|---|---|---|
| `models/adaface.py` | mk-minchul/AdaFace | Project wrapper/integration code | Connects the external AdaFace model builder/checkpoint format to SmartIdentity aligned crops, embeddings, FAISS and evaluation. |
| `models/focusface.py` | NetoPedro/FocusFace | Project wrapper/integration code | Imports external FocusFace model code and exposes a stable normalized embedding API for backend/evaluation comparison. |
| `models/maskinv.py` | fdbtrs/Masked-Face-Recognition-KD / MaskInv | Project wrapper/integration code | Loads the external MaskInv ONNX recognizer and exposes a SmartIdentity aligned-crop embedding API. |

## Original Project Code Inspired By External Sources

| Local file / area | External reference | Classification | Student contribution |
|---|---|---|---|
| `tools/dataset_preparation/prepare_apff.py` | Arab Public Figures Facial Recognition dataset | Original project code inspired by external dataset requirements | Implements project-specific detection, alignment, identity grouping and split-preparation logic rather than copying source code. |
| `client/src` localisation UI | i18next-browser-languageDetector considered/reference option | Original project code; library not implemented | Uses local/static language handling rather than importing the browser language detector. |
| `evaluation/*.py` | Model wrappers and APFF evaluation setup | Original project evaluation orchestration | Runs local recognition experiments and writes result CSVs/plots using the implemented wrappers. |

## External Vendor / Reference Folders

| Folder | Source | Status | Required at runtime? | Submission caution |
|---|---|---|---|---|
| `external/adaface-test` | AdaFace-related external code/reference copy | External vendor/reference folder; exact local modifications require manual check | Required for AdaFace wrapper if it imports `external/adaface-test/AdaFace/net.py` | Submit only if MIT licence and model-weight terms are acceptable and file-size rules allow it. |
| `external/FocusFace` | https://github.com/NetoPedro/FocusFace | External vendor/reference folder | Required for FocusFace wrapper because `models/focusface.py` imports `model.FocusFace` from this folder | Submit only if MIT licence and model-weight terms are acceptable. |
| `external/Masked-Face-Recognition-KD` | https://github.com/fdbtrs/Masked-Face-Recognition-KD | External vendor/reference folder | Not directly imported by the current wrapper, but relevant for provenance/reference and synthetic-mask setup | CC BY-NC 4.0; do not redistribute until university supporting-material rules are checked. |

## Dataset / External Assets

| Asset | Local location | Source / status | Ownership note | Submission caution |
|---|---|---|---|---|
| Arab Public Figures Facial Recognition dataset-derived images | `datasets/` | Kaggle dataset by Ashraf Khalil / APFF | External public-figure dataset used for offline gallery/test evaluation and prototyping | Do not submit images unless Kaggle licence and university rules allow redistribution. |
| Evaluation CSVs and plots | `evaluation/results_*.csv`, `evaluation/plots/` | Generated locally from project evaluation scripts | Generated project results, but based on external dataset/model outputs | Keep as evidence if allowed; document dataset and model limitations. |
| FAISS vector-search library | `database/face_db.py`, `requirements.txt` | External library by Facebook Research | Used for embedding similarity search; project code wraps it in model-specific runtime galleries | Cite FAISS and treat generated indexes as local artefacts rather than original source code. |
| SCRFD/ArcFace weights | `weights/det_10g.onnx`, `weights/w600k_*.onnx` | External pretrained model weights | Required for detector/ArcFace modes | Check download source and redistribution permission. |
| AdaFace weights | `weights/adaface_ir18_webface4m.ckpt` | External pretrained checkpoint | Required for AdaFace mode/evaluation | Check checkpoint licence/source before submitting. |
| FocusFace weights | `weights/focus_face*.mdl` | External pretrained checkpoint | Required for FocusFace mode/evaluation | Check checkpoint licence/source before submitting. |
| MaskInv weights | `weights/maskinv/*` | External pretrained ONNX model | Required for MaskInv mode/evaluation | Check CC BY-NC / model-weight terms before submitting. |
| FAISS databases | `database/face_database*` if present | Generated local artefacts | Generated identity galleries from local data and selected recognizer | Usually reproducible artefacts; consider excluding unless needed for demonstration. |

## Unclear / Manual Review Required

- Confirm exact redistribution terms for all pretrained weights and checkpoints.
- Confirm whether `external/adaface-test` is a clean AdaFace copy, a modified local test copy, or a mixture.
- Confirm whether the university wants external vendor folders submitted or only cited.
- Confirm whether the Kaggle APFF dataset files may be included in supporting material.
- Confirm whether generated `datasets/`, `weights/` and FAISS database folders exceed submission size limits.
- Confirm whether any future frontend dependency changes add actual `i18next` or `i18next-browser-languageDetector` usage.
- Confirm whether `FINAL_SUBMISSION_CHECKLIST.md` exists in the intended submission set; it was listed for audit but was not part of the files edited here.
