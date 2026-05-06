# Third-Party Attribution

Project: Smart Identity Recognition System for Airports  
Student: Orjuwan Almotarafi  
Main code folder: `SmartIdentity/smart-identity-kiosk`

This file records external implementation references, model repositories, libraries and datasets used or considered by this project. It separates copied unchanged material, lightly modified external code, adapted code, project wrappers and original project code inspired by external sources.

## Attribution Categories Used

| Category | Meaning in this project |
|---|---|
| Copied unchanged from external source | The local block appears to preserve the external source without meaningful changes. |
| Copied and lightly modified from external source | The local block remains close to the external source but has small edits for imports, paths, logging, typing, configuration or compatibility. |
| Adapted from external source | The local block is strongly based on external code but was substantially modified for SmartIdentity requirements. |
| Project wrapper/integration code | The local file was written to connect an external model/reference implementation to this project's alignment, embedding, FAISS, backend, evaluation or kiosk pipeline. |
| Original project code inspired by external design pattern | The local code follows a general approach or dataset requirement but does not copy source code. |

## Source Table

| External source | URL | Licence found | Local files affected | Status in local project | What the source provides | Student contribution | Required at runtime? | Should it be submitted? | Manual checks still required |
|---|---|---|---|---|---|---|---|---|---|
| yakhyo/face-reidentification | https://github.com/yakhyo/face-reidentification | MIT indicated by repository/GitHub README badge; local cloned repository did not contain a standalone `LICENSE` file | `models/arcface.py`, `models/scrfd.py`, `database/face_db.py`, `utils/helpers.py`, `main.py`, retained lower README section | Lightly modified external code plus adapted baseline code | SCRFD detector wrapper, ArcFace ONNX embedding wrapper, FAISS gallery/search structure, helper functions, live recognition pipeline and upstream README material | Added logging/validation/integration edits, model-specific database handling, thread-safe/batch database operations, multi-recognizer routing, kiosk POST integration, mock airport data joining and evaluation-oriented database paths | Partly. Adapted local code is required; the external clone itself is not required unless re-comparing source | Do not submit the external repository unless licence and university rules are checked. Submit attribution and adapted local code | Confirm redistribution terms because no standalone licence file was found in the local clone |
| mk-minchul/AdaFace | https://github.com/mk-minchul/AdaFace | MIT | `models/adaface.py`, `external/adaface-test/AdaFace`, `weights/adaface_ir18_webface4m.ckpt`, evaluation scripts that call AdaFace | Project wrapper/integration code, with one lightly modified input-conversion block | AdaFace model architecture, checkpoint loading pattern and quality-aware face recognition model reference | Wrote a SmartIdentity wrapper that imports the external model builder, loads a local checkpoint, accepts aligned BGR crops and returns normalized embeddings for FAISS/evaluation | Yes if AdaFace mode/evaluation is used. `external/adaface-test/AdaFace` is imported by `models/adaface.py` | External folder/checkpoint should only be submitted if licence, model-weight terms and file-size rules allow it | Confirm checkpoint licence/source and whether model weights may be redistributed |
| fdbtrs/Masked-Face-Recognition-KD | https://github.com/fdbtrs/Masked-Face-Recognition-KD | Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) stated in README | `models/maskinv.py`, `tools/mask_generation/add_mask.py`, `external/Masked-Face-Recognition-KD`, `weights/maskinv/*`, masked evaluation outputs | Project wrapper/integration code plus adapted synthetic-mask mapping | Mask-invariant recognition reference implementation, trained model approach and synthetic mask template/mapping | Wrote a SmartIdentity ONNX wrapper for aligned 112x112 crops and adapted the synthetic-mask mapping for fixed aligned APFF test images, local tinting and recursive dataset processing | Yes if MaskInv mode/evaluation is used. External folder may be needed only for reference or reproducibility | CC BY-NC means do not redistribute for commercial use. Confirm with university whether external folder/weights should be submitted | Confirm exact weight licence and whether supporting material may include non-commercial code/weights |
| NetoPedro/FocusFace | https://github.com/NetoPedro/FocusFace | MIT | `models/focusface.py`, `external/FocusFace`, `weights/focus_face*.mdl`, FocusFace evaluation scripts | Project wrapper/integration code | FocusFace masked/occluded face recognition model code and architecture | Wrote a stable project-facing wrapper, image transforms and normalized embedding API for backend/evaluation comparison | Yes if FocusFace mode/evaluation is used. `external/FocusFace` is imported by `models/focusface.py` | External folder/checkpoints should only be submitted if model-weight terms and file-size rules allow it | Confirm trained-model weight licence/source separately from repository code |
| Facebook Research FAISS | https://github.com/facebookresearch/faiss | MIT | `database/face_db.py`, `requirements.txt`, runtime `database/face_database/<model>/` artefacts | External library used through Python package | Efficient vector similarity search over face embeddings | Integrated FAISS `IndexFlatIP` with model-specific metadata, persistence, locking and kiosk/evaluation search flows | Yes for runtime recognition databases and evaluation search | The package dependency may be installed from requirements; generated FAISS indexes should be treated as local artefacts | Confirm whether generated databases should be submitted or rebuilt locally |
| Arab Public Figures Facial Recognition dataset | https://www.kaggle.com/datasets/ashkhalil/arab-public-figures-facial-recognition | Kaggle page/search result indicates Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) | `datasets/`, `tools/dataset_preparation/prepare_apff.py`, `evaluation/results_*.csv`, evaluation plots | Dataset; `tools/dataset_preparation/prepare_apff.py` is original project code inspired by dataset preparation needs | Public-figure face images/videos for offline recognition prototyping and gallery/test identity experiments | Implemented local detection, alignment, identity grouping, split preparation, evaluation outputs and plotting around the dataset | Dataset files are required only to reproduce offline evaluation, not to run the kiosk demo with prepared identities | Do not submit dataset images unless Kaggle licence and university supporting-material rules allow it | Confirm Kaggle licence while logged in, consent/redistribution terms and file-size limits |
| i18next/i18next-browser-languageDetector | https://github.com/i18next/i18next-browser-languageDetector | MIT | No current source-code usage found in `client/src` or `client/package.json`; built `client/dist` contains unrelated browser globals text only | Considered/reference option only, not implemented | Browser language detection for i18next using sources such as navigator, storage, query string and cookies | Current SmartIdentity localisation appears to use local/static language handling rather than this library | No | No, because it is not currently implemented as a dependency | Confirm again before final submission if frontend dependencies change |

## Copied / Modified / Wrapper Summary

| Local file or block | Classification | Student modification identified |
|---|---|---|
| `utils/helpers.py` ArcFace landmark template values | Copied unchanged from external source | None to the landmark values; retained for compatibility with ArcFace alignment. |
| `models/arcface.py` | Copied and lightly modified from external source | Minor edits for logging, validation/error handling, output-shape checks and project alignment/database integration. |
| `models/scrfd.py` | Copied and lightly modified from external source | Minor edits for imports, type hints, local demo/debug helpers and backend/evaluation integration. |
| `utils/helpers.py` alignment, SCRFD decode and visualisation helpers | Copied and lightly modified from external source | Minor edits for typing, docstrings, comments, local style and reuse across wrappers/evaluation. |
| `database/face_db.py` | Adapted from external source | Substantially modified with model-specific database folders, thread-safe access, batch/parallel search, persistence safeguards and cleanup methods. |
| `main.py` | Adapted from external source | Substantially modified with multi-recognizer routing, MaskInv/AdaFace/FocusFace/hybrid paths, mock airport data joining, kiosk POST integration and evaluation-oriented database handling. |
| `models/adaface.py` file | Project wrapper/integration code | Connects AdaFace external model/checkpoint loading to SmartIdentity aligned crops, embeddings, FAISS and evaluation. |
| `models/adaface.py` `_to_input` block | Copied and lightly modified from external source | Adapted AdaFace input conversion from PIL/RGB to SmartIdentity aligned BGR crops. |
| `models/focusface.py` | Project wrapper/integration code | Imports external FocusFace model and exposes a stable normalized embedding API. |
| `models/maskinv.py` | Project wrapper/integration code | Loads MaskInv ONNX and exposes a SmartIdentity aligned-crop embedding API. |
| `tools/mask_generation/add_mask.py` MaskInv mapping block | Adapted from external source | Modified for fixed ArcFace-aligned 112x112 APFF images, local mask tinting and recursive dataset processing. |
| `tools/dataset_preparation/prepare_apff.py` | Original project code inspired by external dataset requirements | Implements project-specific detection, alignment, identity grouping and split preparation; no Kaggle code is claimed to be copied. |
| `README.md` retained lower upstream section | Copied and lightly modified external documentation | SmartIdentity README content was added above; retained upstream block may differ due to local formatting/encoding. |
| `evaluation/*.py` | Original project evaluation orchestration, as far as this audit found | Uses wrappers and external model outputs but no source-attribution comments were added because no copied external evaluation code was identified. |

## External Vendor / Reference Folders

The folders under `external/` are third-party/vendor/reference material and should not be described as original student code:

- `external/adaface-test`: AdaFace-related external code/reference folder. Manual check required to confirm whether it is a clean external copy or a modified local test copy.
- `external/FocusFace`: FocusFace external repository folder.
- `external/Masked-Face-Recognition-KD`: MaskInv/knowledge-distillation external repository folder.

These folders may contain copied external repository code. They should be cited and submitted only if the relevant licence, model-weight terms, file-size limits and university supporting-material rules allow it.

## Redistribution Caution

Licence not confirmed - do not redistribute until checked manually applies to any source where a standalone licence file, checkpoint licence, dataset redistribution permission or university supporting-material rule cannot be verified. In particular, model checkpoints, generated FAISS databases and dataset images should be treated as external artefacts, not original source code.

## Sources That Should Be Cited In The Report

- yakhyo/face-reidentification, because it is the baseline implementation adapted or lightly modified for SCRFD, ArcFace and FAISS-style recognition.
- AdaFace, because the project wraps and evaluates an external AdaFace recognizer.
- Masked-Face-Recognition-KD / MaskInv, because the project uses MaskInv as an external masked-face recognizer reference and adapts the synthetic-mask mapping.
- FocusFace, because the project imports the external FocusFace implementation as a comparison recognizer.
- Arab Public Figures Facial Recognition dataset, because it is used for offline evaluation/prototyping.
- FAISS, because it is the vector-search library used for runtime and evaluation similarity search.
- i18next browser language detector only if discussed as a considered localisation reference, not as implemented code.
