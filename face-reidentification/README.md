<<<<<<< HEAD
# Smart Identity Recognition System for Airports

Undergraduate Final Year Project implementation for BSc Computer Science with Artificial Intelligence.

This project demonstrates a smart airport kiosk that recognises a traveller from a live camera feed, joins the recognised identity to simulated passport, boarding pass, and flight records, and displays the result in a React kiosk interface.

## Main Components

- `server/` - FastAPI kiosk backend, webcam stream, SSE event stream, recognition worker, and traveller data joining.
- `client/` - React/Vite kiosk frontend.
- `models/` - wrappers for SCRFD, ArcFace, AdaFace, FocusFace, and MaskInv.
- `database/face_database/` - generated FAISS indexes and metadata for each recognition model.
- `data/` - simulated passenger, boarding pass, and flight JSON files.
- `evaluation/` - evaluation scripts, result CSVs, and generated plots for report evidence.
- `assets/faces/` - enrolment/demo face images used to build recognition databases.
- `external/` - copied third-party/reference model repositories used by wrappers or retained for research traceability.
- `weights/` - required model weights. These are large and should not be committed.

## Backend Setup

From PowerShell:

```powershell
cd C:\Users\orjoa\OneDrive\Desktop\project\SmartIdentity\face-reidentification
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

The backend now starts gracefully if the webcam, data files, model weights, or FAISS databases are unavailable. Check `/health` for the exact readiness status.

## Frontend Setup

From PowerShell:

```powershell
cd C:\Users\orjoa\OneDrive\Desktop\project\SmartIdentity\face-reidentification\client
npm install
npm run dev
```

Open:

- Frontend: http://localhost:5173

The Vite dev server proxies `/kiosk`, `/health`, `/admin`, and `/traveller` requests to `http://127.0.0.1:8000`.

## Required Model Weights

The current kiosk backend expects:

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

- `data/passports.json`
- `data/boarding_passes.json`
- `data/flights.json`

Joining logic:

1. Recognised face label is normalised and matched to `passports[*].full_name`.
2. The passport number is matched to `boarding_passes[*].passport_number`.
3. The boarding pass flight number is matched to `flights[*].flight_number`.

Do not use real passenger data in these files for the final submission. Keep it clearly described as mock/simulated data in the report and viva.

## Troubleshooting

- If `/health` returns `camera.ready: false`, check that a webcam is connected or set `CAMERA_INDEX`.
- If `/health` shows a model as missing, restore the corresponding file in `weights/`.
- If `/health` shows a database as missing, rebuild or restore the matching `database/face_database/<model>/` folder.
- If the frontend shows an event stream error, start the backend first and confirm `http://127.0.0.1:8000/kiosk/events` responds.
- If the camera image does not load in the browser, open `http://127.0.0.1:8000/kiosk/video` directly.
- If a traveller is recognised but no details appear, check spelling consistency between the face database label and `data/passports.json`.

## Evaluation Material

Evaluation scripts and saved result CSVs live in `evaluation/`. The plotting script reads the saved CSVs and writes report-ready summaries/plots to `evaluation/plots/`.

```powershell
cd C:\Users\orjoa\OneDrive\Desktop\project\SmartIdentity\face-reidentification
.\venv\Scripts\Activate.ps1
python evaluation\plot_results.py
```

The evaluation code is separate from the live kiosk runtime and should not modify the runtime FAISS databases.

## Cleaned External Repositories

Copied model/research repositories are grouped under `external/`:

- `external/adaface-test/` - AdaFace reference repository used by `models/adaface.py`.
- `external/FocusFace/` - FocusFace reference repository used by `models/focusface.py`.
- `external/Masked-Face-Recognition-KD/` - MaskInv/knowledge-distillation reference repository retained as research/vendor evidence.

The current run commands are unchanged after this cleanup.

## Final University Supporting Material Notes

For final submission or viva support, include:

- a short architecture diagram showing frontend, FastAPI backend, model wrappers, FAISS databases, and mock airport data;
- screenshots of `/health`, the kiosk scanning view, and a successful recognised traveller screen;
- evaluation tables/plots from `evaluation/plots/`;
- a clear statement that passenger records are simulated;
- a note that `weights/`, datasets, logs, generated databases, and built frontend files are excluded from Git/submission unless specifically required by the university.

---

The original upstream face re-identification README is retained below for reference.

# Real-Time Face Re-Identification with FAISS, ArcFace & SCRFD

![Downloads](https://img.shields.io/github/downloads/yakhyo/face-reidentification/total)
[![GitHub Repo stars](https://img.shields.io/github/stars/yakhyo/face-reidentification)](https://github.com/yakhyo/face-reidentification/stargazers)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/face-reidentification)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-yakhyo%2Fface--reidentification-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/yakhyo/face-reidentification)



<!--
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5>
-->

<video controls autoplay loop src="https://github.com/user-attachments/assets/16d63ac6-57a4-464b-8d82-948e1a06b6e3" muted="false" width="100%"></video>

## Key Features

- **Real-Time Face Recognition**: Live processing from webcam or video files with optimized performance
- **Production-Ready Accuracy**: State-of-the-art SCRFD + ArcFace models for reliable detection and recognition
- **Intelligent Processing**: Smart batch optimization that adapts to face count for optimal performance
- **Scalable Vector Search**: FAISS-powered similarity search with thread-safe database operations
- **Flexible Model Selection**: Multiple model sizes available to balance speed vs accuracy for your use case
- **Robust Resource Management**: Built-in memory leak prevention and automatic cleanup for long-running applications

## Performance Optimizations

- **Intelligent Batch Processing**: Automatically switches between sequential and parallel processing based on face count
- **Thread-Safe Database**: Robust FAISS integration with proper resource management
- **Memory Efficient**: Context managers ensure proper cleanup and prevent memory leaks
- **Optimized for Video**: Designed for typical video scenarios (1-5 faces per frame)

> [!NOTE]
> Place your target face images in the `assets/faces/` directory. The filenames will be used as identity labels during recognition.

## Architecture

The system combines three powerful components:
1. **SCRFD** ([Paper](https://arxiv.org/abs/2105.04714)): Efficient face detection
2. **ArcFace** ([Paper](https://arxiv.org/abs/1801.07698)): Robust face recognition
3. **FAISS**: Fast similarity search for face re-identification

### Available Models

| Category | Model | Size | Description |
|----------|-------|------|-------------|
| Detection | SCRFD 500M | 2.41 MB | Lightweight face detection |
| Detection | SCRFD 2.5G | 3.14 MB | Balanced performance |
| Detection | SCRFD 10G | 16.1 MB | High accuracy |
| Recognition | ArcFace MobileFace | 12.99 MB | Mobile-friendly recognition |
| Recognition | ArcFace ResNet-50 | 166 MB | High-accuracy recognition |

## Project Structure

```
├── assets/
│   ├── demo.mp4
│   |── in_video.mp4
|   └── faces/           # Place target face images here
│     ├── face1.jpg
│     ├── face2.jpg
│     └── ...
├── database/           # FAISS database implementation
├── models/            # Neural network models
├── weights/           # Model weights (download required)
├── utils/            # Helper functions
├── main.py           # Main application entry
└── requirements.txt  # Dependencies
```

## Getting Started

### Prerequisites

> [!IMPORTANT]
> Make sure you have Python 3.7+ installed on your system.

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yakyo/face-reidentification.git
cd face-reidentification
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download model weights:**

<details>
<summary>Click to see download links 📥</summary>

| Model | Download Link | Size |
|-------|--------------|------|
| SCRFD 500M | [det_500m.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_500m.onnx) | 2.41 MB |
| SCRFD 2.5G | [det_2.5g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_2.5g.onnx) | 3.14 MB |
| SCRFD 10G | [det_10g.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/det_10g.onnx) | 16.1 MB |
| ArcFace MobileFace | [w600k_mbf.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx) | 12.99 MB |
| ArcFace ResNet-50 | [w600k_r50.onnx](https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_r50.onnx) | 166 MB |

</details>

**Quick download (Linux/Mac):**
```bash
sh download.sh
```

4. **Add target faces:**
Place face images in `assets/faces/` directory. The filename will be used as the person's identity.

## Usage

### Basic Usage
```bash
python main.py --source assets/in_video.mp4
```

### Command Line Arguments

> [!TIP]
> Use these arguments to customize the recognition behavior:

```bash
usage: main.py [-h] [--det-weight DET_WEIGHT] [--rec-weight REC_WEIGHT] 
               [--similarity-thresh SIMILARITY_THRESH] [--confidence-thresh CONFIDENCE_THRESH]
               [--faces-dir FACES_DIR] [--source SOURCE] [--max-num MAX_NUM]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--det-weight` | Detection model path | `weights/det_10g.onnx` |
| `--rec-weight` | Recognition model path | `weights/w600k_r50.onnx` |
| `--similarity-thresh` | Face similarity threshold | `0.4` |
| `--confidence-thresh` | Detection confidence threshold | `0.5` |
| `--faces-dir` | Target faces directory | `assets/faces` |
| `--source` | Video source (file or camera index) | `0` |
| `--max-num` | Max faces per frame | `5` |
| `--db-path` | Custom database storage location | `./database/face_database` |
| `--update-db` | Force rebuild face database | `False` |
| `--output` | Specify output video path | `output_video.mp4` |

## Technical Notes

### Database Behavior
- Face database automatically saves/loads from disk
- Intelligent processing: sequential for <10 faces, parallel for larger batches
- Thread-safe operations with proper resource cleanup

### Performance Tips
- **Typical video processing**: Sequential processing is automatically used for 1-5 faces per frame
- **Threading optimization**: Parallel processing kicks in only for 10+ faces (rare in normal video)
- **GPU acceleration**: Install `onnxruntime-gpu` for faster model inference
- **Memory efficiency**: Context managers ensure proper resource cleanup in long-running applications
- **Database persistence**: Face database automatically saves/loads, avoiding rebuild on restart

## References

> [!NOTE]
> This project builds upon the following research:

1. [SCRFD: Sample and Computation Redistribution for Efficient Face Detection](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
2. [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)

<!-- ## Support

If you find this project useful, please consider giving it a star on GitHub! -->
=======
>>>>>>> fe7f3b857ae02bee28786004c26d9f82f583fc96

