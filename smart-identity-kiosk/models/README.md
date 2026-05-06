# Model Wrappers

This folder contains the stable project-facing wrappers for external face detection and recognition models.

- `scrfd.py` detects faces and landmarks using SCRFD ONNX weights.
- `arcface.py` extracts ArcFace embeddings from aligned face crops.
- `adaface.py` wraps the external AdaFace model builder and checkpoint format.
- `focusface.py` wraps the external FocusFace implementation.
- `maskinv.py` wraps the MaskInv ONNX recognizer.

The wrappers integrate external models into the SmartIdentity kiosk, FAISS database and evaluation flows. They should not be described as newly invented model architectures.

