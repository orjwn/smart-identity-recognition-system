# Evaluation Evidence

This folder contains the offline evaluation scripts, saved result CSVs and generated plots used as report evidence.

- `evaluate_*.py` scripts run model-specific APFF-derived evaluation passes.
- `results_*.csv` files are saved evaluation outputs and should not be regenerated before submission unless the report numbers are intentionally being updated.
- `plot_results.py` reads the saved CSVs and writes summary CSVs/plots to `plots/`.

The evaluation folder is separate from the live kiosk runtime. Evaluation scripts read `datasets/gallery`, `datasets/test` and `datasets/test_masked`; they do not modify the runtime FAISS databases in `database/face_database/`.

