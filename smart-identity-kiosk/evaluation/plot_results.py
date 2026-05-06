import os
from typing import Optional, Dict, List

import pandas as pd
import matplotlib.pyplot as plt

"""
Generate report-ready charts from saved evaluation CSV files.

The script reads only CSV files inside evaluation/. It does not load or modify
the runtime FAISS databases used by the kiosk demo.
"""


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")


# Single-run files
RUN_FILES = {
    "AdaFace IR18": os.path.join(EVAL_DIR, "results_adaface_ir18.csv"),
    "ArcFace MBF": os.path.join(EVAL_DIR, "results_arcface_mbf.csv"),
    "ArcFace R50": os.path.join(EVAL_DIR, "results_arcface_r50.csv"),
    "Hybrid MBF + AdaFace": os.path.join(EVAL_DIR, "results_hybrid_mbf_adaface.csv"),
    "FocusFace (Unmasked)": os.path.join(EVAL_DIR, "results_focusface.csv"),
    "FocusFace (Masked)": os.path.join(EVAL_DIR, "results_focusface_masked.csv"),
    "MaskInv student (Unmasked)": os.path.join(EVAL_DIR, "results_maskinv_hg.csv"),
    "MaskInv student (Masked)": os.path.join(EVAL_DIR, "results_maskinv_hg_masked.csv"),
}

# Speed comparison across core models (unmasked clean runs)
SPEED_FILES = {
    "AdaFace IR18": os.path.join(EVAL_DIR, "results_adaface_ir18.csv"),
    "ArcFace MBF": os.path.join(EVAL_DIR, "results_arcface_mbf.csv"),
    "ArcFace R50": os.path.join(EVAL_DIR, "results_arcface_r50.csv"),
    "Hybrid MBF + AdaFace": os.path.join(EVAL_DIR, "results_hybrid_mbf_adaface.csv"),
    "FocusFace": os.path.join(EVAL_DIR, "results_focusface.csv"),
    "MaskInv student": os.path.join(EVAL_DIR, "results_maskinv_hg.csv"),
}

# FocusFace vs MaskInv comparisons
PAIRWISE_UNMASKED = {
    "FocusFace": os.path.join(EVAL_DIR, "results_focusface.csv"),
    "MaskInv student": os.path.join(EVAL_DIR, "results_maskinv_hg.csv"),
}

PAIRWISE_MASKED = {
    "FocusFace": os.path.join(EVAL_DIR, "results_focusface_masked.csv"),
    "MaskInv student": os.path.join(EVAL_DIR, "results_maskinv_hg_masked.csv"),
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        value = str(value).strip()
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def safe_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def summarize_csv(csv_path: str) -> Dict[str, float]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    required_base = {"predicted_name", "correct", "status", "inference_ms"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(f"{os.path.basename(csv_path)} missing columns: {sorted(missing_base)}")

    # support both normal and hybrid result schemas
    if "similarity" in df.columns:
        similarity_col = "similarity"
    elif "final_similarity" in df.columns:
        similarity_col = "final_similarity"
    else:
        raise ValueError(
            f"{os.path.basename(csv_path)} missing similarity column "
            f"(expected 'similarity' or 'final_similarity')"
        )

    total = len(df)

    ok_df = df[df["status"].astype(str).str.strip().str.lower() == "ok"].copy()
    error_count = total - len(ok_df)

    ok_df["predicted_name_clean"] = ok_df["predicted_name"].astype(str).str.strip().str.lower()
    ok_df["correct_bool"] = ok_df["correct"].apply(safe_bool)
    ok_df["similarity_num"] = ok_df[similarity_col].apply(safe_float)
    ok_df["inference_ms_num"] = ok_df["inference_ms"].apply(safe_float)

    unknown_count = int((ok_df["predicted_name_clean"] == "unknown").sum())
    correct_count = int(ok_df["correct_bool"].sum())
    accuracy = (correct_count / total * 100.0) if total > 0 else 0.0

    accepted_df = ok_df[
        (ok_df["predicted_name_clean"] != "unknown") &
        (ok_df["similarity_num"].notna())
    ]
    avg_similarity = float(accepted_df["similarity_num"].mean()) if not accepted_df.empty else 0.0

    infer_df = ok_df[ok_df["inference_ms_num"].notna()]
    avg_inference_ms = float(infer_df["inference_ms_num"].mean()) if not infer_df.empty else 0.0

    return {
        "total": int(total),
        "correct": int(correct_count),
        "accuracy": accuracy,
        "unknown": int(unknown_count),
        "errors": int(error_count),
        "avg_similarity": avg_similarity,
        "avg_inference_ms": avg_inference_ms,
    }

def build_summary(file_map: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for label, path in file_map.items():
        summary = summarize_csv(path)
        row = {"label": label, "file": os.path.basename(path)}
        row.update(summary)
        rows.append(row)
    return pd.DataFrame(rows)


def save_single_model_chart(row: pd.Series) -> None:
    labels = ["Accuracy (%)", "Unknown", "Avg Similarity", "Inference (ms)"]
    values = [
        row["accuracy"],
        row["unknown"],
        row["avg_similarity"],
        row["avg_inference_ms"],
    ]

    filename = (
        row["label"]
        .replace(" + ", "_plus_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .lower()
        + "_summary.png"
    )

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(row["label"])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close()


def save_metric_comparison(df: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(df["label"], df[metric])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close()


def print_metric_notes() -> None:
    print("\nMetric notes:")
    print("- accuracy: correct predictions divided by all rows in the CSV, including errors")
    print("- unknown: rows where the model rejected the face as Unknown")
    print("- errors: rows that could not be processed successfully")
    print("- avg_similarity: mean similarity for accepted non-Unknown predictions")
    print("- avg_inference_ms: mean per-image embedding/search time recorded by the evaluator")


def main() -> None:
    ensure_dir(PLOTS_DIR)

    # 1) each model run on its own
    runs_df = build_summary(RUN_FILES)
    runs_df.to_csv(os.path.join(PLOTS_DIR, "all_run_summaries.csv"), index=False)

    for _, row in runs_df.iterrows():
        save_single_model_chart(row)

    # 2) all-model speed comparison
    speed_df = build_summary(SPEED_FILES)
    speed_df.to_csv(os.path.join(PLOTS_DIR, "speed_comparison_summary.csv"), index=False)

    save_metric_comparison(
        speed_df,
        metric="avg_inference_ms",
        ylabel="Inference Time (ms)",
        title="All Models Speed Comparison",
        filename="all_models_speed_comparison.png",
    )

    # 3) FocusFace vs MaskInv student (unmasked)
    pair_unmasked_df = build_summary(PAIRWISE_UNMASKED)
    pair_unmasked_df.to_csv(os.path.join(PLOTS_DIR, "focusface_vs_maskinv_unmasked_summary.csv"), index=False)

    save_metric_comparison(
        pair_unmasked_df,
        metric="accuracy",
        ylabel="Accuracy (%)",
        title="FocusFace vs MaskInv student (Unmasked)",
        filename="focusface_vs_maskinv_unmasked_accuracy.png",
    )
    save_metric_comparison(
        pair_unmasked_df,
        metric="unknown",
        ylabel="Unknown Count",
        title="FocusFace vs MaskInv student (Unmasked)",
        filename="focusface_vs_maskinv_unmasked_unknown.png",
    )
    save_metric_comparison(
        pair_unmasked_df,
        metric="avg_similarity",
        ylabel="Average Similarity",
        title="FocusFace vs MaskInv student (Unmasked)",
        filename="focusface_vs_maskinv_unmasked_similarity.png",
    )

    # 4) FocusFace vs MaskInv student (masked)
    pair_masked_df = build_summary(PAIRWISE_MASKED)
    pair_masked_df.to_csv(os.path.join(PLOTS_DIR, "focusface_vs_maskinv_masked_summary.csv"), index=False)

    save_metric_comparison(
        pair_masked_df,
        metric="accuracy",
        ylabel="Accuracy (%)",
        title="FocusFace vs MaskInv student (Masked)",
        filename="focusface_vs_maskinv_masked_accuracy.png",
    )
    save_metric_comparison(
        pair_masked_df,
        metric="unknown",
        ylabel="Unknown Count",
        title="FocusFace vs MaskInv student (Masked)",
        filename="focusface_vs_maskinv_masked_unknown.png",
    )
    save_metric_comparison(
        pair_masked_df,
        metric="avg_similarity",
        ylabel="Average Similarity",
        title="FocusFace vs MaskInv student (Masked)",
        filename="focusface_vs_maskinv_masked_similarity.png",
    )

    print("\nSaved plots to:")
    print(PLOTS_DIR)

    print("\nSingle-run summaries:")
    print(runs_df[[
        "label", "total", "correct", "accuracy",
        "unknown", "errors", "avg_similarity", "avg_inference_ms"
    ]].to_string(index=False))

    print("\nSpeed comparison:")
    print(speed_df[[
        "label", "avg_inference_ms"
    ]].to_string(index=False))
    print_metric_notes()


if __name__ == "__main__":
    main()
