import re
from pathlib import Path
import numpy as np

# plotting
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    pd = None


RESULT_GROUP_LABELS = {
    "apff_eval": "APFF (on-the-fly masks)",
    "apff_eval_rectfolder": "APFF (rect pre-masked folder)",
    "mydb_eval": "My DB",
}

# If you evaluated a pre-masked folder but used scenario "uu" to avoid double masking,
# we can relabel it for plotting clarity.
SCENARIO_REMAP = {
    "apff_eval_rectfolder": {"uu": "mm_rect"},
    # add template folder later: {"apff_eval_templatefolder": {"uu": "mm_template"}}
}

SCENARIO_ORDER = ["uu", "um", "mm", "mm_rect", "mm_template"]


def find_latest_runs(results_root: Path):
    """
    Find latest run directories per (group, model).
    A run dir contains summary.csv and scores_*.npy files.
    """
    latest = {}  # (group, model) -> run_dir
    for group_dir in results_root.iterdir():
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        # each run dir has name like apff_clean_arcface_YYYYMMDD_HHMMSS
        for run_dir in group_dir.iterdir():
            if not run_dir.is_dir():
                continue
            summary = run_dir / "summary.csv"
            if not summary.exists():
                continue
            # try to infer model from folder name
            m = None
            if "arcface" in run_dir.name.lower():
                m = "arcface"
            elif "maskinv" in run_dir.name.lower():
                m = "maskinv"
            if not m:
                continue
            key = (group, m)
            # choose lexicographically latest (your names include timestamp)
            if key not in latest or run_dir.name > latest[key].name:
                latest[key] = run_dir
    return latest


def read_summary(run_dir: Path):
    if pd is None:
        # minimal CSV loader fallback
        import csv
        rows = []
        with open(run_dir / "summary.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows
    else:
        return pd.read_csv(run_dir / "summary.csv")


def load_scores(run_dir: Path, scenario: str):
    g = np.load(run_dir / f"scores_{scenario}_genuine.npy")
    i = np.load(run_dir / f"scores_{scenario}_impostor.npy")
    return g.astype(np.float32), i.astype(np.float32)


def roc_points(genuine: np.ndarray, impostor: np.ndarray, n_thresh: int = 400):
    scores = np.concatenate([genuine, impostor])
    # use quantiles for stable thresholds
    qs = np.linspace(0.0, 1.0, n_thresh)
    thresh = np.quantile(scores, qs)

    fars = np.array([(impostor >= t).mean() for t in thresh], dtype=np.float32)
    tars = np.array([(genuine >= t).mean() for t in thresh], dtype=np.float32)
    return fars, tars


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def scenario_label(group: str, sc: str) -> str:
    return SCENARIO_REMAP.get(group, {}).get(sc, sc)


def plot_bars(df, out_path: Path, metric: str, title: str):
    # df expected columns: group, dataset_label, model, scenario, value
    groups = list(df["dataset_label"].unique())
    for ds in groups:
        sub = df[df["dataset_label"] == ds].copy()
        # order scenarios
        sub["scenario"] = sub["scenario"].apply(lambda s: s if s in SCENARIO_ORDER else s)
        sub["scenario"] = pd.Categorical(sub["scenario"], categories=SCENARIO_ORDER, ordered=True)
        sub = sub.sort_values("scenario")

        # pivot: scenario x model
        pv = sub.pivot_table(index="scenario", columns="model", values=metric, aggfunc="first")

        ax = pv.plot(kind="bar")
        ax.set_title(f"{title} — {ds}")
        ax.set_xlabel("Scenario")
        ax.set_ylabel(metric)
        ax.legend(title="Model")
        plt.tight_layout()

        file = out_path / f"{metric}_{re.sub(r'[^a-zA-Z0-9]+','_',ds.lower())}.png"
        plt.savefig(file, dpi=180)
        plt.close()
        print("Saved:", file)


def plot_roc_overlay(latest_runs: dict, results_root: Path, out_path: Path, group: str, scenario: str):
    """
    Overlay ROC for arcface vs maskinv for a given results group and scenario.
    """
    arc_dir = latest_runs.get((group, "arcface"))
    mki_dir = latest_runs.get((group, "maskinv"))
    if not arc_dir or not mki_dir:
        print(f"Skip ROC {group}/{scenario}: missing one of models.")
        return

    # some groups are relabeled; score file names still use original scenario keys
    # for rectfolder, you ran scenario 'uu' intentionally.
    sc_key = scenario

    try:
        g1, i1 = load_scores(arc_dir, sc_key)
        g2, i2 = load_scores(mki_dir, sc_key)
    except FileNotFoundError:
        print(f"Skip ROC {group}/{scenario}: score files not found.")
        return

    far1, tar1 = roc_points(g1, i1)
    far2, tar2 = roc_points(g2, i2)

    plt.figure()
    plt.plot(far1, tar1, label="arcface")
    plt.plot(far2, tar2, label="maskinv")
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.xlim(max(1e-4, float(min(far1.min(), far2.min()))), 1.0)
    plt.xlabel("FAR (log scale)")
    plt.ylabel("TAR")
    ds_label = RESULT_GROUP_LABELS.get(group, group)
    plt.title(f"ROC — {ds_label} — scenario {scenario_label(group, scenario)}")
    plt.legend()
    plt.tight_layout()

    file = out_path / f"roc_{group}_{scenario}.png"
    plt.savefig(file, dpi=180)
    plt.close()
    print("Saved:", file)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="./results", help="results folder")
    ap.add_argument("--out-dir", default="./results/plots", help="where to save plots")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    latest = find_latest_runs(results_root)
    if not latest:
        print("No runs found under:", results_root.resolve())
        return

    # Build a combined dataframe from latest summaries
    rows = []
    for (group, model), run_dir in latest.items():
        summ = read_summary(run_dir)
        if pd is None:
            # list[dict] fallback
            for r in summ:
                sc = scenario_label(group, r["scenario"])
                rows.append({
                    "group": group,
                    "dataset_label": RESULT_GROUP_LABELS.get(group, group),
                    "model": model,
                    "scenario": sc,
                    "top1": float(r["top1"]),
                    "eer": float(r["eer"]),
                    "tar@far1e-3": float(r["tar@far1e-3"]) if r["tar@far1e-3"] != "" else np.nan,
                })
        else:
            for _, r in summ.iterrows():
                sc = scenario_label(group, str(r["scenario"]))
                rows.append({
                    "group": group,
                    "dataset_label": RESULT_GROUP_LABELS.get(group, group),
                    "model": model,
                    "scenario": sc,
                    "top1": float(r["top1"]),
                    "eer": float(r["eer"]),
                    "tar@far1e-3": float(r["tar@far1e-3"]),
                })

    if pd is None:
        print("Please install pandas for plotting: pip install pandas")
        return

    df = pd.DataFrame(rows)

    # Bar charts
    plot_bars(df, out_dir, metric="top1", title="Top-1 Identification")
    plot_bars(df, out_dir, metric="tar@far1e-3", title="TAR @ FAR=1e-3")

    # ROC overlays (use scenario keys that exist in score files)
    # apff_eval has uu/um/mm
    for sc in ["uu", "um", "mm"]:
        plot_roc_overlay(latest, results_root, out_dir, group="apff_eval", scenario=sc)

    # rectfolder: you ran scenario uu but it represents mm_rect
    plot_roc_overlay(latest, results_root, out_dir, group="apff_eval_rectfolder", scenario="uu")

    print("\nAll plots saved in:", out_dir.resolve())


if __name__ == "__main__":
    main()
    