import os, sys, json, time, random, csv
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2

# Ensure project imports work when running from tools/archive/.
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from models import ArcFace, MaskInv
from utils.helpers import reference_alignment  # 5x2 canonical landmarks


def l2norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def apply_lower_face_occlusion(aligned_bgr: np.ndarray, frac: float = 0.45, color=None) -> np.ndarray:
    """Simple synthetic mask: cover bottom frac of the aligned face."""
    h, w = aligned_bgr.shape[:2]
    y0 = int(h * (1.0 - frac))
    out = aligned_bgr.copy()
    if color is None:
        color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
    cv2.rectangle(out, (0, y0), (w, h), color, thickness=-1)
    return out


def embed_image(model_name: str, arc: ArcFace, mki: MaskInv, img_path: str, masked: bool) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read: {img_path}")

    if img.shape[0] != 112 or img.shape[1] != 112:
        img = cv2.resize(img, (112, 112))

    if masked:
        img = apply_lower_face_occlusion(img)

    # aligned_112 images are already in canonical pose, so reuse canonical landmarks
    kps = reference_alignment.copy().astype(np.float32)

    if model_name == "arcface":
        emb = arc.get_embedding(img, kps, normalized=True)
        return emb.astype(np.float32)

    emb = mki.get_embedding(img, kps)  # already normalized
    return emb.astype(np.float32)


def compute_metrics(genuine: np.ndarray, impostor: np.ndarray):
    genuine = genuine.astype(np.float32)
    impostor = impostor.astype(np.float32)

    scores = np.concatenate([genuine, impostor])
    thresholds = np.unique(np.round(scores, 6))
    thresholds.sort()

    fars = np.array([(impostor >= t).mean() for t in thresholds], dtype=np.float32)
    tars = np.array([(genuine >= t).mean() for t in thresholds], dtype=np.float32)

    frrs = 1.0 - tars
    idx = int(np.argmin(np.abs(fars - frrs)))
    eer = float((fars[idx] + frrs[idx]) / 2.0)
    eer_thr = float(thresholds[idx])

    def tar_at_far(target_far: float):
        # choose the LOWEST threshold that already satisfies FAR <= target_far
        ok = np.where(fars <= target_far)[0]
        if len(ok) == 0:
            return None, None
        j = int(ok[0])
        return float(tars[j]), float(thresholds[j])

    tar_1e2, thr_1e2 = tar_at_far(1e-2)
    tar_1e3, thr_1e3 = tar_at_far(1e-3)

    return {
        "eer": eer,
        "eer_threshold": eer_thr,
        "tar@far1e-2": tar_1e2,
        "thr@far1e-2": thr_1e2,
        "tar@far1e-3": tar_1e3,
        "thr@far1e-3": thr_1e3,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./datasets/processed/apff_clean", help="processed dataset root")
    ap.add_argument("--split", default="split_seed42.json", help="split file name inside splits/")
    ap.add_argument("--arc-weight", default="./weights/w600k_mbf.onnx")
    ap.add_argument("--maskinv-weight", default="./weights/maskinv/maskinv_hg.onnx")
    ap.add_argument("--model", choices=["arcface", "maskinv"], required=True)
    ap.add_argument("--scenario", choices=["uu", "um", "mm", "all"], default="all",
                    help="uu=unmasked->unmasked, um=unmasked->masked, mm=masked->masked")
    ap.add_argument("--impostor-samples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=None,
                    help="optional open-set threshold for Unknown; if not set, closed-set Top-1 only")

    # Directory used to save archived evaluation outputs.
    ap.add_argument("--out-dir", type=str, default="./results/apff_eval", help="Directory to save outputs")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = Path(args.dataset)
    split_path = dataset / "splits" / args.split
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split: {split_path}")

    # Timestamped output folder for this archived run.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{Path(args.dataset).name}_{args.model}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(split_path.read_text(encoding="utf-8"))
    enroll = split["enroll"]
    probe = split["probe"]

    arc = ArcFace(args.arc_weight)
    mki = MaskInv(args.maskinv_weight)

    cache = {}

    def get_emb(path: str, masked: bool) -> np.ndarray:
        key = (args.model, path, masked)
        if key in cache:
            return cache[key]
        e = embed_image(args.model, arc, mki, path, masked)
        cache[key] = e
        return e

    scenarios = ["uu", "um", "mm"] if args.scenario == "all" else [args.scenario]

    summary_rows = []

    for sc in scenarios:
        enroll_masked = (sc == "mm")
        probe_masked = (sc in ("um", "mm"))

        ids = sorted(enroll.keys())

        # enroll matrix
        E = []
        for pid in ids:
            p0 = enroll[pid][0]
            E.append(get_emb(p0, enroll_masked))
        E = np.stack(E, axis=0)
        E = np.array([l2norm(x) for x in E], dtype=np.float32)

        id_to_index = {pid: i for i, pid in enumerate(ids)}

        total = 0
        correct = 0
        unknown = 0

        genuine_scores = []
        impostor_scores = []

        # build probe list once
        all_probe = []
        for pid in ids:
            for p in probe.get(pid, []):
                all_probe.append((pid, p))

        t0 = time.time()
        for pid, p in all_probe:
            q = l2norm(get_emb(p, probe_masked))
            sims = E @ q
            best_i = int(np.argmax(sims))
            best_id = ids[best_i]
            best_sim = float(sims[best_i])

            # verification genuine score
            true_i = id_to_index[pid]
            true_sim = float(E[true_i] @ q)
            genuine_scores.append(true_sim)

            # verification impostor score
            wrong = random.choice(ids)
            while wrong == pid:
                wrong = random.choice(ids)
            impostor_scores.append(float(E[id_to_index[wrong]] @ q))

            total += 1

            # identification decision
            if args.threshold is not None and best_sim < args.threshold:
                unknown += 1
                continue

            if best_id == pid:
                correct += 1

        dt = time.time() - t0

        if args.threshold is None:
            top1 = correct / max(1, total)
        else:
            denom = max(1, (total - unknown))
            top1 = correct / denom

        genuine_scores = np.array(genuine_scores, dtype=np.float32)
        impostor_scores = np.array(impostor_scores, dtype=np.float32)

        # optionally subsample impostors
        if args.impostor_samples and len(impostor_scores) > args.impostor_samples:
            idx = np.random.choice(len(impostor_scores), size=args.impostor_samples, replace=False)
            impostor_scores = impostor_scores[idx]

        ver = compute_metrics(genuine_scores, impostor_scores) if len(impostor_scores) else {}

        # Save raw score arrays for downstream plots.
        np.save(run_dir / f"scores_{sc}_genuine.npy", genuine_scores)
        np.save(run_dir / f"scores_{sc}_impostor.npy", impostor_scores)

        # build summary row
        row = {
            "run_id": run_id,
            "dataset": str(dataset),
            "split": args.split,
            "model": args.model,
            "scenario": sc,
            "ids": len(ids),
            "probes": total,
            "time_sec": float(dt),
            "top1": float(top1),
            "unknown": int(unknown),
            "threshold": None if args.threshold is None else float(args.threshold),
            "eer": float(ver.get("eer", np.nan)),
            "eer_threshold": float(ver.get("eer_threshold", np.nan)),
            "tar@far1e-2": ver.get("tar@far1e-2", None),
            "thr@far1e-2": ver.get("thr@far1e-2", None),
            "tar@far1e-3": ver.get("tar@far1e-3", None),
            "thr@far1e-3": ver.get("thr@far1e-3", None),
        }
        summary_rows.append(row)

        # print
        print("\n==============================")
        print(f"MODEL: {args.model} | SCENARIO: {sc} (enroll_masked={enroll_masked}, probe_masked={probe_masked})")
        print(f"IDs: {len(ids)} | probes: {total} | time: {dt:.2f}s")
        if args.threshold is None:
            print(f"Top-1 (closed-set): {top1:.4f}")
        else:
            print(f"Top-1 (open-set): {top1:.4f} | Unknown rejected: {unknown}/{total} | thr={args.threshold:.3f}")

        if ver:
            print(f"EER: {ver['eer']:.4f} @ thr={ver['eer_threshold']:.3f}")
            if ver["tar@far1e-2"] is not None:
                print(f"TAR@FAR=1e-2: {ver['tar@far1e-2']:.4f} @ thr={ver['thr@far1e-2']:.3f}")
            if ver["tar@far1e-3"] is not None:
                print(f"TAR@FAR=1e-3: {ver['tar@far1e-3']:.4f} @ thr={ver['thr@far1e-3']:.3f}")

    # Save summary files for this archived run.
    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"

    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nDone.")
    print(f"Saved results to: {run_dir.resolve()}")
    print(f"- {summary_csv.name}")
    print(f"- {summary_json.name}")
    print("- scores_*.npy (raw genuine/impostor arrays per scenario)")


if __name__ == "__main__":
    main()
