"""Score a run on the synthetic scene against gt.json.

The output lives in the run's own room frame, so we first fit a similarity transform
from the estimated camera centres to the true ones (same frames, known indices).
A detection counts if its type matches and its centroid is within --radius of the
true one (for 'moved', both the start and end positions must match).

    python tools/eval_synthetic.py --gt data/synthetic/gt.json --run out/synthetic
"""
import argparse
import json
from pathlib import Path

import numpy as np


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    X, Y = src - ms, dst - md
    U, S, Vt = np.linalg.svd(Y.T @ X)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (X ** 2).sum()
    return s, R, md - s * R @ ms


def label_ok(pred, true):
    syn = {"box": {"box", "cardboard box", "suitcase", "basket"}, "ball": {"ball", "toy"},
           "chair": {"chair", "office chair", "stool"}}
    return pred in syn.get(true, {true})


def evaluate(gt_path, run_dir, radius=0.5):
    gt = json.loads(Path(gt_path).read_text())
    run = Path(run_dir)
    cams = json.loads((run / "cameras.json").read_text())
    det = json.loads((run / "changes.json").read_text())["changes"]
    src = np.array([c["center"] for c in cams])
    dst = []
    for c in cams:
        p = gt[c["rec"]][c["frame_index"]]
        dst.append(-np.array(p["R"]).T @ np.array(p["t"]))
    s, R, t = umeyama(src, np.array(dst))
    rmse = float(np.sqrt(((s * src @ R.T + t - np.array(dst)) ** 2).sum(1).mean()))

    def tf(x):
        return s * R @ np.asarray(x) + t

    def dist2d(a, b):
        return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))

    used, rows, tp_labels = set(), [], 0
    for g in gt["changes"]:
        hit = None
        for i, d in enumerate(det):
            if i in used or d["type"] != g["type"]:
                continue
            if g["type"] == "moved":
                ok = (dist2d(tf(d["from"]["centroid"]), g["from"]) < radius and
                      dist2d(tf(d["to"]["centroid"]), g["to"]) < radius)
                err = dist2d(tf(d["to"]["centroid"]), g["to"])
            else:
                err = dist2d(tf(d["centroid"]), g["centroid"])
                ok = err < radius
            if ok and (hit is None or err < hit[1]):
                hit = (i, err)
        if hit:
            used.add(hit[0])
            lab = det[hit[0]]["label"]
            tp_labels += label_ok(lab, g["label"])
            rows.append((g["type"], g["label"], f"found as '{lab}', error {hit[1]:.2f} m"))
        else:
            rows.append((g["type"], g["label"], "missed"))
    tp = len(used)
    res = {
        "pose_alignment_rmse_m": round(rmse, 4), "scale_to_gt": round(float(s), 3),
        "detections": len(det), "ground_truth": len(gt["changes"]), "true_positives": tp,
        "precision": round(tp / max(len(det), 1), 3), "recall": round(tp / len(gt["changes"]), 3),
        "label_accuracy_on_tp": round(tp_labels / max(tp, 1), 3),
        "false_positives": [f"{d['type']} {d['label']}" for i, d in enumerate(det) if i not in used],
        "per_change": [" | ".join(r) for r in rows],
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/synthetic/gt.json")
    ap.add_argument("--run", default="out/synthetic")
    ap.add_argument("--radius", type=float, default=0.5)
    a = ap.parse_args()
    res = evaluate(a.gt, a.run, a.radius)
    print(json.dumps(res, indent=2))
    (Path(a.run) / "eval.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
