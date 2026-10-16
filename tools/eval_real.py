"""Score static-mode output on the real clips against data/real/ground_truth.json.

A detection matches a ground-truth change if the types agree and the box centre is within
`radius` (fraction of the image) of the annotated position; a moved object needs both ends.
A moved object reported as a separate removed + added pair at the right places counts as
found, but with the wrong type (listed under 'split').

    python tools/eval_real.py --run out/real
"""
import argparse
import json
from pathlib import Path

import numpy as np


def centre(box, size):
    return np.array([(box[0] + box[2]) / 2 / size[0], (box[1] + box[3]) / 2 / size[1]])


def near(p, q, r):
    return float(np.linalg.norm(np.asarray(p) - np.asarray(q))) < r


def score_pair(gt, det, size, r):
    pts = []
    for d in det:
        if d["type"] == "moved":
            pts.append(("moved", centre(d["from"]["box_px"], size), centre(d["to"]["box_px"], size)))
        else:
            pts.append((d["type"], centre(d["box_px"], size), None))
    used, tp, fn, split, opt_hit = set(), 0, 0, 0, 0
    for g in gt:
        hit = None
        for i, (t, a, b) in enumerate(pts):
            if i in used:
                continue
            if g["type"] == "moved" and t == "moved" and near(a, g["from"], r) and near(b, g["to"], r):
                hit = [i]
            elif g["type"] != "moved" and t == g["type"] and near(a, g["at"], r):
                hit = [i]
            if hit:
                break
        if not hit and g["type"] == "moved":  # removed + added at the right places
            ri = [i for i, (t, a, _) in enumerate(pts) if i not in used and t == "removed" and near(a, g["from"], r)]
            ai = [i for i, (t, a, _) in enumerate(pts) if i not in used and t == "added" and near(a, g["to"], r)]
            if ri and ai:
                hit = [ri[0], ai[0]]
                split += 1
        if hit:
            used |= set(hit)
            tp += 1
            opt_hit += bool(g.get("optional"))
        elif not g.get("optional"):
            fn += 1
    fp = len(pts) - len(used)
    return tp, fp, fn, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/real")
    ap.add_argument("--gt", default="data/real/ground_truth.json")
    a = ap.parse_args()
    gt = json.loads(Path(a.gt).read_text())
    rows, T = [], np.zeros(4, int)
    for key, changes in gt["pairs"].items():
        b, c = key.split("->")
        d = Path(a.run) / f"{b}_vs_{c}"
        if not (d / "changes.json").exists():
            print(f"{key}: no output in {d}, skipped")
            continue
        det = json.loads((d / "changes.json").read_text())["changes"]
        size = json.loads((d / "static_meta.json").read_text())["image_size"]
        s = np.array(score_pair(changes, det, size, gt["radius"]))
        T += s
        rows.append({"pair": key, "tp": int(s[0]), "fp": int(s[1]), "fn": int(s[2]), "split_moves": int(s[3]),
                     "detected": [f"{x['type']} {x['label']}" for x in det]})
        print(f"{key:16s} tp {s[0]}  fp {s[1]}  fn {s[2]}  split {s[3]}   {', '.join(rows[-1]['detected']) or '-'}")
    res = {"pairs": rows, "true_positives": int(T[0]), "false_positives": int(T[1]), "false_negatives": int(T[2]),
           "split_moves": int(T[3]), "precision": round(T[0] / max(T[0] + T[1], 1), 3),
           "recall": round(T[0] / max(T[0] + T[2], 1), 3)}
    print(f"total: precision {res['precision']}, recall {res['recall']} "
          f"({T[0]} tp, {T[1]} fp, {T[2]} fn; {T[3]} moves reported as removed + added)")
    (Path(a.run) / "eval_real.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
