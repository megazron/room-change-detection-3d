"""Template sentences from geometry + labels.

Room frame: z up from the floor, x points the way the first 'before' frame looked,
y to its left. Distances come from monocular metric depth, so they are approximate.
"""
import numpy as np


def _region(c, lo, hi):
    f = (c[:2] - lo) / np.maximum(hi - lo, 1e-6)
    depth = "near" if f[0] < 0.33 else "far" if f[0] > 0.66 else "middle"
    side = "left" if f[1] > 0.66 else "right" if f[1] < 0.33 else ""
    if depth == "middle" and not side:
        return "the middle of the room"
    return f"the {depth}{'-' + side if side else ''} part of the room"


def _support(bmin):
    # lower parts of an object are hard to separate from the floor, so the detected
    # cluster often starts a bit above it; anything starting below 0.4 m is called "on the floor"
    if bmin[2] < 0.4:
        return "on the floor"
    return f"on a surface about {bmin[2]:.1f} m high"


def _size(bmin, bmax):
    e = np.sort((bmax - bmin)[:2])[::-1]
    h = bmax[2] if bmin[2] < 0.4 else bmax[2] - bmin[2]
    return f"about {e[0]:.1f} x {e[1]:.1f} m footprint, {h:.1f} m tall"


def _direction(d):
    parts = []
    n = np.linalg.norm(d[:2]) + 1e-9
    if abs(d[0]) > 0.3 * n:
        parts.append("away from the start point" if d[0] > 0 else "toward the start point")
    if abs(d[1]) > 0.3 * n:
        parts.append("to the left" if d[1] > 0 else "to the right")
    return " and ".join(parts) or "slightly"


def _near(p, others):
    best = None
    for lab, q in others:
        dist = np.linalg.norm(p[:2] - q[:2])
        if dist < 1.2 and (best is None or dist < best[0]):
            best = (dist, lab)
    return f", next to the {best[1]}" if best else ""


def describe_all(changes, room_xyz):
    lo, hi = np.percentile(room_xyz[:, :2], 2, 0), np.percentile(room_xyz[:, :2], 98, 0)
    anchors = []  # where things are in the 'after' room
    for ch in changes:
        if ch["type"] == "moved":
            anchors.append((ch["label"], ch["to"]["centroid"]))
        elif ch["type"] == "added":
            anchors.append((ch["label"], ch["cluster"]["centroid"]))
    for ch in changes:
        lab = ch["label"]
        hedge = " (low label confidence)" if ch["label_conf"] < 0.3 else ""
        if ch["type"] == "moved":
            a, b = ch["from"], ch["to"]
            d = b["centroid"] - a["centroid"]
            ch["displacement"] = d
            others = [x for x in anchors if x[1] is not b["centroid"]]
            ra, rb = _region(a["centroid"], lo, hi), _region(b["centroid"], lo, hi)
            where = f"within {ra}" if ra == rb else f"from {ra} to {rb}"
            ch["text"] = (f"{lab.capitalize()} moved approx. {np.linalg.norm(d[:2]):.1f} m {_direction(d)}, "
                          f"{where}{_near(b['centroid'], others)}{hedge}.")
        else:
            c = ch["cluster"]
            others = [x for x in anchors if x[1] is not c["centroid"]]
            where = _region(c["centroid"], lo, hi)
            size = _size(c["bbox_min"], c["bbox_max"])
            if ch["type"] == "added":
                ch["text"] = (f"New {lab} {_support(c['bbox_min'])} in {where}"
                              f"{_near(c['centroid'], others)}, {size}{hedge}.")
            else:
                ch["text"] = (f"{lab.capitalize()} removed: it was {_support(c['bbox_min'])} in {where}, "
                              f"{size}; that space is now empty{hedge}.")
        if ch.get("structure_like"):
            ch["text"] += " Label is a room structure; may be a depth artefact."
