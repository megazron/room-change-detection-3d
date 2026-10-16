"""Build the README GIFs from the real clips and the synthetic benchmark.

    python tools/make_readme_visuals.py [--only raw,flicker,timeline,orbit,synthetic]

Needs `make real` (plates in out/real/cache, changes in out/real/*/changes.json) and, for the
synthetic orbit, `make demo` (SfM/depth cache in out/demo/cache). Writes to docs/.
"""
import argparse
import json
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rayvote.static import (make_models, normalise_exposure, person_masks, process_clip,  # noqa: E402
                            read_samples)

REAL = ROOT / "data/real"
OUT_REAL = ROOT / "out/real"
DOCS = ROOT / "docs"
COL = {"removed": (214, 39, 40), "added": (44, 160, 44), "moved_from": (255, 159, 28), "moved_to": (123, 44, 191)}
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(_FONT_B if bold else _FONT, size)
    except OSError:
        return ImageFont.load_default()


def save_gif(frames, path, fps=10, durations=None, colors=128):
    """frames: list of HxWx3 uint8. One global palette (from a frame sample) keeps colours stable."""
    ims = [Image.fromarray(f) for f in frames]
    step = max(1, len(ims) // 40)
    probe = np.concatenate([np.asarray(im)[::2, ::2] for im in ims[::step]], 0)
    # small saturated marks get merged away by median cut: give each marker colour its own block
    w = probe.shape[1]
    sw = np.concatenate([np.full((48, w, 3), c, np.uint8) for c in list(COL.values()) + [(255, 255, 255)]], 0)
    probe = np.concatenate([probe, sw], 0)
    pal = Image.fromarray(probe).quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
    dur = durations or [int(1000 / fps)] * len(q)
    q[0].save(path, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True, disposal=1)
    print(f"  {path.relative_to(ROOT)}: {len(q)} frames, {path.stat().st_size / 1e6:.2f} MB")


def label_bar(w, text, h=24, size=14, bg=(30, 30, 30), fg=(255, 255, 255), bold=True):
    im = Image.new("RGB", (w, h), bg)
    ImageDraw.Draw(im).text((8, (h - size) // 2 - 1), text, font=font(size, bold), fill=fg)
    return np.asarray(im)


def caption_bar(w, lines, h, size=13):
    im = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(im)
    y = 6
    for ln, bold in lines:
        d.text((8, y), ln, font=font(size, bold), fill=(20, 20, 20))
        y += size + 5
    return np.asarray(im)


# ------------------------------------------------------------------ real-clip helpers

_clips = {}


def clip(i):
    """Cached static-mode result for clipNN (plate, depth, never)."""
    if i not in _clips:
        models = make_models("cuda" if _cuda() else "cpu")
        _clips[i] = process_clip(REAL / f"clip{i:02d}.mp4", OUT_REAL / "cache", models, lambda *a: None)
    return _clips[i]


def _cuda():
    import torch
    return torch.cuda.is_available()


def plate_like(i, ref=1):
    """Plate of clip i with exposure matched to clip `ref`, as in the comparison."""
    a, b = clip(ref), clip(i)
    if i == ref:
        return a["plate"].copy()
    return normalise_exposure(b["plate"], a["plate"], ~(a["never"] | b["never"]))


def changes(pair):
    return json.loads((OUT_REAL / pair / "changes.json").read_text())["changes"]


def _dashed_rect(d, box, col, w=2, dash=8):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        n = max(1, int(np.hypot(bx - ax, by - ay) // dash))
        for k in range(0, n, 2):
            t0, t1 = k / n, min(1, (k + 1) / n)
            d.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)],
                   fill=col, width=w)


def _arrow(d, a, b, col=(255, 255, 255), w=4):
    a, b = np.array(a, float), np.array(b, float)
    v = b - a
    L = np.linalg.norm(v) + 1e-6
    u, n = v / L, np.array([-v[1], v[0]]) / L
    tip = b - 6 * u
    d.line([tuple(a), tuple(tip)], fill=(0, 0, 0), width=w + 3)
    d.line([tuple(a), tuple(tip)], fill=col, width=w)
    head = [tuple(b), tuple(b - 18 * u + 9 * n), tuple(b - 18 * u - 9 * n)]
    d.polygon(head, fill=col, outline=(0, 0, 0))


def _tag(d, xy, text, col, size=14):
    f = font(size, True)
    x, y = xy
    tw = d.textlength(text, font=f)
    y = max(0, y - size - 6)
    d.rectangle([x, y, x + tw + 8, y + size + 5], fill=col)
    d.text((x + 4, y + 1), text, font=f, fill=(255, 255, 255))


def draw_changes(img, chs, side):
    """Boxes on a plate. side='before' or 'after': the object's own box is solid where it is present."""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for c in chs:
        if c["type"] == "moved":
            fb, tb = c["from"]["box_px"], c["to"]["box_px"]
            for box, col, present in ((fb, COL["moved_from"], side == "before"), (tb, COL["moved_to"], side == "after")):
                (d.rectangle(box, outline=col, width=3) if present else _dashed_rect(d, box, col))
            ctr = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)  # noqa: E731
            _arrow(d, ctr(fb), ctr(tb))
            box = tb if side == "after" else fb
            _tag(d, (box[0], box[1]), f"moved {c['label']}", COL["moved_to" if side == "after" else "moved_from"])
        else:
            box, col = c["box_px"], COL[c["type"]]
            present = (c["type"] == "removed") == (side == "before")
            (d.rectangle(box, outline=col, width=3) if present else _dashed_rect(d, box, col))
            _tag(d, (box[0], box[1]), f"{c['type']} {c['label']}", col)
    return np.asarray(im)


# ------------------------------------------------------------------ 1. raw | mask | plate

def raw_vs_plate(n_clip=5, n=80, pw=240):
    print("raw_vs_plate")
    frames = read_samples(REAL / f"clip{n_clip:02d}.mp4", n=n)
    from rayvote.static import PersonSegmenter
    masks = person_masks(frames, PersonSegmenter("cuda" if _cuda() else "cpu"))
    ph = int(round(frames.shape[1] * pw / frames.shape[2]))
    sm = np.stack([cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32)
    sk = np.stack([cv2.resize(m.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST) > 0 for m in masks])
    clean = sm.copy()
    clean[sk] = np.nan
    yy, xx = np.mgrid[0:ph, 0:pw]
    checker = np.where(((yy // 8 + xx // 8) % 2)[..., None] == 0, 60, 80).astype(np.float32) * np.ones(3)
    head = np.concatenate([label_bar(pw, "raw"), label_bar(pw, "person mask"), label_bar(pw, "background plate")], 1)
    out = []
    for k in range(n):
        raw = sm[k].astype(np.uint8)
        ov = raw.copy()
        ov[sk[k]] = (0.45 * ov[sk[k]] + 0.55 * np.array([230, 40, 40])).astype(np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pl = np.nanmedian(clean[:k + 1], 0)
        hole = np.isnan(pl).any(-1)
        pl[hole] = checker[hole]
        out.append(np.concatenate([head, np.concatenate([raw, ov, pl.astype(np.uint8)], 1)], 0))
    dur = [100] * (n - 1) + [2500]
    save_gif(out, DOCS / "real/raw_vs_plate.gif", durations=dur)


# ------------------------------------------------------------------ 2. before/after flicker

def flicker(a=1, b=5):
    print("before_after_flicker")
    chs = changes(f"clip{a:02d}_vs_clip{b:02d}")
    pa, pb = plate_like(a, a), plate_like(b, a)
    w = pa.shape[1]
    fa = np.concatenate([label_bar(w, f"clip{a:02d} (before)", h=28, size=16), draw_changes(pa, chs, "before")], 0)
    fb = np.concatenate([label_bar(w, f"clip{b:02d} (after)", h=28, size=16), draw_changes(pb, chs, "after")], 0)
    save_gif([fa, fb], DOCS / "real/before_after_flicker.gif", durations=[1100, 1100], colors=192)


# ------------------------------------------------------------------ 3. timeline

def timeline(n_clips=6, hold=25, fade=4):
    print("timeline")
    steps = [t for t in json.loads((OUT_REAL / "timeline.json").read_text())
             if int(t["after"][-2:]) == int(t["before"][-2:]) + 1]
    plates = {i: plate_like(i, 1) for i in range(1, n_clips + 1)}
    w = plates[1].shape[1]
    wrap = 84
    caps = []
    for t in steps:
        lines = [(f"{t['before']} -> {t['after']}: " + (", ".join(f"{c['type']} {c['label']}" for c in t["changes"])
                                                         or "no change detected"), True)]
        for c in t["changes"]:
            lines += [(s, False) for s in textwrap.wrap(c["description"], wrap)]
        caps.append(lines)
    H = 12 + 18 * max(len(c) for c in caps)
    out, dur = [], []
    first = np.concatenate([label_bar(w, "clip01", h=28, size=16), plates[1],
                            caption_bar(w, [("clip01: starting state", True)], H)], 0)
    out.append(first); dur.append(1500)
    for t, cap in zip(steps, caps):
        i, j = int(t["before"][-2:]), int(t["after"][-2:])
        bar = label_bar(w, f"{t['before']} -> {t['after']}", h=28, size=16)
        cb = caption_bar(w, cap, H)
        for k in range(1, fade + 1):
            a = k / (fade + 1)
            mix = ((1 - a) * plates[i] + a * plates[j]).astype(np.uint8)
            out.append(np.concatenate([bar, mix, cb], 0)); dur.append(100)
        chs = changes(t["dir"])
        out.append(np.concatenate([bar, draw_changes(plates[j], chs, "after"), cb], 0))
        dur.append(hold * 100)
    save_gif(out, DOCS / "real/timeline.gif", durations=dur, colors=192)


# ------------------------------------------------------------------ 4/5. 3D orbits

def _render_orbit(result, path, title, azims, elev=28, n_pts=30000, zoom=1.3, focus=None, colors=96):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rayvote.outputs import _scene_points
    xyz, rgb, _ = _scene_points(result, n_pts if focus is None else 4 * n_pts)
    grey = (0.3 * rgb + 0.7 * 180) / 255
    groups = []
    for ch in result["changes"]:
        if ch["type"] == "moved":
            groups += [(ch["from"]["xyz"], "#d62728"), (ch["to"]["xyz"], "#2ca02c")]
        else:
            groups.append((ch["cluster"]["xyz"], "#d62728" if ch["type"] == "removed" else "#2ca02c"))
    rng = np.random.default_rng(0)
    groups = [(p[rng.choice(len(p), min(len(p), 4000), replace=False)], c) for p, c in groups]
    if focus is not None:  # keep the part of the room around the changes
        cen = np.mean([p.mean(0) for p, _ in groups], 0)
        keep = np.linalg.norm(xyz[:, :2] - cen[:2], axis=1) < focus
        xyz, grey = xyz[keep][:n_pts], grey[keep][:n_pts]
    lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0)
    fig = plt.figure(figsize=(6.4, 4.0), dpi=100)
    ax = fig.add_axes([0, 0.05, 1, 0.9], projection="3d")
    ax.scatter(*xyz.T, c=grey, s=0.5 if focus is None else 1.2, depthshade=False, linewidths=0)
    for p, c in groups:
        ax.scatter(*p.T, c=c, s=3 if focus is None else 6, depthshade=False, linewidths=0)
    for k, ch in enumerate(result["changes"]):
        if ch["type"] == "moved":
            a, b = ch["from"]["centroid"], ch["to"]["centroid"]
            ax.quiver(a[0], a[1], a[2] + 0.3, *(b - a), color="black", linewidth=2.5, arrow_length_ratio=0.25)
        c = ch["to"]["centroid"] if ch["type"] == "moved" else ch["cluster"]["centroid"]
        ax.text(c[0], c[1], c[2] + 0.55 + 0.3 * k, f"{ch['type']} {ch['label']}", fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.3", lw=0.6))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(0, max(hi[2], 1))
    ax.set_box_aspect(hi - np.array([lo[0], lo[1], 0]) + 1e-3, zoom=zoom)
    ax.set_axis_off()
    fig.text(0.02, 0.95, title, fontsize=12, weight="bold")
    fig.text(0.02, 0.02, "grey unchanged   red removed / old position   green added / new position   arrow moved",
             fontsize=8.5, color="0.2")
    frames = []
    for az in azims:
        ax.view_init(elev=elev, azim=az)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    save_gif(frames, path, colors=colors)


def orbit_real(a=1, b=5):
    print("orbit (real)")
    from rayvote.semantics import Clip
    from rayvote.static import compare
    dev = "cuda" if _cuda() else "cpu"
    with tempfile.TemporaryDirectory() as tmp:
        res, _ = compare(clip(a), clip(b), tmp, (f"clip{a:02d}", f"clip{b:02d}"), device=dev, clip=Clip(dev),
                         log=lambda *x: None)
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    # one viewpoint: swing around the camera axis rather than a full turn (the back is empty)
    _render_orbit(res, DOCS / "real/orbit.gif", f"real: clip{a:02d} vs clip{b:02d} (depth on the plates)",
                  180 + 35 * np.sin(t), elev=25, focus=2.5, colors=64, zoom=1.25)


def orbit_synthetic():
    print("orbit (synthetic)")
    from rayvote.pipeline import run
    d = ROOT / "data/synthetic"
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "cache").symlink_to(ROOT / "out/demo/cache")
        res = run(str(d / "before.mp4"), str(d / "after.mp4"), tmp, log=lambda *x: None)
    _render_orbit(res, DOCS / "synthetic_orbit.gif", "synthetic scene a (walkaround mode)",
                  np.linspace(-140, 220, 60, endpoint=False), elev=35, zoom=1.4, n_pts=20000, colors=48)


JOBS = {"raw": raw_vs_plate, "flicker": flicker, "timeline": timeline, "orbit": orbit_real,
        "synthetic": orbit_synthetic}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=",".join(JOBS), help="comma list of " + ", ".join(JOBS))
    for name in ap.parse_args().only.split(","):
        JOBS[name.strip()]()
