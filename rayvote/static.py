"""Static-camera mode: clips filmed from one fixed viewpoint (tripod).

Per clip: mask people (segmenter + motion), build a clean background plate by temporal
median over unmasked samples, run metric monocular depth on the plate. Clips share the
camera, so comparing two of them is free-space voting along each pixel ray:

    d_after > d_before + m  -> the before surface is gone, the ray now passes through: REMOVED
    d_after < d_before - m  -> something new blocks the ray in front:                ADDED
    otherwise               -> consistent
    pixel hidden by a person in all samples of either clip -> UNOBSERVED (no verdict)

A pixel only counts as changed if the plate colour changed too (two independent cues).
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from . import geometry as G
from .frames import InputError
from .pipeline import cached, room_frame, seed_all, to_room, video_sig

HFOV_DEG = 65.0  # assumed horizontal field of view of a phone main camera
WORK_W = 640     # processing width
N_SAMPLES = 150  # frames sampled per clip (short clips: about every frame)
FILL_GROW = 7    # second pass: tighter person mask, used only where the first pass saw nothing
FILL_SEEN = 0.05 # second pass: unobserved only if masked in more than 95 % of the samples


# ---------------------------------------------------------------- per clip

def read_samples(video, n=N_SAMPLES):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise InputError(f"cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    want = set(np.linspace(0, max(total - 1, 0), n).astype(int).tolist()) if total > 0 else None
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if want is None or i in want:
            h = int(round(fr.shape[0] * WORK_W / fr.shape[1]))
            frames.append(cv2.cvtColor(cv2.resize(fr, (WORK_W, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if len(frames) < 5:
        raise InputError(f"{video}: only {len(frames)} frames readable")
    return np.stack(frames)


class PersonSegmenter:
    """Mask R-CNN person instances (boxes survive face pixelation better than semantic seg)."""

    def __init__(self, device):
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
        w = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.net = maskrcnn_resnet50_fpn(weights=w, box_score_thresh=0.4).to(device).eval()
        self.person = w.meta["categories"].index("person")
        self.device = device

    @torch.no_grad()
    def __call__(self, frames, batch=4):
        out = np.zeros(frames.shape[:3], bool)
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(frames[i:i + batch]).permute(0, 3, 1, 2).float().div(255).to(self.device)
            for k, det in enumerate(self.net(list(x))):
                keep = (det["labels"] == self.person) & (det["scores"] > 0.4)
                if keep.any():
                    out[i + k] = (det["masks"][keep, 0] > 0.3).any(0).cpu().numpy()
        return out


def motion_blobs(frames, med, thr=40, min_frac=0.004):
    """Large moving blobs vs the clip median; thin edge flicker from camera shake is dropped."""
    out = np.zeros(frames.shape[:3], bool)
    k = np.ones((7, 7), np.uint8)
    for i, f in enumerate(frames):
        m = (np.abs(f.astype(np.float32) - med).max(-1) > thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        big = np.flatnonzero(st[1:, cv2.CC_STAT_AREA] > min_frac * m.size) + 1
        out[i] = np.isin(lab, big)
    return out


def person_masks(frames, seg, med=None, grow=(21,)):
    """Person instances | large moving blobs, grown by each of `grow` px. One bool stack per size."""
    person = seg(frames)
    motion = motion_blobs(frames, np.median(frames, 0) if med is None else med)
    out = []
    for g in grow:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g))
        out.append(np.stack([cv2.dilate((p | m).astype(np.uint8), k) > 0 for p, m in zip(person, motion)]))
    return out if len(grow) > 1 else out[0]


def _masked_median(frames, masks):
    f = frames.astype(np.float32)
    f[masks] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(f, 0), (~masks).sum(0)


def build_plate(frames, seg):
    """Median background with people removed. Returns plate, mask stack, never-seen mask.

    Per pixel, the median over the samples where that pixel is not person-masked. First pass:
    generous mask (21 px), needs 15 % clean samples. Where that leaves a hole (a person who
    lingers in one place), a second pass with a tight mask (7 px) fills every pixel that is clean
    in at least 5 % of the samples; only the rest is unobserved."""
    med = np.median(frames, 0)
    masks, tight = person_masks(frames, seg, med, grow=(21, FILL_GROW))
    plate, seen = _masked_median(frames, masks)
    never = (seen < max(4, 0.15 * len(frames))) | np.isnan(plate).any(-1)
    fill, seen_t = _masked_median(frames, tight)
    ok = never & (seen_t >= max(3, FILL_SEEN * len(frames))) & ~np.isnan(fill).any(-1)
    plate[ok] = fill[ok]
    never &= ~ok
    plate[np.isnan(plate)] = med[np.isnan(plate)]
    return plate.astype(np.uint8), masks, never


def process_clip(video, cache, models, log):
    def _do():
        frames = read_samples(video)
        plate, masks, never = build_plate(frames, models["seg"]())
        depth = models["depth"]()(Image.fromarray(plate))
        # overlay: a frame with the most person pixels, mask in red
        j = int(np.argmax(masks.reshape(len(masks), -1).sum(1)))
        ov = frames[j].copy()
        ov[masks[j]] = (0.45 * ov[masks[j]] + 0.55 * np.array([230, 40, 40])).astype(np.uint8)
        return {"plate": plate, "depth": depth, "never": never, "overlay": ov,
                "masked_frac": float(masks.mean()), "n": len(frames)}
    res = cached(cache / f"clip_{video_sig([video], {'w': WORK_W, 'n': N_SAMPLES, 'fill': [FILL_GROW, FILL_SEEN], 'v': 4})}.pkl", _do)
    log(f"  {Path(video).name}: {res['n']} samples, person/motion masked {100 * res['masked_frac']:.1f}% "
        f"of pixel-samples, {100 * res['never'].mean():.2f}% of pixels never visible")
    return res


class _Lazy:
    def __init__(self, fn):
        self.fn, self.obj = fn, None

    def __call__(self):
        if self.obj is None:
            self.obj = self.fn()
        return self.obj


def make_models(device):
    from .depth import DepthModel

    class _Depth:
        def __init__(self):
            self.m = DepthModel(device)

        def __call__(self, pil):
            tmp = Path(".rayvote_plate.png")
            pil.save(tmp)
            try:
                return self.m(tmp)
            finally:
                tmp.unlink(missing_ok=True)
    return {"seg": _Lazy(lambda: PersonSegmenter(device)), "depth": _Lazy(_Depth)}


# ---------------------------------------------------------------- comparison

def alignment_check(pa, pb):
    """Feature-match the two plates. Returns (homography b->a, median residual px, median shift px)."""
    sift = cv2.SIFT_create(4000)
    ga, gb = cv2.cvtColor(pa, cv2.COLOR_RGB2GRAY), cv2.cvtColor(pb, cv2.COLOR_RGB2GRAY)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None:
        return np.eye(3), float("nan"), float("nan"), 0
    m = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 20:
        return np.eye(3), float("nan"), float("nan"), len(good)
    pb_ = np.float32([kb[g.queryIdx].pt for g in good])
    pa_ = np.float32([ka[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(pb_, pa_, cv2.RANSAC, 2.0)
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(pb_[inl][None], H)[0]
    res = float(np.median(np.linalg.norm(proj - pa_[inl], axis=1)))
    shift = float(np.median(np.linalg.norm(pb_[inl] - pa_[inl], axis=1)))
    return H, res, shift, int(inl.sum())


def normalise_exposure(src, ref, mask):
    """Per-channel gain/offset from src to ref, fitted on mask pixels with one outlier pass."""
    out = src.astype(np.float32).copy()
    for c in range(3):
        x, y = src[..., c][mask].astype(np.float32), ref[..., c][mask].astype(np.float32)
        g, o = np.polyfit(x, y, 1)
        keep = np.abs(g * x + o - y) < 2.5 * np.std(g * x + o - y)
        g, o = np.polyfit(x[keep], y[keep], 1)
        out[..., c] = g * out[..., c] + o
    return np.clip(out, 0, 255).astype(np.uint8)


def intrinsics(w, h):
    f = w / 2 / np.tan(np.deg2rad(HFOV_DEG) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def compare(A, B, out, labels=("before", "after"), voxel=0.05, device=None, clip=None, log=print,
            m_abs=0.15, m_rel=0.10, col_thr=28, min_area=0.002):
    from .describe import describe_all
    from .depth import fit_scale_shift
    from .outputs import write_outputs
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pa, pb = A["plate"], B["plate"]
    h, w = pa.shape[:2]
    H, res, shift, n_in = alignment_check(pa, pb)
    log(f"  alignment check: {n_in} inlier matches, median shift {shift:.2f} px, residual {res:.2f} px")
    if not np.isfinite(res) or res > 1.0:
        log("  warning: clips do not align well (camera moved or too few matches); results are less reliable")
    db = B["depth"]
    nb = B["never"]
    if np.isfinite(shift) and shift > 1.5:  # camera nudged: warp after-plate into before view
        pb = cv2.warpPerspective(pb, H, (w, h))
        db = cv2.warpPerspective(db, H, (w, h), flags=cv2.INTER_NEAREST)
        nb = cv2.warpPerspective(nb.astype(np.uint8), H, (w, h)) > 0
    da = A["depth"]
    unobs = cv2.dilate((A["never"] | nb).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    # put the after depth on the before depth's scale using the (mostly static) rest of the image
    ok = ~unobs
    idx = np.random.default_rng(0).choice(np.flatnonzero(ok), min(20000, ok.sum()), replace=False)
    s, b, _ = fit_scale_shift(db.ravel()[idx], da.ravel()[idx])
    db = s * db + b

    # auto-exposure / white balance differ between clips: map after-plate colours onto before-plate
    pb = normalise_exposure(pb, pa, ~unobs)

    m = np.maximum(m_abs, m_rel * da)
    lab_a = cv2.cvtColor(pa, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(pb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dcol = cv2.GaussianBlur(np.linalg.norm(lab_a - lab_b, axis=-1), (7, 7), 0)
    colour = dcol > col_thr
    edge_ok = G.window_minmax(da, 5)[1] - G.window_minmax(da, 5)[0] < 0.08 * da  # skip depth edges
    added = (db < da - m) & colour & ~unobs
    removed = (db > da + m) & colour & ~unobs & edge_ok
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    # open removes speckle, close joins the pieces of one object (chair back, seat, legs)
    added = cv2.morphologyEx(cv2.morphologyEx(added.astype(np.uint8), cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, kc) > 0
    removed = cv2.morphologyEx(cv2.morphologyEx(removed.astype(np.uint8), cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, kc) > 0
    added &= ~unobs
    removed &= ~unobs
    # occlusion guard: next to a new foreground object the after depth bleeds (and the old object
    # can be partly hidden), so "the ray now passes through" is not trustworthy there
    removed_raw = removed.copy()
    guard = max(3, int(0.1 * w)) | 1   # about a tenth of the image width
    removed &= ~(cv2.dilate(added.astype(np.uint8), np.ones((guard, guard), np.uint8)) > 0)

    K = intrinsics(w, h)
    view_a = {"K": K, "R": np.eye(3), "t": np.zeros(3), "depth": da.astype(np.float32), "rgb": pa, "rec": "before"}
    view_b = {"K": K, "R": np.eye(3), "t": np.zeros(3), "depth": db.astype(np.float32), "rgb": pb, "rec": "after"}
    cloud_a, cloud_b = G.fuse([view_a], voxel, 1), G.fuse([view_b], voxel, 1)
    fn, fd = G.estimate_floor(cloud_a["xyz"], np.array([0, -1.0, 0]))
    Rr, origin = room_frame([view_a], fn, fd)

    v, u = np.mgrid[0:h, 0:w]

    def pts(mask, depth):
        z = depth[mask]
        x = (u[mask] - K[0, 2]) / K[0, 0] * z
        y = (v[mask] - K[1, 2]) / K[1, 1] * z
        return np.stack([x, y, z], 1)

    cands = []
    for kind, mask, depth, plate_own, plate_other in (("removed", removed, da, pa, pb), ("added", added, db, pb, pa)):
        # group pieces that nearly touch in the image (one chair can split at its dark fabric)
        grp = cv2.dilate(mask.astype(np.uint8), np.ones((41, 41), np.uint8))
        n, cc = cv2.connectedComponents(grp)
        for i in range(1, n):
            cm = (cc == i) & mask
            if cm.sum() < min_area * h * w:
                continue
            xs_ = np.nonzero(cm.any(0))[0]
            if xs_.max() - xs_.min() + 1 < 0.03 * w:   # vertical sliver: depth edge or pixelation seam
                log(f"  skip {kind} sliver ({cm.sum()} px, {xs_.max() - xs_.min() + 1} px wide)")
                continue
            if kind == "removed":
                raw = (cv2.dilate(cm.astype(np.uint8), np.ones((guard, guard), np.uint8)) > 0) & removed_raw
                if cm.sum() < 0.4 * raw.sum():   # most of it was in the shadow of a new object
                    log(f"  skip removed region ({cm.sum()} px): mostly next to an added object")
                    continue
            ys_, xs_ = np.nonzero(cm)
            log(f"  {kind} region: {cm.sum()} px at x {xs_.min()}-{xs_.max()}, y {ys_.min()}-{ys_.max()}")
            ys, xs = np.nonzero(cm)
            stats_i = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            p = pts(cm, depth)
            pr = to_room(p, Rr, origin)
            x0, y0, bw, bh = stats_i
            gap = float(np.median(np.abs(db[cm] - da[cm])))
            c = {"type": kind, "xyz_world": p, "xyz": pr, "n_voxels": int(cm.sum()), "mask": cm,
                 "centroid": np.median(pr, 0), "bbox_min": np.percentile(pr, 3, 0), "bbox_max": np.percentile(pr, 97, 0),
                 "box_px": [int(x0), int(y0), int(x0 + bw), int(y0 + bh)],
                 "evidence": {"pixels": int(cm.sum()), "median_depth_gap_m": round(gap, 2),
                              "confidence": round(min(1.0, gap / 0.5) * float(np.mean(colour[cm])), 2)}}
            pad = 0.15
            X0, Y0 = max(0, int(x0 - pad * bw)), max(0, int(y0 - pad * bh))
            X1, Y1 = min(w, int(x0 + bw * (1 + pad))), min(h, int(y0 + bh * (1 + pad)))
            c["crop_own_img"] = Image.fromarray(plate_own[Y0:Y1, X0:X1])
            # for labelling, fade everything outside the change mask so CLIP looks at the object
            fo = plate_own[Y0:Y1, X0:X1].astype(np.float32)
            mk = cv2.dilate(cm[Y0:Y1, X0:X1].astype(np.uint8), np.ones((9, 9), np.uint8))[..., None] > 0
            c["crop_focus_img"] = Image.fromarray(np.where(mk, fo, 0.25 * fo + 0.75 * 128).astype(np.uint8))
            c["crop_other_img"] = Image.fromarray(plate_other[Y0:Y1, X0:X1])
            cands.append(c)
    log(f"  {len(cands)} change regions")

    from .pipeline import _match_moved
    crops = out / "crops"
    crops.mkdir(exist_ok=True)
    for i, c in enumerate(cands):
        c["id"] = i
        e = clip.embed([c["crop_own_img"], c["crop_focus_img"]]).mean(0)
        emb = e / np.linalg.norm(e)
        c["emb"] = emb
        c["label"], c["label_conf"], c["top3"] = clip.label(emb)
        pair = (c["crop_own_img"], c["crop_other_img"])
        if c["type"] == "added":
            pair = pair[::-1]
        for side, im in zip(("before", "after"), pair):
            im = im.copy()
            im.thumbnail((256, 256))
            im.save(crops / f"{i:02d}_{c['type']}_{side}.jpg", quality=90)
            c[f"thumb_{side}"] = f"crops/{i:02d}_{c['type']}_{side}.jpg"
    changes = _match_moved(cands)
    allxyz = to_room(np.concatenate([cloud_a["xyz"], cloud_b["xyz"]]), Rr, origin)
    describe_all(changes, allxyz)
    for ch in changes:
        ch["text"] = ch["text"].replace("the near part of the room", "the foreground").replace(
            "the far part of the room", "the background").replace("start point", "camera")

    def cl(c, unob_px):
        xyz = to_room(c["xyz"], Rr, origin)
        # unobserved voxels: back-project the never-visible pixels and mark nearest voxels
        un = np.zeros(len(xyz), bool)
        if unob_px.any():
            from scipy.spatial import cKDTree
            up = to_room(pts(unob_px, c["depth"]), Rr, origin)
            un = cKDTree(up).query(xyz, distance_upper_bound=voxel)[0] < voxel
        return {"xyz": xyz, "rgb": c["rgb"], "unobserved": un}
    cloud_a["depth"], cloud_b["depth"] = da, db
    result = {"changes": changes, "cloud_before": cl(cloud_a, unobs), "cloud_after": cl(cloud_b, unobs),
              "cameras": [{"name": labels[0], "rec": "before", "frame_index": 0,
                           "center": to_room(np.zeros(3), Rr, origin).tolist()},
                          {"name": labels[1], "rec": "after", "frame_index": 0,
                           "center": to_room(np.zeros(3), Rr, origin).tolist()}],
              "voxel": voxel}
    write_outputs(result, out, labels[0], labels[1])
    _overlays(out, A, B, pa, pb, changes, unobs, labels)
    from .outputs import write_hero
    ov = np.asarray(Image.open(out / "overlay.jpg"))
    write_hero(result, out, out / "hero.png", scene_img=ov[ov.shape[0] // 2:])
    meta = {"alignment": {"inliers": n_in, "median_shift_px": round(shift, 2), "residual_px": round(res, 2)},
            "depth_scale_after_to_before": [round(s, 3), round(b, 3)],
            "unobserved_pixel_fraction": round(float(unobs.mean()), 4), "image_size": [int(w), int(h)]}
    (out / "static_meta.json").write_text(json.dumps(meta, indent=2))
    return result, meta


def _overlays(out, A, B, pa, pb, changes, unobs, labels):
    """person-mask overlays for both clips, and change masks drawn on the plates."""
    cols = {"removed": (214, 39, 40), "added": (44, 160, 44)}
    img_a, img_b = pa.copy(), pb.copy()
    for ch in changes:
        parts = [("removed", ch["from"]), ("added", ch["to"])] if ch["type"] == "moved" else [(ch["type"], ch["cluster"])]
        for kind, c in parts:
            tgt = img_a if kind == "removed" else img_b
            col = (123, 44, 191) if ch["type"] == "moved" else cols[kind]
            tgt[c["mask"]] = (0.5 * tgt[c["mask"]] + 0.5 * np.array(col)).astype(np.uint8)
            x0, y0, x1, y1 = c["box_px"]
            cv2.rectangle(tgt, (x0, y0), (x1, y1), col, 2)
            cv2.putText(tgt, f"{ch['type']} {ch['label']}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        col, 2)
    for img in (img_a, img_b):
        img[unobs] = (0.6 * img[unobs] + 0.4 * np.array([120, 170, 220])).astype(np.uint8)
    top = np.concatenate([A["overlay"], B["overlay"]], 1)
    bot = np.concatenate([img_a, img_b], 1)
    for im, y, txt in ((top, 22, f"{labels[0]}: person mask (red)"), (bot, 22, f"{labels[0]} plate: removed")):
        cv2.putText(im, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(top, f"{labels[1]}: person mask (red)", (pa.shape[1] + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    cv2.putText(bot, f"{labels[1]} plate: added", (pa.shape[1] + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    Image.fromarray(np.concatenate([top, bot], 0)).save(out / "overlay.jpg", quality=88)


# ---------------------------------------------------------------- entry points

def run_static(videos, out, pairs=None, device=None, log=print):
    """Compare clips filmed from one fixed camera. pairs: list of (i, j) indices; default consecutive."""
    seed_all(0)
    t0 = time.time()
    for v in videos:
        if not Path(v).exists():
            raise InputError(f"video not found: {v}")
    out = Path(out)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache = out / "cache"
    models = make_models(device)
    log(f"[static] device={device}; building background plates")
    clips = [process_clip(v, cache, models, log) for v in videos]
    shapes = {c["plate"].shape for c in clips}
    if len(shapes) > 1:
        raise InputError(f"clips differ in aspect ratio or orientation {sorted(shapes)}; "
                         "static mode needs the same framing for every clip")
    models = None
    torch.cuda.empty_cache() if device == "cuda" else None
    from .semantics import Clip
    clip = Clip(device)
    names = [Path(v).stem for v in videos]
    pairs = pairs or [(i, i + 1) for i in range(len(videos) - 1)]
    timeline = []
    for i, j in pairs:
        d = out / f"{names[i]}_vs_{names[j]}"
        log(f"[static] {names[i]} -> {names[j]}")
        res, meta = compare(clips[i], clips[j], d, (names[i], names[j]), device=device, clip=clip, log=log)
        cj = json.loads((d / "changes.json").read_text())
        timeline.append({"before": names[i], "after": names[j], "dir": d.name, "meta": meta,
                         "changes": [{"type": c["type"], "label": c["label"], "description": c["description"]}
                                     for c in cj["changes"]]})
    (out / "timeline.json").write_text(json.dumps(timeline, indent=2))
    L = ["# Change timeline", ""]
    for t in timeline:
        L.append(f"## {t['before']} -> {t['after']}  ({len(t['changes'])} change(s))")
        L.append("")
        L += [f"- {c['description']}" for c in t["changes"]] or ["- no change detected"]
        L += ["", f"![]({t['dir']}/overlay.jpg)", ""]
    (out / "timeline.md").write_text("\n".join(L))
    rows = []
    for t in timeline:
        im = np.asarray(Image.open(out / t["dir"] / "overlay.jpg"))
        im = cv2.resize(im[im.shape[0] // 2:], None, fx=0.5, fy=0.5)
        bar = np.full((28, im.shape[1], 3), 255, np.uint8)
        txt = f"{t['before']} -> {t['after']}: " + (", ".join(f"{c['type']} {c['label']}" for c in t["changes"])
                                                    or "no change")
        cv2.putText(bar, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        rows += [bar, im]
    Image.fromarray(np.concatenate(rows, 0)).save(out / "timeline.jpg", quality=85)
    log(f"[static] done in {time.time() - t0:.0f} s -> {out}")
    return timeline
