"""End-to-end: two videos -> shared 3D model -> free-space votes -> labelled changes."""
import hashlib
import json
import pickle
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from . import geometry as G
from .depth import DepthModel, edge_mask, fit_scale_shift, sparse_depth
from .frames import InputError, select_frames
from .semantics import STRUCTURE, Clip

DEPTH_W = 384  # working resolution for depth maps


def seed_all(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def video_sig(paths, extra):
    h = hashlib.sha1()
    for p in paths:
        st = Path(p).stat()
        h.update(f"{Path(p).resolve()}:{st.st_size}:{int(st.st_mtime)}".encode())
    h.update(json.dumps(extra, sort_keys=True).encode())
    return h.hexdigest()[:12]


def cached(path, fn):
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    res = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


# ---------------------------------------------------------------- geometry helpers

def scale_K(K, s):
    K = K.copy()
    K[:2] *= s
    return K


def project(xyz, vw):
    pc = xyz @ vw["R"].T + vw["t"]
    z = pc[:, 2]
    uvw = pc @ vw["K"].T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)
    h, w = vw["depth"].shape
    ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    return uv, z, ok


def room_frame(views, floor_n, floor_d):
    """Room frame: z up from the floor, origin under the first 'before' camera,
    x = that camera's viewing direction flattened onto the floor, y = left."""
    v0 = next(v for v in views if v["rec"] == "before")
    c0 = -v0["R"].T @ v0["t"]
    fwd = v0["R"][2]
    fwd = fwd - (fwd @ floor_n) * floor_n
    fwd /= np.linalg.norm(fwd)
    left = np.cross(floor_n, fwd)
    Rr = np.stack([fwd, left, floor_n])
    origin = c0 - (c0 @ floor_n + floor_d) * floor_n
    return Rr, origin


def to_room(x, Rr, origin):
    return (x - origin) @ Rr.T


# ---------------------------------------------------------------- main

def run(before, after, out, n_frames=40, voxel=0.05, gt_poses=None, device=None, vocab=None, log=print,
        max_same_sim=0.76):
    t0 = time.time()
    seed_all(0)
    for p in (before, after):
        if not Path(p).exists():
            raise InputError(f"video not found: {p}")
    out = Path(out)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sig = video_sig([before, after], {"n": n_frames, "gt": str(gt_poses)})
    cache = out / "cache" / sig
    img_dir = cache / "images"
    log(f"[rayvote] device={device} cache={cache}")

    # 1. keyframes
    def _frames():
        return {"before": select_frames(before, img_dir / "before", n_frames, prefix="b"),
                "after": select_frames(after, img_dir / "after", n_frames, prefix="a")}
    log("[1/6] selecting keyframes")
    frames = cached(cache / "frames.pkl", _frames)
    fidx = {f"{rec}/{f['name']}": (rec, f["frame_index"]) for rec in frames for f in frames[rec]}

    # 2. poses
    if gt_poses:
        log("[2/6] using ground-truth poses (synthetic test only)")
        cams = _gt_cameras(gt_poses, fidx, img_dir)
    else:
        log("[2/6] joint SfM over both recordings (pycolmap)")

        def _sfm():
            from .sfm import export_cameras, run_sfm
            return export_cameras(run_sfm(img_dir, cache / "sfm", log=log))
        cams = cached(cache / "sfm.pkl", _sfm)
    nb = sum(fidx[n][0] == "before" for n in cams if n in fidx)
    na = sum(fidx[n][0] == "after" for n in cams if n in fidx)
    log(f"      registered {nb} before / {na} after frames")
    if nb < 5 or na < 5:
        raise InputError(f"only {nb} before / {na} after frames registered into a shared model; the two "
                         "recordings need more overlap of static, textured background")

    # 3. depth
    log("[3/6] monocular depth + alignment to SfM points")

    def _depth():
        net = DepthModel(device)
        res = {}
        for name in sorted(cams):
            res[name] = net(img_dir / name)
        del net
        torch.cuda.empty_cache() if device == "cuda" else None
        return res
    raw = cached(cache / "depth_raw.pkl", _depth)
    views = _align(cams, raw, fidx, img_dir, use_sfm=not gt_poses, log=log)

    # 4. fusion
    log("[4/6] fusing voxel clouds")
    B = [v for v in views if v["rec"] == "before"]
    A = [v for v in views if v["rec"] == "after"]
    G.consistency_filter(B)
    G.consistency_filter(A)
    cloud_b, cloud_a = G.fuse(B, voxel), G.fuse(A, voxel)
    allxyz = np.concatenate([cloud_b["xyz"], cloud_a["xyz"]])
    up_guess = -np.mean([v["R"][1] for v in views], 0)
    fn, fd = G.estimate_floor(allxyz, up_guess)
    Rr, origin = room_frame(views, fn, fd)
    log(f"      before {len(cloud_b['xyz'])} voxels, after {len(cloud_a['xyz'])} voxels")

    # 5. free-space evidence
    log("[5/6] free-space voting")
    vb = G.free_space_votes(cloud_b["xyz"], A, G.normals(cloud_b["xyz"]))   # before-points vs after-cameras
    va = G.free_space_votes(cloud_a["xyz"], B, G.normals(cloud_a["xyz"]))
    rem_mask, unobs_b = G.classify(vb)
    add_mask, unobs_a = G.classify(va)
    cands = []
    for kind, cloud, votes, mask, own, other in (("removed", cloud_b, vb, rem_mask, B, A),
                                                 ("added", cloud_a, va, add_mask, A, B)):
        cands += _clusters(kind, cloud, votes, mask, own, other, voxel, Rr, origin, fn, fd,
                           veto_pts=_sparse_points(other))
    log(f"      {len(cands)} change clusters after filtering")

    # 6. semantics + moved matching
    log("[6/6] labelling (CLIP) and matching moved objects")
    changes = []
    if cands:
        clip = Clip(device, vocab)
        crops_dir = out / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(cands):
            c["id"] = i
            _crops(c, img_dir, crops_dir)
            emb = clip.embed([c["crop_own_img"]])[0]
            c["emb"] = emb
            c["label"], c["label_conf"], c["top3"] = clip.label(emb)
            if c.get("crop_other_img") is not None:
                c["other_sim"] = float(clip.embed([c["crop_other_img"]])[0] @ emb)
        # appearance check: if the other recording's view of the region looks the same as
        # this one, the free-space votes came from geometry error, not a real change
        # (only for small clusters: CLIP similarity of two crops of a busy room is a weak signal)
        kept = [c for c in cands if c.get("other_sim", 0) < max_same_sim or c["n_voxels"] >= 200]
        for c in cands:
            if c not in kept:
                log(f"      dropped {c['type']} '{c['label']}' ({c['n_voxels']} voxels): before/after crops "
                    f"look the same (CLIP sim {c['other_sim']:.2f})")
        changes = _match_moved(kept)
    room_xyz = to_room(allxyz, Rr, origin)
    from .describe import describe_all
    describe_all(changes, room_xyz)

    result = {
        "changes": changes,
        "cloud_before": {**cloud_b, "xyz": to_room(cloud_b["xyz"], Rr, origin), "unobserved": unobs_b,
                         "votes": vb},
        "cloud_after": {**cloud_a, "xyz": to_room(cloud_a["xyz"], Rr, origin), "unobserved": unobs_a,
                        "votes": va},
        "cameras": [{"name": v["name"], "rec": v["rec"], "frame_index": v["frame_index"],
                     "center": to_room(-v["R"].T @ v["t"], Rr, origin).tolist()} for v in views],
        "voxel": voxel,
        "runtime_s": time.time() - t0,
    }
    from .outputs import write_outputs
    write_outputs(result, out, before, after)
    log(f"[rayvote] done in {result['runtime_s']:.0f} s -> {out}")
    return result


def _gt_cameras(gt_path, fidx, img_dir):
    gt = json.loads(Path(gt_path).read_text())
    cams = {}
    for name, (rec, fi) in fidx.items():
        im = cv2.imread(str(img_dir / name))
        s = im.shape[1] / gt["width"]
        p = gt[rec][fi]
        cams[name] = {"K": scale_K(np.array(gt["K"]), s), "R": np.array(p["R"]), "t": np.array(p["t"]),
                      "width": im.shape[1], "height": im.shape[0], "points": np.zeros((0, 3))}
    return cams


def _align(cams, raw, fidx, img_dir, use_sfm, log):
    """Bring each depth map into the SfM frame, then rescale everything to (approx.) metres.

    The metric-depth network gives rough metres; SfM gives consistent but unscaled geometry.
    Global scale = median(pred / sfm depth); per-frame scale+shift fixes the rest.
    """
    k = 1.0
    if use_sfm:
        ratios = []
        for name, cam in cams.items():
            uv, z = sparse_depth(cam, cam["points"])
            if len(z):
                d = raw[name][uv[:, 1].astype(int), uv[:, 0].astype(int)]
                ratios.append(np.median(d / z))
        k = float(np.median(ratios))
        log(f"      SfM->metric scale {k:.3f}")
    views = []
    for name, cam in sorted(cams.items()):
        d = raw[name]
        s, b = 1.0, 0.0
        if use_sfm:
            pts = cam["points"] * k
            c2 = dict(cam, t=cam["t"] * k)
            uv, z = sparse_depth(c2, pts)
            p = d[uv[:, 1].astype(int), uv[:, 0].astype(int)]
            s, b, _ = fit_scale_shift(p, z)
            corr = _local_correction(s * d + b, s * p + b, z, uv)
        else:
            corr = 1.0
        sc = DEPTH_W / cam["width"]
        h = int(round(cam["height"] * sc))
        dd = cv2.resize((s * d + b) * corr, (DEPTH_W, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        edges = cv2.erode((edge_mask(dd) & (dd >= 0.2)).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
        dd[edges] = np.nan
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(img_dir / name)), (DEPTH_W, h)), cv2.COLOR_BGR2RGB)
        rec, fi = fidx[name]
        views.append({"name": name, "rec": rec, "frame_index": fi, "K": scale_K(cam["K"], sc),
                      "R": cam["R"], "t": cam["t"] * k, "depth": dd, "rgb": rgb, "full_scale": sc,
                      "sparse": cam["points"] * k})
    return views


def _local_correction(pred_map, pred, z, uv, k=12, alpha=0.3, step=4):
    """Per-pixel multiplicative correction from SfM points.

    Each pixel takes the median log-ratio (sfm depth / predicted depth) of its k nearest SfM
    points in (u, v, alpha * log predicted depth) space, so a table edge does not borrow the
    correction of the floor right behind it. Computed on a coarse pixel grid, then upsampled.
    """
    from scipy.spatial import cKDTree
    H, W = pred_map.shape
    if len(z) < k:
        return 1.0
    lp = np.log(np.maximum(pred, 1e-3))
    feat = np.stack([uv[:, 0] / W, uv[:, 1] / W, alpha * lp], 1)
    r = np.log(np.clip(z / np.maximum(pred, 1e-3), 0.2, 5))
    tree = cKDTree(feat)
    pd = pred_map
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    q = np.stack([xs.ravel() / W, ys.ravel() / W, alpha * np.log(np.maximum(pd[ys, xs].ravel(), 1e-3))], 1)
    _, idx = tree.query(q, k=k)
    corr = np.median(r[idx], 1).reshape(ys.shape).astype(np.float32)
    return np.exp(cv2.resize(corr, (W, H), interpolation=cv2.INTER_LINEAR))


def _sparse_points(views):
    pts = [v["sparse"] for v in views if len(v["sparse"])]
    if not pts:
        return None
    return np.unique(np.round(np.concatenate(pts), 4), axis=0)


def _clusters(kind, cloud, votes, mask, own, other, voxel, Rr, origin, fn, fd, veto_pts=None, veto_r=0.1,
              veto_frac=0.25,
              min_vox=25, min_height=0.08, min_width=0.08, min_thick=0.04):
    xyz = cloud["xyz"][mask]
    vv = votes[mask]
    veto_tree = None
    lab = G.merge_close(xyz, G.cluster(xyz, eps=0.18), dist=0.4)
    res = []
    for l in set(lab) - {-1}:
        p = xyz[lab == l]
        if len(p) < min_vox:
            continue
        h = p @ fn + fd
        if np.percentile(h, 95) < min_height:      # flat stuff on the floor: depth noise
            continue
        ev = np.linalg.svd(p - p.mean(0), compute_uv=False) / np.sqrt(len(p))
        if 2 * ev[1] < min_width or 2 * ev[2] < min_thick:   # line or flat patch: depth-edge residue
            continue
        # sparse veto: SfM points triangulated by the other recording sit right here, so the
        # surface was there in that recording too and the free-space votes came from depth error
        if veto_pts is not None:
            above = h > 0.12
            if above.sum() >= 5:
                if veto_tree is None:
                    veto_tree = cKDTree(veto_pts)
                near = veto_tree.query(p[above], distance_upper_bound=veto_r)[0] < veto_r
                if near.mean() > veto_frac:
                    continue
        pr = to_room(p, Rr, origin)
        v = vv[lab == l]
        ev = {"free_views": float(np.median(v[:, G.FREE])),
              "consistent_views": float(np.median(v[:, G.CONSISTENT])),
              "occluded_views": float(np.median(v[:, G.OCCLUDED])),
              "free_ratio": float(np.median(v[:, G.FREE] / np.maximum(v[:, G.FREE] + v[:, G.CONSISTENT], 1)))}
        res.append({"evidence": ev, "type": kind, "xyz_world": p, "xyz": pr, "n_voxels": int(len(p)),
                    "centroid": np.median(pr, 0), "bbox_min": pr.min(0), "bbox_max": pr.max(0),
                    "own": own, "other": other})
    return res


def _best_box(xyz, views, mode):
    """Best view and pixel box of a cluster. mode='visible': points agree with that view's depth;
    mode='free': that view looks through the region (the 'empty' side of the comparison)."""
    best = None
    for vw in views:
        uv, z, ok = project(xyz, vw)
        if ok.sum() < 5:
            continue
        ui, vi = np.round(uv[ok]).astype(int).T
        d = vw["depth"][vi, ui]
        m = np.maximum(0.12, 0.08 * z[ok])
        good = np.abs(d - z[ok]) < m if mode == "visible" else d > z[ok] + m
        frac_in = ok.mean()
        score = good.sum() * frac_in ** 2
        if best is None or score > best[0]:
            best = (score, vw, uv[ok], good.sum())
    if best is None:
        return None
    _, vw, uv, n = best
    lo, hi = np.percentile(uv, 2, 0), np.percentile(uv, 98, 0)
    return vw, lo, hi, int(n)


def _crop(img_dir, vw, lo, hi, pad=0.25, min_px=48):
    img = Image.open(img_dir / vw["name"]).convert("RGB")
    s = vw["full_scale"]
    lo, hi = lo / s, hi / s
    c, half = (lo + hi) / 2, np.maximum((hi - lo) * (0.5 + pad), min_px / 2)
    box = (*np.maximum(c - half, 0), *np.minimum(c + half, img.size))
    return img.crop(tuple(int(v) for v in box)), [int(v) for v in box]


def _crops(c, img_dir, crops_dir):
    bv = _best_box(c["xyz_world"], c["own"], "visible")
    if bv is None:
        bv = _best_box(c["xyz_world"], c["own"], "free")
    vw, lo, hi, n = bv
    c["crop_own_img"], c["box_own"] = _crop(img_dir, vw, lo, hi)
    c["view_own"] = vw["name"]
    ov = _best_box(c["xyz_world"], c["other"], "free")
    if ov is not None:
        c["crop_other_img"], c["box_other"] = _crop(img_dir, ov[0], ov[1], ov[2])
        c["view_other"] = ov[0]["name"]
    tag = f"{c['id']:02d}_{c['type']}"
    # before image first, after image second
    pair = (c["crop_own_img"], c.get("crop_other_img"))
    if c["type"] == "added":
        pair = pair[::-1]
    for side, im in zip(("before", "after"), pair):
        if im is not None:
            im.thumbnail((256, 256))
            p = crops_dir / f"{tag}_{side}.jpg"
            im.save(p, quality=90)
            c[f"thumb_{side}"] = f"crops/{p.name}"


def _match_moved(cands, sim_same=0.72, sim_any=0.86, min_size_ratio=0.3):
    rem = [c for c in cands if c["type"] == "removed"]
    add = [c for c in cands if c["type"] == "added"]

    def size(c):
        return np.sort(c["bbox_max"] - c["bbox_min"])

    pairs = []
    if rem and add:
        S = np.zeros((len(rem), len(add)))
        for i, r in enumerate(rem):
            for j, a in enumerate(add):
                sim = float(r["emb"] @ a["emb"])
                ratio = float(np.mean(np.minimum(size(r), size(a)) / np.maximum(size(r), size(a))))
                same = r["label"] == a["label"]
                ok = ratio >= min_size_ratio and (sim >= sim_any or (same and sim >= sim_same))
                S[i, j] = sim + 0.1 * same + 0.2 * ratio if ok else -1
        ri, ai = linear_sum_assignment(-S)
        pairs = [(rem[i], add[j], S[i, j]) for i, j in zip(ri, ai) if S[i, j] > 0]
    used = set()
    changes = []
    for r, a, s in pairs:
        used |= {id(r), id(a)}
        lab, conf = (r["label"], r["label_conf"]) if r["label_conf"] >= a["label_conf"] else (a["label"], a["label_conf"])
        changes.append({"type": "moved", "label": lab, "label_conf": conf, "match_score": float(s),
                        "from": r, "to": a})
    for c in cands:
        if id(c) not in used:
            changes.append({"type": c["type"], "label": c["label"], "label_conf": c["label_conf"],
                            "cluster": c})
    for ch in changes:
        ch["structure_like"] = ch["label"] in STRUCTURE
    return changes
