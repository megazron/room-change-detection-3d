"""Person paths, top-down occupancy and a robot navigation baseline for static-camera clips.

Everything lives in one metric floor frame shared by all clips (same tripod):

  * the floor is a RANSAC plane fitted to the clip01 plate depth; each other clip's plate depth is
    put on clip01's scale with a robust scale/shift fit on the static pixels
  * person track: per frame the person instance mask (Mask R-CNN); ground contact = bottom of the
    mask, its pixel ray intersected with the floor plane. If the feet are cut by the image border or
    hidden behind furniture, the head top + the person's height (measured on frames where both
    ends are visible) gives the floor point instead; frames where neither works are dropped
  * occupancy: plate pixels back-projected to 3D; 0.10-1.20 m above the floor = obstacle,
    floor-height points = observed free floor, the rest of the camera's floor footprint = unknown
  * planner: 8-connected A* on the grid inflated by the robot radius; unknown cells cost extra
"""
import heapq
from pathlib import Path

import cv2
import numpy as np
import torch

from . import geometry as G
from .depth import fit_scale_shift
from .pipeline import cached, room_frame, to_room, video_sig
from .static import WORK_W, intrinsics

CELL = 0.05           # grid resolution (m)
BAND = (0.10, 1.20)   # obstacle height band above the floor (m)
ROBOT_R = 0.25        # robot radius for inflation (m)
STEP = 2              # process every STEP-th video frame


# ---------------------------------------------------------------- per-frame person tracks

def _frames(video, step=STEP):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            h = int(round(fr.shape[0] * WORK_W / fr.shape[1]))
            yield i / fps, cv2.cvtColor(cv2.resize(fr, (WORK_W, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


@torch.no_grad()
def person_track(video, cache, device=None, log=print):
    """Per processed frame: time, best person mask (tracked by overlap), packed. Cached."""
    def _do():
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        w = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        net = maskrcnn_resnet50_fpn(weights=w, box_score_thresh=0.5).to(dev).eval()
        pid = w.meta["categories"].index("person")
        ts, masks, scores = [], [], []
        prev = None
        for t, fr in _frames(video):
            x = torch.from_numpy(fr).permute(2, 0, 1).float().div(255).to(dev)
            det = net([x])[0]
            keep = (det["labels"] == pid) & (det["scores"] > 0.6)
            m = np.zeros(fr.shape[:2], bool)
            sc = 0.0
            if keep.any():
                ms = (det["masks"][keep, 0] > 0.5).cpu().numpy()
                ss = det["scores"][keep].cpu().numpy()
                if prev is not None and prev.any():
                    ov = np.array([(mm & prev).sum() / max(1, (mm | prev).sum()) for mm in ms])
                    k = int(np.argmax(ov + 0.1 * ss)) if ov.max() > 0.05 else int(np.argmax(ss * ms.reshape(len(ms), -1).sum(1)))
                else:  # first sighting: the biggest confident person
                    k = int(np.argmax(ss * ms.reshape(len(ms), -1).sum(1)))
                m, sc = ms[k], float(ss[k])
            prev = m if m.any() else prev
            ts.append(t)
            masks.append(np.packbits(m))
            scores.append(sc)
        return {"t": np.array(ts), "masks": masks, "score": np.array(scores), "shape": fr.shape[:2]}
    res = cached(Path(cache) / f"track_{video_sig([video], {'w': WORK_W, 's': STEP, 'v': 1})}.pkl", _do)
    log(f"  {Path(video).name}: {len(res['t'])} frames, person in {np.mean(res['score'] > 0):.0%}")
    return res


def unpack(track, i):
    h, w = track["shape"]
    return np.unpackbits(track["masks"][i])[:h * w].reshape(h, w).astype(bool)


@torch.no_grad()
def torso_depth(video, track, plate_depth, cache, device=None, log=print):
    """Per frame: camera-frame 3D point of the person's torso from monocular depth on the live
    frame, put on the plate-depth scale by a robust fit on the pixels away from the person. Used
    when the feet are hidden (the camera is near head height, so a height prior is ill-conditioned)."""
    def _do():
        from .depth import DepthModel
        from PIL import Image
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dm = DepthModel(dev)
        h, w = track["shape"]
        K = intrinsics(w, h)
        out = np.full((len(track["t"]), 3), np.nan)
        k5 = np.ones((15, 15), np.uint8)
        for i, (t, fr) in enumerate(_frames(video)):
            if i >= len(out):
                break
            m = unpack(track, i)
            if m.sum() < 400:
                continue
            x = dm.proc(images=Image.fromarray(fr), return_tensors="pt").to(dev)
            d = dm.net(**x).predicted_depth[None]
            d = torch.nn.functional.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            bg = ~(cv2.dilate(m.astype(np.uint8), k5) > 0)
            idx = np.random.default_rng(i).choice(np.flatnonzero(bg), min(8000, int(bg.sum())), replace=False)
            sc, b, _ = fit_scale_shift(d.ravel()[idx], plate_depth.ravel()[idx])
            d = sc * d + b
            ys, xs = np.nonzero(m)
            y0, y1 = ys.min(), ys.max()
            core = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            vv, uu = np.nonzero(core)
            sel = (vv > y0 + 0.2 * (y1 - y0)) & (vv < y0 + 0.55 * (y1 - y0))
            if sel.sum() < 30:
                continue
            z = np.median(d[vv[sel], uu[sel]])
            u, v = np.median(uu[sel]), np.median(vv[sel])
            out[i] = ((u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z)
        return out
    res = cached(Path(cache) / f"torso_{video_sig([video], {'w': WORK_W, 's': STEP, 'v': 1})}.pkl", _do)
    return res


@torch.no_grad()
def keypoint_track(video, track, cache, device=None):
    """COCO keypoints (Keypoint R-CNN) of the tracked person per frame: (n, 17, 3) x, y, score."""
    def _do():
        from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
                                        box_score_thresh=0.5).to(dev).eval()
        out = np.full((len(track["t"]), 17, 3), np.nan)
        for i, (t, fr) in enumerate(_frames(video)):
            if i >= len(out):
                break
            m = unpack(track, i)
            if m.sum() < 400:
                continue
            det = net([torch.from_numpy(fr).permute(2, 0, 1).float().div(255).to(dev)])[0]
            if not len(det["boxes"]):
                continue
            # the detection that overlaps the tracked mask most
            best, bo = -1, 0.0
            for j, b in enumerate(det["boxes"].cpu().numpy().astype(int)):
                x0, y0, x1, y1 = np.clip(b, 0, [m.shape[1], m.shape[0], m.shape[1], m.shape[0]])
                ov = m[y0:y1, x0:x1].sum() / max(1, m.sum())
                if ov > bo:
                    best, bo = j, ov
            if bo > 0.5:
                kp = det["keypoints"][best].cpu().numpy()
                kp[:, 2] = det["keypoints_scores"][best].cpu().numpy()
                out[i] = kp
        return out
    return cached(Path(cache) / f"kp_{video_sig([video], {'w': WORK_W, 's': STEP, 'v': 1})}.pkl", _do)


# ---------------------------------------------------------------- shared floor frame

class Scene:
    """Camera intrinsics, floor plane and room frame from the reference (clip01) plate depth."""

    def __init__(self, ref_depth):
        h, w = ref_depth.shape
        self.h, self.w = h, w
        self.K = intrinsics(w, h)
        vw = {"K": self.K, "R": np.eye(3), "t": np.zeros(3), "depth": ref_depth, "rgb": np.zeros((h, w, 3), np.uint8)}
        xyz, _ = G.backproject(vw, stride=2)
        self.n, self.d = G.estimate_floor(xyz, np.array([0, -1.0, 0]))   # n.x + d = height (camera frame)
        self.Rr, self.origin = room_frame([dict(vw, rec="before")], self.n, self.d)
        self.cam_height = float(self.d)   # camera at x=0

    def rays(self, uv):
        uv = np.atleast_2d(uv).astype(np.float64)
        return np.stack([(uv[:, 0] - self.K[0, 2]) / self.K[0, 0], (uv[:, 1] - self.K[1, 2]) / self.K[1, 1],
                         np.ones(len(uv))], 1)

    def floor_hit(self, uv, height=0.0):
        """Point on the ray through pixel uv at `height` above the floor (camera frame); nan if behind."""
        r = self.rays(uv)
        t = (height - self.d) / (r @ self.n)
        t[t <= 0] = np.nan
        return r * t[:, None]

    def ground_from_top(self, uv, H):
        """Floor point under a person of height H whose head top is at pixel uv."""
        return self.floor_hit(uv, H) - H * self.n

    def room(self, xyz):
        return to_room(xyz, self.Rr, self.origin)

    def cam(self, xyz_room):
        return xyz_room @ self.Rr + self.origin

    def project(self, xyz_room):
        pc = self.cam(xyz_room)
        uv = pc @ self.K.T
        return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6), pc[:, 2]

    def cloud(self, depth, stride=2):
        h, w = depth.shape
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        z = depth[::stride, ::stride]
        pc = np.stack([(u - self.K[0, 2]) / self.K[0, 0] * z, (v - self.K[1, 2]) / self.K[1, 1] * z, z], -1)
        return self.room(pc.reshape(-1, 3)), (v.ravel(), u.ravel())


def align_depth(depth, ref, valid):
    idx = np.random.default_rng(0).choice(np.flatnonzero(valid), min(20000, int(valid.sum())), replace=False)
    s, b, _ = fit_scale_shift(depth.ravel()[idx], ref.ravel()[idx])
    return s * depth + b


# ---------------------------------------------------------------- ground track

def pose_foot_uv(kp, k_ratio, min_score=1.0):
    """Foot pixel extrapolated from shoulders and hips: hip + k_ratio * (hip - shoulder) in rows."""
    sh, hp = kp[:, 5:7, :2].mean(1), kp[:, 11:13, :2].mean(1)
    ok = (np.nanmin(kp[:, [5, 6, 11, 12], 2], 1) > min_score)
    d = hp[:, 1] - sh[:, 1]
    ok &= d > 5
    uv = np.c_[hp[:, 0], hp[:, 1] + k_ratio * d]
    uv[~ok] = np.nan
    return uv, d


def feet_visible(track, height_map):
    """Per frame: foot pixel (bottom of the person mask) and whether the feet are really visible."""
    h, w = track["shape"]
    n = len(track["t"])
    foot_uv = np.full((n, 2), np.nan)
    ok = np.zeros(n, bool)
    for i in range(n):
        m = unpack(track, i)
        if m.sum() < 400:
            continue
        ys, xs = np.nonzero(m)
        y1, hgt = ys.max(), ys.max() - ys.min()
        band = ys > y1 - max(3, 0.04 * hgt)
        foot_uv[i] = (np.median(xs[band]), y1)
        # not cut by the image border, and the plate right below the feet is floor-level
        # (0.25 m tolerance: monocular depth bends the floor up by ~0.15 m near the image bottom)
        below = height_map[min(h - 1, y1 + 2):min(h, y1 + 8), max(0, int(foot_uv[i, 0]) - 6):int(foot_uv[i, 0]) + 7]
        ok[i] = (y1 < h - 4 and below.size > 0 and np.nanmedian(below) < 0.25 and xs.min() > 1 and xs.max() < w - 2)
    return foot_uv, ok


def ground_track(track, kp, scene, height_map, k_ratio, smooth_s=0.4):
    """Floor (x, y) per frame in the room frame + which cue gave it.

    'foot': bottom of the person mask, ray intersected with the floor plane.
    'pose': feet hidden (image border / furniture) -> foot pixel extrapolated from the shoulder and
    hip keypoints (hip + k_ratio * torso length, k_ratio calibrated on frames with visible feet),
    then the same floor-plane intersection. The mask top is not used: faces are pixelated, so the
    head is often missing from the mask, and the camera is near head height anyway."""
    t = track["t"]
    n = len(t)
    foot_uv, foot_ok = feet_visible(track, height_map)
    pose_uv, torso = pose_foot_uv(kp, k_ratio)
    # mask ends well above where the body says the feet are: cut by furniture the depth test missed
    short = (pose_uv[:, 1] - foot_uv[:, 1]) > 0.5 * torso
    foot_ok &= ~np.nan_to_num(short, nan=0).astype(bool)
    gf = scene.room(scene.floor_hit(foot_uv))
    gp = scene.room(scene.floor_hit(pose_uv))
    sc = track["score"]
    xy = np.full((n, 2), np.nan)
    src = np.array([""] * n, dtype=object)
    use_f = foot_ok & np.isfinite(gf).all(1)
    use_p = ~use_f & np.isfinite(gp).all(1) & (sc > 0.6)
    xy[use_f], src[use_f] = gf[use_f, :2], "foot"
    xy[use_p], src[use_p] = gp[use_p, :2], "pose"
    # reject jumps (bad masks / keypoints): > 0.6 m from the running median of neighbours
    good = np.isfinite(xy).all(1)
    for _ in range(2):
        idx = np.flatnonzero(good)
        if len(idx) < 5:
            break
        med = np.array([np.median(xy[idx[max(0, k - 4):k + 5]], 0) for k in range(len(idx))])
        bad = np.linalg.norm(xy[idx] - med, axis=1) > 0.6
        good[idx[bad]] = False
    xy[~good] = np.nan
    src[~good] = ""
    sm = xy.copy()
    idx = np.flatnonzero(good)
    for k in idx:
        wts = np.exp(-0.5 * ((t[idx] - t[k]) / smooth_s) ** 2)
        sm[k] = (wts[:, None] * xy[idx]).sum(0) / wts.sum()
    return {"t": t, "xy": sm, "raw": xy, "src": src, "foot_uv": foot_uv, "pose_uv": pose_uv}


def calibrate_k(tracks, kps, hmaps):
    """Median (foot row - hip row) / (hip row - shoulder row) over frames with visible feet."""
    ks = []
    for tr, kp, hm in zip(tracks, kps, hmaps):
        fuv, ok = feet_visible(tr, hm)
        sh, hp = kp[:, 5:7, :2].mean(1), kp[:, 11:13, :2].mean(1)
        d = hp[:, 1] - sh[:, 1]
        g = ok & (np.nanmin(kp[:, [5, 6, 11, 12], 2], 1) > 1.0) & (d > 5)
        ks += list((fuv[g, 1] - hp[g, 1]) / d[g])
    return float(np.median(ks)), len(ks)


def path_stats(g, stop_speed=0.15):
    ok = np.isfinite(g["xy"]).all(1)
    t, xy = g["t"][ok], g["xy"][ok]
    if len(t) < 2:
        return None
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    L = float(seg.sum())
    dur = float(t[-1] - t[0])
    v = seg / np.maximum(np.diff(t), 1e-3)
    moving = v > stop_speed
    # stop point: where the person comes to rest (last sample after the final moving step)
    last_move = np.flatnonzero(moving)
    stop_i = int(last_move[-1] + 1) if len(last_move) else len(t) - 1
    return {"length_m": round(L, 2), "duration_s": round(dur, 2), "mean_speed_mps": round(L / max(dur, 1e-3), 2),
            "moving_speed_mps": round(float(np.median(v[moving])) if moving.any() else 0.0, 2),
            "start_xy": [round(float(a), 2) for a in xy[0]], "stop_xy": [round(float(a), 2) for a in xy[stop_i]],
            "stop_t_s": round(float(t[stop_i]), 2),
            "coverage": round(float(ok.mean()), 2),
            "from_feet": round(float(np.mean(g["src"] == "foot")), 2),
            "from_pose": round(float(np.mean(g["src"] == "pose")), 2)}


# ---------------------------------------------------------------- occupancy

class Grid:
    def __init__(self, xlim=(0.0, 12.0), ylim=(-6.0, 6.0), cell=CELL):
        self.x0, self.y0, self.cell = xlim[0], ylim[0], cell
        self.nx, self.ny = int(round((xlim[1] - xlim[0]) / cell)), int(round((ylim[1] - ylim[0]) / cell))

    def idx(self, xy):
        xy = np.atleast_2d(xy)
        return np.stack([np.floor((xy[:, 0] - self.x0) / self.cell), np.floor((xy[:, 1] - self.y0) / self.cell)],
                        1).astype(int)

    def xy(self, ij):
        ij = np.atleast_2d(ij)
        return np.stack([self.x0 + (ij[:, 0] + 0.5) * self.cell, self.y0 + (ij[:, 1] + 0.5) * self.cell], 1)

    def inside(self, ij):
        return (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.ny)


UNKNOWN, FREE, OCC = 0, 1, 2


def occupancy(scene, depth, grid, never=None, max_range=9.0, stride=2):
    """Grid of UNKNOWN / FREE / OCC (indexed [ix, iy]) from one plate depth map.

    OCC: >= 2 surface points in the obstacle height band. FREE: floor-height points, plus cells a
    pixel ray crosses below the top of the band before it hits a surface (nothing there in the band
    at that height). Pixels on depth discontinuities are skipped (monocular depth smears there)."""
    dmin, dmax = G.window_minmax(depth, 5)
    edge = (dmax - dmin) > 0.04 * depth
    pts, (v, u) = scene.cloud(depth, stride)
    keep = np.isfinite(pts).all(1) & (np.hypot(pts[:, 0], pts[:, 1]) < max_range) & ~edge[v, u]
    if never is not None:
        keep &= ~never[v, u]
    pts = pts[keep]
    ij = grid.idx(pts[:, :2])
    ins = grid.inside(ij)
    z = pts[:, 2]
    occ_cnt = np.zeros((grid.nx, grid.ny), int)
    free_cnt = np.zeros((grid.nx, grid.ny), int)
    ob = ins & (z > BAND[0]) & (z < BAND[1])
    fl = ins & (np.abs(z) < 0.06)
    np.add.at(occ_cnt, (ij[ob, 0], ij[ob, 1]), 1)
    np.add.at(free_cnt, (ij[fl, 0], ij[fl, 1]), 1)
    # ray casting: camera centre -> surface point, free where the ray is inside the band
    cam = scene.room(np.zeros((1, 3)))[0]
    sub = pts[::3]
    L = np.linalg.norm(sub - cam, axis=1)
    for f in np.arange(0.02, 0.98, 0.02):
        q = cam + f * (sub - cam)
        ok = (q[:, 2] > 0) & (q[:, 2] < BAND[1]) & (L * (1 - f) > 0.15)
        qi = grid.idx(q[ok, :2])
        qi = qi[grid.inside(qi)]
        np.add.at(free_cnt, (qi[:, 0], qi[:, 1]), 1)
    g = np.zeros((grid.nx, grid.ny), np.uint8)
    g[free_cnt > 0] = FREE
    occ = occ_cnt >= 2
    occ = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    g[occ] = OCC
    return g


def inflate(g, radius=ROBOT_R, cell=CELL):
    r = int(np.ceil(radius / cell))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate((g == OCC).astype(np.uint8), k) > 0


def clearance(g, cell=CELL):
    """Distance (m) from each cell to the nearest obstacle cell."""
    return cv2.distanceTransform((g != OCC).astype(np.uint8), cv2.DIST_L2, 5) * cell


# ---------------------------------------------------------------- planner

def astar(blocked, cost, start, goal):
    nx, ny = blocked.shape
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    gs = {start: 0.0}
    par = {}
    hq = [(0.0, start)]
    gx, gy = goal
    while hq:
        f, c = heapq.heappop(hq)
        if c == goal:
            path = [c]
            while c in par:
                c = par[c]
                path.append(c)
            return np.array(path[::-1])
        g0 = gs[c]
        if f - np.hypot(c[0] - gx, c[1] - gy) > g0 + 1e-9:
            continue
        for dx, dy, w in moves:
            a, b = c[0] + dx, c[1] + dy
            if not (0 <= a < nx and 0 <= b < ny) or blocked[a, b]:
                continue
            if dx and dy and (blocked[c[0] + dx, c[1]] or blocked[c[0], c[1] + dy]):
                continue
            ng = g0 + w * cost[a, b]
            if ng < gs.get((a, b), np.inf):
                gs[(a, b)] = ng
                par[(a, b)] = c
                heapq.heappush(hq, (ng + np.hypot(a - gx, b - gy), (a, b)))
    return None


def nearest_free(blocked, ij):
    if not blocked[ij[0], ij[1]]:
        return tuple(int(a) for a in ij)
    fr = np.argwhere(~blocked)
    k = np.argmin(((fr - np.asarray(ij)) ** 2).sum(1))
    return tuple(int(a) for a in fr[k])


def shortcut(path_xy, blocked, grid):
    """Line-of-sight smoothing of a grid path."""
    if path_xy is None or len(path_xy) < 3:
        return path_xy
    out = [path_xy[0]]
    i = 0
    while i < len(path_xy) - 1:
        j = len(path_xy) - 1
        while j > i + 1:
            n = int(np.ceil(np.linalg.norm(path_xy[j] - path_xy[i]) / (grid.cell / 2))) + 1
            ij = grid.idx(np.linspace(path_xy[i], path_xy[j], n))
            if not blocked[ij[:, 0], ij[:, 1]].any():
                break
            j -= 1
        out.append(path_xy[j])
        i = j
    return np.array(out)


def plan(g, grid, start_xy, goal_xy, radius=ROBOT_R, unknown_cost=3.0):
    blocked = inflate(g, radius, grid.cell)
    # never plan outside what the camera saw at all: unknown far from any observed cell is blocked
    seen = cv2.dilate((g != UNKNOWN).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    blocked_all = blocked | ~seen
    cost = np.where(g == UNKNOWN, unknown_cost, 1.0)
    s = nearest_free(blocked_all, grid.idx(start_xy)[0])
    e = nearest_free(blocked_all, grid.idx(goal_xy)[0])
    p = astar(blocked_all, cost, s, e)
    if p is None:
        return None, blocked_all
    return shortcut(grid.xy(p), blocked_all, grid), blocked_all


def polyline_len(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if p is not None and len(p) > 1 else float("nan")


def resample(p, step=0.05):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    q = np.arange(0, s[-1] + 1e-9, step)
    return np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1)


def path_clearance(p, clr, grid):
    q = resample(p)
    ij = grid.idx(q)
    ins = grid.inside(ij)
    return clr[ij[ins, 0], ij[ins, 1]]
