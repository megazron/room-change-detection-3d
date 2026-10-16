"""Voxel fusion, free-space evidence, floor estimation and clustering."""
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

FREE, CONSISTENT, OCCLUDED = 0, 1, 2


def backproject(view, stride=1):
    d = view["depth"][::stride, ::stride]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    ok = np.isfinite(d) & (d > 0)
    u, v, z = u[ok] * stride, v[ok] * stride, d[ok]
    K = view["K"]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    pw = (pc - view["t"]) @ view["R"]  # R^T (pc - t)
    col = view["rgb"][::stride, ::stride][ok]
    return pw, col


def fuse(views, voxel, min_views=2):
    """Fuse depth maps into a voxel set. A voxel survives if >= min_views frames hit it."""
    pts, cols, vids = [], [], []
    for i, vw in enumerate(views):
        p, c = backproject(vw, stride=2)
        pts.append(p)
        cols.append(c)
        vids.append(np.full(len(p), i))
    pts, cols, vids = np.concatenate(pts), np.concatenate(cols), np.concatenate(vids)
    keys = np.floor(pts / voxel).astype(np.int64)
    keys -= keys.min(0)
    span = keys.max(0) + 1
    code = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    uniq, inv = np.unique(code, return_inverse=True)
    pair = np.unique(inv * len(views) + vids)
    nviews = np.bincount(pair // len(views), minlength=len(uniq))
    cnt = np.bincount(inv, minlength=len(uniq))
    pos = np.stack([np.bincount(inv, weights=pts[:, k], minlength=len(uniq)) for k in range(3)], 1)
    rgb = np.stack([np.bincount(inv, weights=cols[:, k], minlength=len(uniq)) for k in range(3)], 1)
    keep = nviews >= min_views
    return {"xyz": (pos / cnt[:, None])[keep], "rgb": (rgb / cnt[:, None])[keep].astype(np.uint8),
            "nviews": nviews[keep]}


def window_minmax(depth, k=5):
    d = np.nan_to_num(depth, nan=0.0).astype(np.float32)
    big = np.where(d > 0, d, 1e6).astype(np.float32)
    dmin = cv2.erode(big, np.ones((k, k), np.uint8))
    dmin[dmin >= 1e6] = 0
    dmax = cv2.dilate(d, np.ones((k, k), np.uint8))
    return dmin, dmax


def normals(xyz, k=12):
    from scipy.spatial import cKDTree
    _, idx = cKDTree(xyz).query(xyz, k=k)
    nb = xyz[idx] - xyz[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    return np.linalg.eigh(cov)[1][:, :, 0]


WINDOWS = (5, 9, 13, 17, 25)


def free_space_votes(xyz, views, nrm=None, m_abs=0.12, m_rel=0.08, radius=0.08, min_sin=0.3,
                     border=0.05, min_z=0.6):
    """For each 3D point, count per-view verdicts from the other recording.

    Per view, with z the point's depth and d the observed depth along that pixel ray:
      d > z + m  for the whole window  -> ray passes through the point: FREE
      d < z - m  for the whole window  -> something in front: OCCLUDED (no evidence)
      otherwise                        -> CONSISTENT
    with m = max(m_abs, m_rel * z) / max(sin(incidence), min_sin): a ray that grazes the
    surface turns a small depth error into a large along-ray error, so it needs a wider margin.
    "Whole window" is the point's projected disc of the given metric radius (its position
    uncertainty), so rays that only skim past an edge in the other view do not count as free.
    Points outside the frustum, near the image border, closer than min_z, or on invalid depth
    get no vote (monocular depth is least reliable at borders and at close range).
    """
    votes = np.zeros((len(xyz), 3), np.int32)
    for vw in views:
        if "dmin" not in vw:
            mm = [window_minmax(vw["depth"], k) for k in WINDOWS]
            vw["dmin"] = np.stack([a for a, _ in mm])
            vw["dmax"] = np.stack([b for _, b in mm])
        pc = xyz @ vw["R"].T + vw["t"]
        z = pc[:, 2]
        front = z > min_z
        uvw = pc @ vw["K"].T
        u = uvw[:, 0] / np.maximum(uvw[:, 2], 1e-6)
        v = uvw[:, 1] / np.maximum(uvw[:, 2], 1e-6)
        h, w = vw["depth"].shape
        b = border * w
        inside = front & (u >= b) & (u < w - 1 - b) & (v >= b) & (v < h - 1 - b)
        idx = np.nonzero(inside)[0]
        ui, vi = np.round(u[idx]).astype(int), np.round(v[idx]).astype(int)
        dc = vw["depth"][vi, ui]
        zi = z[idx]
        diam = 2 * vw["K"][0, 0] * radius / zi
        wi = np.clip(np.searchsorted(WINDOWS, diam), 0, len(WINDOWS) - 1)
        dmin, dmax = vw["dmin"][wi, vi, ui], vw["dmax"][wi, vi, ui]
        m = np.maximum(m_abs, m_rel * zi)
        if nrm is not None:
            c = -vw["R"].T @ vw["t"]
            ray = xyz[idx] - c
            ray /= np.linalg.norm(ray, axis=1, keepdims=True)
            m = m / np.maximum(np.abs(np.sum(ray * nrm[idx], 1)), min_sin)
        valid = np.isfinite(dc) & (dc > 0)
        free = valid & (dmin > 0) & (dmin > zi + m)
        occ = valid & ~free & (dmax > 0) & (dmax < zi - m)
        cons = valid & ~free & ~occ
        votes[idx[free], FREE] += 1
        votes[idx[occ], OCCLUDED] += 1
        votes[idx[cons], CONSISTENT] += 1
    return votes


def classify(votes, k_free=3, ratio=0.7, k_obs=2):
    """changed: enough free-space votes and they dominate; unobserved: too few votes either way."""
    f, c = votes[:, FREE], votes[:, CONSISTENT]
    changed = (f >= k_free) & (f >= ratio * (f + c))
    unobserved = (f + c) < k_obs
    return changed, unobserved & ~changed


def estimate_floor(xyz, up_guess, iters=400, thr=0.03, seed=0):
    """RANSAC plane with normal near up_guess and most inliers. Returns (n, d) with n.x + d = height."""
    rng = np.random.default_rng(seed)
    pts = xyz[rng.choice(len(xyz), min(len(xyz), 60000), replace=False)]
    up_guess = up_guess / np.linalg.norm(up_guess)
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ up_guess < 0:
            n = -n
        if n @ up_guess < np.cos(np.deg2rad(25)):
            continue
        d = -n @ a
        h = pts @ n + d
        cnt = np.sum(np.abs(h) < thr)
        # floor is the big plane with (almost) nothing below it
        below = np.mean(h < -0.15)
        score = cnt * (1.0 - 4 * below)
        if score > best_n:
            best_n, best = score, (n, d)
    n, d = best
    inl = np.abs(xyz @ n + d) < thr
    # refine with SVD on inliers
    q = xyz[inl]
    cen = q.mean(0)
    n2 = np.linalg.svd(q - cen)[2][-1]
    if n2 @ n < 0:
        n2 = -n2
    return n2, -n2 @ cen


def cluster(xyz, eps, min_samples=4):
    if len(xyz) == 0:
        return np.zeros(0, int)
    return DBSCAN(eps=eps, min_samples=min_samples).fit(xyz).labels_


def consistency_filter(views, rel=0.05, min_ok=2, n_nbr=8):
    """Keep depth pixels that agree with at least `min_ok` neighbouring frames of the same recording.

    Monocular depth errors are mostly per-frame; requiring agreement across frames removes
    the bulk of them before any cross-recording comparison happens.
    """
    centers = np.array([-v["R"].T @ v["t"] for v in views])
    dirs = np.array([v["R"][2] for v in views])
    out = []
    for i, vi in enumerate(views):
        h, w = vi["depth"].shape
        v_, u_ = np.mgrid[0:h, 0:w]
        d = vi["depth"]
        ok = np.isfinite(d)
        pts, _ = backproject(vi)
        uu, vv = u_[ok], v_[ok]
        good = np.zeros(len(pts), np.int32)
        cand = [j for j in range(len(views)) if j != i and dirs[i] @ dirs[j] > 0.5]
        cand = sorted(cand, key=lambda j: np.linalg.norm(centers[j] - centers[i]))[:n_nbr]
        for j in cand:
            vj = views[j]
            pc = pts @ vj["R"].T + vj["t"]
            z = pc[:, 2]
            uvw = pc @ vj["K"].T
            u = np.round(uvw[:, 0] / np.maximum(uvw[:, 2], 1e-6)).astype(int)
            v = np.round(uvw[:, 1] / np.maximum(uvw[:, 2], 1e-6)).astype(int)
            hh, ww = vj["depth"].shape
            ins = (z > 0.1) & (u >= 0) & (u < ww) & (v >= 0) & (v < hh)
            dj = np.full(len(z), np.nan)
            dj[ins] = vj["depth"][v[ins], u[ins]]
            good += (np.abs(dj - z) < rel * z)
        nd = np.full_like(d, np.nan)
        keep = good >= min_ok
        nd[vv[keep], uu[keep]] = d[vv[keep], uu[keep]]
        out.append(nd)
    for vw, nd in zip(views, out):
        vw["depth_raw"] = vw["depth"]
        vw["depth"] = nd


def merge_close(xyz, lab, dist):
    """Merge clusters whose centroids are closer than dist (one object seen as separate patches)."""
    ids = [l for l in np.unique(lab) if l >= 0]
    cen = {l: np.median(xyz[lab == l], 0) for l in ids}
    parent = {l: l for l in ids}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if np.linalg.norm(cen[a] - cen[b]) < dist:
                parent[find(b)] = find(a)
    return np.array([find(l) if l >= 0 else -1 for l in lab])
