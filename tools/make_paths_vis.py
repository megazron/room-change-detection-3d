"""Occupancy, plans, metrics and figures for tools/make_paths.py."""
import io
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rayvote import paths as P  # noqa: E402

COLS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
S_COL, E_COL, ROBOT_COL = "#1a9e3a", "#d01c1c", "#7b2cbf"


# ---------------------------------------------------------------- maps

def build_maps(scene, depths, plates, grid):
    raw = [P.occupancy(scene, d, grid, c["never"]) for d, c in zip(depths, plates)]
    maps = []
    for k, g in enumerate(raw):
        # cells this clip never saw (person in the way, plate gaps): borrow the other clips' majority
        others = [r for j, r in enumerate(raw) if j != k]
        occ = sum((r == P.OCC).astype(int) for r in others)
        fr = sum((r == P.FREE).astype(int) for r in others)
        g = g.copy()
        fill = (g == P.UNKNOWN) & (occ + fr > 0)
        g[fill & (occ > fr)] = P.OCC
        g[fill & (occ <= fr)] = P.FREE
        maps.append(g)
    return raw, maps


def changed_cells(prev, cur):
    """Obstacle cells added / removed between two maps (opened to drop single-cell flicker)."""
    k = np.ones((3, 3), np.uint8)
    a = cv2.morphologyEx(((cur == P.OCC) & (prev == P.FREE)).astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    r = cv2.morphologyEx(((prev == P.OCC) & (cur == P.FREE)).astype(np.uint8), cv2.MORPH_OPEN, k) > 0

    def big(m, n=40):
        nn, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        return np.isin(lab, np.flatnonzero(st[1:, cv2.CC_STAT_AREA] >= n) + 1)
    return big(a), big(r)


# ---------------------------------------------------------------- top-down drawing

class TopDown:
    def __init__(self, grid, maps, pad=0.6):
        known = np.zeros((grid.nx, grid.ny), bool)
        for g in maps:
            known |= g != P.UNKNOWN
        ij = np.argwhere(known)
        lo, hi = grid.xy(ij.min(0))[0] - pad, grid.xy(ij.max(0))[0] + pad
        self.grid = grid
        self.xlim, self.ylim = (max(lo[0], grid.x0 - 0.3), hi[0]), (lo[1], hi[1])

    def ax_xy(self, xy):
        """room (x fwd, y left) -> plot (right, up)."""
        xy = np.atleast_2d(xy)
        return -xy[:, 1], xy[:, 0]

    def draw_map(self, ax, g, inflated=None, add=None, rem=None):
        img = np.full(g.shape + (3,), 0.86)
        img[g == P.FREE] = (1, 1, 1)
        if inflated is not None:
            img[inflated & (g != P.OCC)] = (0.93, 0.85, 0.85)
        img[g == P.OCC] = (0.25, 0.25, 0.28)
        if add is not None:
            img[add] = (0.15, 0.7, 0.25)
        if rem is not None:
            img[rem] = (0.95, 0.55, 0.1)
        gr = self.grid
        ax.imshow(img[:, ::-1], origin="lower", interpolation="nearest",
                  extent=[-(gr.y0 + gr.ny * gr.cell), -gr.y0, gr.x0, gr.x0 + gr.nx * gr.cell])
        ax.set_xlim(-self.ylim[1], -self.ylim[0])
        ax.set_ylim(self.xlim[0], self.xlim[1])
        ax.set_aspect("equal")
        ax.plot([0], [0], marker="^", color="k", ms=9)
        ax.annotate("camera", (0, 0), xytext=(6, -2), textcoords="offset points", fontsize=8)
        ax.set_xlabel("right of camera (m)")
        ax.set_ylabel("forward from camera (m)")

    def path(self, ax, xy, **kw):
        a, b = self.ax_xy(xy)
        ax.plot(a, b, **kw)

    def se(self, ax, s, e, col_s=S_COL, col_e=E_COL, ms=11):
        for p, lab, c in ((s, "S", col_s), (e, "E", col_e)):
            a, b = self.ax_xy(p)
            ax.plot(a, b, "o", ms=ms, mfc=c, mec="white", mew=1.5, zorder=10)
            ax.text(a[0], b[0], lab, color="white", fontsize=7, ha="center", va="center", weight="bold", zorder=11)


def fig_to_rgb(fig, dpi=80):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(frames, path, ms, colors=64):
    # one shared palette from a strip of sample frames, so small coloured overlays keep their colour
    picks = frames[::max(1, len(frames) // 6)] + [frames[-1]]
    keys = [(160, 60, 220), (123, 44, 191), (255, 140, 30), (26, 158, 58), (208, 28, 28), (255, 255, 255), (0, 0, 0)]
    sw = 40
    strip = Image.new("RGB", (frames[0].width, frames[0].height * len(picks) + sw * len(keys)))
    for i, f in enumerate(picks):
        strip.paste(f, (0, i * f.height))
    for j, c in enumerate(keys):   # large swatches of the overlay colours so they get their own entries
        strip.paste(c, (0, frames[0].height * len(picks) + j * sw, frames[0].width, frames[0].height * len(picks) + (j + 1) * sw))
    ref = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True)


# ---------------------------------------------------------------- main build

def human_segment(g, ep):
    """Smoothed human track between the start and end detections (valid samples only)."""
    k0, k1 = ep["start_k"], ep["end_k"]
    xy, t = g["xy"][k0:k1 + 1], g["t"][k0:k1 + 1]
    ok = np.isfinite(xy).all(1)
    xy, t = xy[ok], t[ok]
    xy = np.concatenate([[ep["start_xy"]], xy, [ep["end_xy"]]])
    t = np.concatenate([[ep["start_t"]], t, [ep["end_t"]]])
    return xy, t


def build_all(videos, names, plates, depths, scene, tracks, grounds, eps, out, log):
    grid = P.Grid()
    raw, maps = build_maps(scene, depths, plates, grid)
    td = TopDown(grid, maps)
    rows, plans, humans, blockeds = [], [], [], []
    for k, (n, g, ep, m) in enumerate(zip(names, grounds, eps, maps)):
        hxy, ht = human_segment(g, ep)
        plan, blocked, snap = P.plan(m, grid, ep["start_xy"], ep["end_xy"])
        clr = P.clearance(m, grid.cell)
        hc = P.path_clearance(hxy, clr, grid)
        rc = P.path_clearance(plan, clr, grid) if plan is not None else np.array([np.nan])
        st = P.path_stats(g)
        L = P.polyline_len(hxy)
        dur = ht[-1] - ht[0]
        prev_blocked = None
        if k > 0 and plans[-1] is not None:
            q = P.resample(plans[-1])
            ij = grid.idx(q)
            ins = grid.inside(ij)
            prev_blocked = bool(blocked[ij[ins, 0], ij[ins, 1]].any())
        row = {"clip": n, "start_xy": [round(float(v), 2) for v in ep["start_xy"]],
               "end_xy": [round(float(v), 2) for v in ep["end_xy"]],
               "start_t_s": round(ep["start_t"], 2), "end_t_s": round(ep["end_t"], 2),
               "start_frame": int(round(ep["start_t"] * 30)), "end_frame": int(round(ep["end_t"] * 30)),
               "start_conf": ep["start_conf"], "end_conf": ep["end_conf"],
               "human_length_m": round(L, 2), "human_duration_s": round(dur, 2),
               "human_speed_mps": round(L / max(dur, 1e-3), 2),
               "human_min_clearance_m": round(float(np.nanmin(hc)), 2),
               "human_frac_in_inflated": round(float(np.mean(hc < P.ROBOT_R)), 2),
               "robot_length_m": round(P.polyline_len(plan), 2) if plan is not None else None,
               "robot_min_clearance_m": round(float(np.nanmin(rc)), 2),
               "prev_plan_blocked": prev_blocked,
               "start_snap_m": round(snap[0], 2), "goal_snap_m": round(snap[1], 2), "track": st}
        rows.append(row)
        plans.append(plan)
        humans.append((hxy, ht))
        blockeds.append(P.inflate(m, P.ROBOT_R, grid.cell))
        log(f"{n}: human {L:.2f} m / {dur:.1f} s, robot {row['robot_length_m']} m, human min clr "
            f"{row['human_min_clearance_m']} m, robot min clr {row['robot_min_clearance_m']} m, prev blocked {prev_blocked}")

    # fixed task: the same start/goal (median over clips) on every map isolates the effect of the chairs
    S = np.median([e["start_xy"] for e in eps], 0)
    E = np.median([e["end_xy"] for e in eps], 0)
    fixed = []
    for k, m in enumerate(maps):
        p, _, _ = P.plan(m, grid, S, E)
        fixed.append(p)
        rows[k]["fixed_task_robot_length_m"] = round(P.polyline_len(p), 2) if p is not None else None
        if k:
            rows[k]["fixed_task_replanned"] = bool(_differs(fixed[k - 1], p))
    (out / "paths.json").write_text(json.dumps({"cell_m": grid.cell, "robot_radius_m": P.ROBOT_R,
                                                "obstacle_band_m": P.BAND,
                                                "camera_height_m": round(scene.cam_height, 2),
                                                "fixed_task": {"start": S.round(2).tolist(), "end": E.round(2).tolist()},
                                                "clips": rows}, indent=2))

    person_paths_png(td, maps[0], names, humans, eps, out / "person_paths.png")
    replan_gif(td, grid, maps, names, humans, plans, eps, blockeds, rows, out / "replan.gif")
    k_ov = names.index("clip05") if "clip05" in names else 0
    overlay_gif(videos[k_ov], names[k_ov], tracks[k_ov], grounds[k_ov], humans[k_ov], plans[k_ov], eps[k_ov], scene,
                out / "path_overlay.gif")
    paths_3d_gif(scene, depths[0], plates[0]["plate"], names, humans, plans, eps, out / "paths_3d.gif")
    return rows


def _differs(a, b, tol=0.3):
    if a is None or b is None:
        return a is not b
    qa, qb = P.resample(a), P.resample(b)
    d = np.sqrt(((qa[:, None] - qb[None]) ** 2).sum(-1))
    return max(d.min(1).max(), d.min(0).max()) > tol


def person_paths_png(td, g0, names, humans, eps, path):
    fig, ax = plt.subplots(figsize=(7, 7.5))
    td.draw_map(ax, g0)
    for k, (n, (hxy, _), ep) in enumerate(zip(names, humans, eps)):
        td.path(ax, hxy, color=COLS[k], lw=2.2, label=n)
        td.se(ax, ep["start_xy"][None], ep["end_xy"][None], col_s=COLS[k], col_e=COLS[k], ms=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title("Person paths, all six clips, on the clip01 occupancy map\n"
                 "dark = obstacle 0.1-1.2 m above floor, grey = not observed, S/E = start/rest point", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def replan_gif(td, grid, maps, names, humans, plans, eps, blockeds, rows, path, n_anim=14):
    frames, durs = [], []
    for k, (n, m) in enumerate(zip(names, maps)):
        add = rem = None
        if k:
            add, rem = changed_cells(maps[k - 1], m)
        hxy, ht = humans[k]
        plan = plans[k]
        r = rows[k]
        pq = P.resample(plan) if plan is not None else None
        hs = np.linspace(0, 1, len(hxy))
        for j in range(n_anim + 1):
            f = j / n_anim
            fig, ax = plt.subplots(figsize=(6, 6.4))
            td.draw_map(ax, m, inflated=blockeds[k], add=add, rem=rem)
            if k and plans[k - 1] is not None:
                td.path(ax, plans[k - 1], color=ROBOT_COL, lw=1.2, ls=":", alpha=0.6)
            td.path(ax, hxy, color="#1f77b4", lw=2, alpha=0.9)
            if plan is not None:
                td.path(ax, plan, color=ROBOT_COL, lw=2.2, ls="--")
            td.se(ax, eps[k]["start_xy"][None], eps[k]["end_xy"][None], ms=13)
            # animated: human dot along its path, robot disc along the plan (same normalised time)
            hp = np.array([np.interp(f, hs, hxy[:, 0]), np.interp(f, hs, hxy[:, 1])])
            a, b = td.ax_xy(hp)
            ax.plot(a, b, "o", color="#1f77b4", ms=7, mec="white", zorder=8)
            if pq is not None:
                rp = pq[min(len(pq) - 1, int(round(f * (len(pq) - 1))))]
                a, b = td.ax_xy(rp)
                ax.add_patch(plt.Circle((a[0], b[0]), P.ROBOT_R, color=ROBOT_COL, alpha=0.75, zorder=8))
            blk = r["prev_plan_blocked"]
            sub = ("" if blk is None else ("previous plan BLOCKED: replan" if blk else "previous plan still clear"))
            short = f", stops {r['goal_snap_m']} m short of E" if r["goal_snap_m"] > 0.1 else ""
            ax.set_title(f"{n}: human {r['human_length_m']} m (blue), robot {r['robot_length_m']} m (purple){short}\n"
                         f"green/orange = obstacle added/removed vs previous clip. {sub}", fontsize=8)
            fig.tight_layout()
            frames.append(fig_to_rgb(fig, dpi=72))
            durs.append(1400 if j == n_anim else 110)
    pal = [f.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)


def overlay_gif(video, name, track, g, human, plan, ep, scene, path, n=40, scale=0.75):
    from make_paths import read_frame
    hxy, ht = human
    t0, t1 = ep["start_t"], ep["end_t"] + 0.8
    ts = np.linspace(max(0, t0 - 0.5), t1, n)
    pq = P.resample(plan) if plan is not None else None
    # human foot track in the image: project the smoothed floor track
    ok = np.isfinite(g["xy"]).all(1)
    frames = []
    for t in ts:
        im = read_frame(video, int(round(t * 30)))
        sel = ok & (g["t"] <= t)
        if sel.sum() > 1:
            uv, _ = scene.project(np.c_[g["xy"][sel], np.zeros(sel.sum())])
            cv2.polylines(im, [uv.astype(np.int32)], False, (30, 140, 255), 3, cv2.LINE_AA)
        if pq is not None:
            uv, z = scene.project(np.c_[pq, np.zeros(len(pq))])
            good = z > 0.2
            cv2.polylines(im, [uv[good].astype(np.int32)], False, (191, 44, 123), 3, cv2.LINE_AA)
            f = np.clip((t - t0) / max(ep["end_t"] - t0, 1e-3), 0, 1)
            rp = pq[int(round(f * (len(pq) - 1)))]
            ang = np.linspace(0, 2 * np.pi, 40)
            circ = np.c_[rp[0] + P.ROBOT_R * np.cos(ang), rp[1] + P.ROBOT_R * np.sin(ang), np.zeros(40)]
            cu, _ = scene.project(circ)
            ov = im.copy()
            cv2.fillPoly(ov, [cu.astype(np.int32)], (220, 60, 160), cv2.LINE_AA)
            im = cv2.addWeighted(ov, 0.75, im, 0.25, 0)
            cv2.polylines(im, [cu.astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
        for key, lab, col in (("start", "S", (58, 158, 26)), ("end", "E", (28, 28, 208))):
            uv, _ = scene.project(np.array([[*ep[f"{key}_xy"], 0.0]]))
            from make_paths import marker
            marker(im, uv[0], lab, col)
        cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(im, f"{name}  t={t:4.1f}s   orange: person's foot track   purple: robot plan + 0.25 m disc",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    save_gif(frames, path, 120, colors=128)


def paths_3d_gif(scene, depth, plate, names, humans, plans, eps, path, n=36):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    pts, (v, u) = scene.cloud(depth, 3)
    col = plate[v, u] / 255.0
    keep = np.isfinite(pts).all(1) & (pts[:, 2] < 2.0) & (pts[:, 2] > -0.1) & (pts[:, 0] < 8) & (np.abs(pts[:, 1]) < 4)
    pts, col = pts[keep], col[keep]
    frames = []
    for i in range(n):
        fig = plt.figure(figsize=(6.4, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(-pts[:, 1], pts[:, 0], pts[:, 2], c=col, s=1.6, linewidths=0, depthshade=False)
        for k, (hxy, _) in enumerate(humans):
            ax.plot(-hxy[:, 1], hxy[:, 0], np.full(len(hxy), 0.03), color=COLS[k], lw=2.5)
        if plans[0] is not None:
            p = plans[0]
            ax.plot(-p[:, 1], p[:, 0], np.full(len(p), 0.05), color=ROBOT_COL, lw=3, ls="--")
        ax.scatter([0], [0], [scene.cam_height], marker="^", color="k", s=40)
        ax.view_init(elev=32, azim=-90 + 45 * np.sin(2 * np.pi * i / n))
        ax.set_xlim(-4, 4)
        ax.set_ylim(0.5, 8)
        ax.set_zlim(0, 2)
        ax.set_box_aspect((8, 7.5, 2), zoom=1.45)
        ax.set_axis_off()
        ax.set_title("clip01 plate point cloud, person paths (clips 1-6),\nclip01 robot plan (purple dashed)", fontsize=8)
        fig.tight_layout()
        frames.append(fig_to_rgb(fig, dpi=72))
    save_gif(frames, path, 110, colors=96)
