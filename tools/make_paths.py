"""Person paths, occupancy maps and robot plans for the six static-camera clips (docs/real/paths).

    python tools/make_paths.py            # needs `make real` (plates + depth cached in out/real/cache)
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rayvote import paths as P  # noqa: E402
from rayvote.pipeline import video_sig  # noqa: E402
from rayvote.static import WORK_W  # noqa: E402

COLS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_plates(videos, cache, log):
    out = []
    for v in videos:
        f = Path(cache) / f"clip_{video_sig([v], {'w': WORK_W, 'v': 3})}.pkl"
        if f.exists():
            out.append(pickle.load(open(f, "rb")))
        else:
            import torch
            from rayvote.static import make_models, process_clip
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            out.append(process_clip(v, Path(cache), make_models(dev), log))
    return out


# ---------------------------------------------------------------- endpoints

def runs(mask):
    """(start, end) index pairs of True runs, end exclusive."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def endpoints(g, score, rest_speed=0.2, rest_s=1.0, start_run=3, min_score=0.6):
    """START: first run of `start_run` consecutive confident frames with a floor fix (feet, or the
    height prior when the feet are hidden), median of those fixes. END: the last sustained rest
    window (smoothed speed < rest_speed for >= rest_s) in the second half of the clip, median foot
    point over it; if there is none, the last confident fix."""
    t, raw, sm, src = g["t"], g["raw"], g["xy"], g["src"]
    valid = np.isfinite(raw).all(1) & (score > min_score)
    vv = np.flatnonzero(valid)
    s_idx = vv[:start_run]
    for j in range(len(vv) - start_run + 1):   # first `start_run` fixes within 0.5 s of each other
        if t[vv[j + start_run - 1]] - t[vv[j]] <= 0.5:
            s_idx = vv[j:j + start_run]
            break
    start = np.median(raw[s_idx], 0)
    s_cue = "foot" if np.all(src[s_idx] == "foot") else ("pose" if np.all(src[s_idx] == "pose") else "mixed")
    s_conf = "high" if s_cue == "foot" else "medium"
    ok = np.isfinite(sm).all(1)
    vi = np.flatnonzero(ok)
    sp = np.full(len(t), np.nan)
    sp[vi] = np.linalg.norm(np.gradient(sm[vi], t[vi], axis=0), axis=1)
    # rest windows over the valid samples (short detection gaps < 1 s do not break a window)
    slow_v = sp[vi] < rest_speed
    win = None
    for a, b in runs(slow_v)[::-1]:
        seg = vi[a:b]
        if t[seg[-1]] - t[seg[0]] >= rest_s and t[seg[-1]] > 0.5 * t[-1] and np.diff(t[seg]).max(initial=0) < 1.0:
            win = seg
            break
    if win is None:
        last = np.flatnonzero(valid)[-start_run:]
        win, e_conf = last, "low"
    else:
        e_conf = "high"
    win = win[np.isfinite(raw[win]).all(1)]
    fw = win[src[win] == "foot"]
    use = fw if len(fw) >= max(3, 0.3 * len(win)) else win
    if use is win and e_conf == "high":
        e_conf = "medium"
    end = np.median(raw[use], 0)
    spread = float(np.median(np.linalg.norm(raw[use] - end, axis=1)))
    if spread > 0.2 and e_conf == "high":
        e_conf = "medium"
    k_s = int(s_idx[0])
    k_e = int(use[np.argmin(np.linalg.norm(raw[use] - end, axis=1))])
    return {"start_xy": start, "end_xy": end, "start_k": k_s, "end_k": k_e, "start_t": float(t[k_s]),
            "end_window_t": [float(t[win[0]]), float(t[win[-1]])], "end_t": float(t[k_e]),
            "start_cue": s_cue, "end_cue": "foot" if use is fw else "pose",
            "start_conf": s_conf, "end_conf": e_conf, "end_spread_m": spread}


# ---------------------------------------------------------------- drawing helpers

def read_frame(video, idx):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    h = int(round(fr.shape[0] * WORK_W / fr.shape[1]))
    return cv2.resize(fr, (WORK_W, h), interpolation=cv2.INTER_AREA)


def marker(img, uv, txt, col):
    u, v = int(round(uv[0])), int(round(uv[1]))
    cv2.circle(img, (u, v), 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (u, v), 5, col, -1, cv2.LINE_AA)
    cv2.putText(img, txt, (u + 8, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (u + 8, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)


def endpoint_image(video, track, g, ep, scene, out):
    fr_idx = np.round(track["t"] * 30).astype(int)
    tiles = []
    for key, lab, col in (("start", "S", (40, 170, 40)), ("end", "E", (40, 40, 220))):
        k = ep[f"{key}_k"]
        im = read_frame(video, fr_idx[k])
        m = P.unpack(track, k)
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnt, -1, (0, 220, 255), 1)
        xy = ep[f"{key}_xy"]
        uv, _ = scene.project(np.array([[xy[0], xy[1], 0.0]]))
        marker(im, uv[0], lab, col)
        src = g["src"][k] or "-"
        txt = (f"{lab}: t={track['t'][k]:.2f}s frame {fr_idx[k]}  ({xy[0]:.2f}, {xy[1]:.2f}) m  "
               f"cue={ep[key + '_cue']} conf={ep[key + '_conf']}")
        cv2.rectangle(im, (0, 0), (im.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(im, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    cv2.imwrite(str(out), np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--cache", default="out/real/cache")
    ap.add_argument("--out", default="docs/real/paths")
    ap.add_argument("--stage", default="all", choices=["endpoints", "all"])
    a = ap.parse_args()
    log = print
    videos = sorted(Path(a.data).glob("clip*.mp4"))
    names = [v.stem for v in videos]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plates = load_plates(videos, a.cache, log)
    ref = plates[0]
    scene = P.Scene(ref["depth"])
    log(f"floor: camera {scene.cam_height:.2f} m above the fitted floor plane")
    depths = [ref["depth"]] + [P.align_depth(c["depth"], ref["depth"], ~(c["never"] | ref["never"]))
                               for c in plates[1:]]
    # plate height above the floor for every pixel (feet-occlusion test)
    hmaps = []
    for d in depths:
        pts, _ = scene.cloud(d, 1)
        hmaps.append(pts[:, 2].reshape(d.shape))

    tracks = [P.person_track(v, Path(a.cache).parent / "paths_cache", log=log) for v in videos]
    kps = [P.keypoint_track(v, tr, Path(a.cache).parent / "paths_cache") for v, tr in zip(videos, tracks)]
    k_ratio, n_k = P.calibrate_k(tracks, kps, hmaps)
    log(f"pose foot extrapolation: hip + {k_ratio:.2f} x torso (from {n_k} frames with visible feet)")
    grounds = [P.ground_track(tr, kp, scene, hm, k_ratio) for tr, kp, hm in zip(tracks, kps, hmaps)]
    eps = []
    for v, n, tr, g in zip(videos, names, tracks, grounds):
        ep = endpoints(g, tr["score"])
        eps.append(ep)
        endpoint_image(v, tr, g, ep, scene, out / f"endpoints_{n}.jpg")
        log(f"{n}: S=({ep['start_xy'][0]:.2f},{ep['start_xy'][1]:.2f}) t={ep['start_t']:.2f}s "
            f"{ep['start_conf']}  E=({ep['end_xy'][0]:.2f},{ep['end_xy'][1]:.2f}) t={ep['end_t']:.2f}s "
            f"{ep['end_conf']} rest {ep['end_window_t'][0]:.1f}-{ep['end_window_t'][1]:.1f}s spread {ep['end_spread_m']:.2f} "
            f"cues {ep['start_cue']}/{ep['end_cue']}")
    if a.stage == "endpoints":
        return

    from make_paths_vis import build_all   # noqa: E402  (visuals + plans live next to this file)
    build_all(videos, names, plates, depths, scene, tracks, grounds, eps, out, log)


if __name__ == "__main__":
    main()
