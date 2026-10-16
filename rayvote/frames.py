"""Pick sharp, time-spread keyframes from a video."""
from pathlib import Path

import cv2
import numpy as np


class InputError(RuntimeError):
    pass


def select_frames(video, out_dir, n=40, max_side=1024, prefix="f"):
    """Split the video into n time bins and keep the sharpest frame of each.

    Sharpness = variance of the Laplacian on a downscaled grey image.
    Returns a list of dicts {name, frame_index}.
    """
    video = Path(video)
    if not video.exists():
        raise InputError(f"video not found: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise InputError(f"cannot open video (unsupported codec?): {video}")
    sharp = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (320, int(320 * fr.shape[0] / fr.shape[1]))), cv2.COLOR_BGR2GRAY)
        sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
    cap.release()
    total = len(sharp)
    if total < 10:
        raise InputError(f"{video} has only {total} readable frames; need a few seconds of video")
    n = min(n, total)
    bins = np.linspace(0, total, n + 1).astype(int)
    keep = [int(lo + np.argmax(sharp[lo:hi])) for lo, hi in zip(bins[:-1], bins[1:]) if hi > lo]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    want, idx, res = set(keep), 0, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in want:
            s = max_side / max(fr.shape[:2])
            if s < 1:
                fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            name = f"{prefix}_{idx:05d}.jpg"
            cv2.imwrite(str(out_dir / name), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            res.append({"name": name, "frame_index": idx})
        idx += 1
    cap.release()
    return res
