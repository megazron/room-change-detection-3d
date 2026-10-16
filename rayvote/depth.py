"""Monocular depth per keyframe, aligned to the SfM points."""
import numpy as np
import torch
from PIL import Image

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class DepthModel:
    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(MODEL)
        self.net = AutoModelForDepthEstimation.from_pretrained(MODEL).to(device).eval()

    @torch.no_grad()
    def __call__(self, path):
        img = Image.open(path).convert("RGB")
        x = self.proc(images=img, return_tensors="pt").to(self.device)
        d = self.net(**x).predicted_depth[None]
        d = torch.nn.functional.interpolate(d, size=img.size[::-1], mode="bilinear", align_corners=False)
        return d[0, 0].float().cpu().numpy()


def sparse_depth(cam, pts):
    """Project SfM points into a camera; returns (u, v, z) for points in front and in frame."""
    pc = pts @ cam["R"].T + cam["t"]
    z = pc[:, 2]
    ok = z > 1e-6
    uv = (pc[ok] @ cam["K"].T)
    uv = uv[:, :2] / uv[:, 2:3]
    z = z[ok]
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < cam["width"] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < cam["height"] - 1)
    return uv[inb], z[inb]


def fit_scale_shift(pred, target, iters=200, seed=0):
    """Robust fit target ~ s * pred + b (RANSAC on pairs, then LS on inliers)."""
    rng = np.random.default_rng(seed)
    n = len(pred)
    if n < 15:
        s = np.median(target / np.maximum(pred, 1e-6))
        return float(s), 0.0, n
    best, best_in = (np.median(target / pred), 0.0), None
    thr = 0.05 * np.median(target)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(pred[i] - pred[j]) < 1e-6:
            continue
        s = (target[i] - target[j]) / (pred[i] - pred[j])
        if s <= 0:
            continue
        b = target[i] - s * pred[i]
        inl = np.abs(s * pred + b - target) < thr
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (s, b), inl
    if best_in is not None and best_in.sum() >= 10:
        A = np.stack([pred[best_in], np.ones(best_in.sum())], 1)
        s, b = np.linalg.lstsq(A, target[best_in], rcond=None)[0]
        if s > 0:
            best = (s, b)
    return float(best[0]), float(best[1]), int(0 if best_in is None else best_in.sum())


def edge_mask(depth, rel=0.06):
    """False on depth discontinuities (flying pixels between foreground and background)."""
    import cv2
    k = np.ones((3, 3), np.uint8)
    mx = cv2.dilate(depth, k)
    mn = cv2.erode(depth, k)
    return (mx - mn) < rel * depth
