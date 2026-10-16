"""Privacy preprocessing: pixelate heads in a video before it goes anywhere else.

Run once on the original recordings; only the output is kept. Three cues are combined
so a head is covered even when one detector misses it:
  1. MediaPipe pose landmarks -> head circle sized from the shoulders,
  2. the largest moving blob vs the clip median -> its top is taken as the head,
  3. MediaPipe face detections, accepted only inside moving regions (avoids posters).
A region is held for 15 frames after the last detection. Audio and metadata are not copied.

    pip install -r requirements-anonymize.txt   # separate venv: mediapipe pins its own numpy
    python tools/anonymize.py original.mp4 clip01.mp4 --models path/with/pose_full.task+face.task
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--models", required=True, help="directory with pose_full.task and face.task")
    ap.add_argument("--fps", type=float, default=30)
    a = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=f"{a.models}/pose_full.task"),
        running_mode=vision.RunningMode.VIDEO, num_poses=2,
        min_pose_detection_confidence=0.3, min_tracking_confidence=0.3))
    face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=f"{a.models}/face.task"), min_detection_confidence=0.3))

    cap = cv2.VideoCapture(a.src)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    H, W = frames[0].shape[:2]
    bg = np.median(np.stack(frames[::max(1, len(frames) // 40)]), 0).astype(np.uint8)
    out = cv2.VideoWriter(a.dst, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    last, hold, stats = [], 0, [0, 0, 0]
    for i, f in enumerate(frames):
        regions = []
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        r = pose.detect_for_video(rgb, int(i * 1000 / a.fps))
        for lm in r.pose_landmarks:
            p = np.array([[l.x * W, l.y * H] for l in lm])
            head = p[0:11]
            sh = np.linalg.norm(p[11] - p[12])
            c = head.mean(0)
            rad = max(1.15 * sh, 2.4 * np.ptp(head[:, 0]), 30)
            regions.append((c[0], c[1] - 0.1 * rad, rad))
        stats[0] += bool(r.pose_landmarks)

        fg = (cv2.absdiff(f, bg).max(2) > 40).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg = cv2.dilate(fg, np.ones((15, 15), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
        if n > 1:
            k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
            x, y, w, h, area = st[k]
            if area > 4000 and h > 0.65 * w:
                hw = max(w * 0.45, h * 0.11)
                ys, xs = np.where(lab[y:y + int(hw * 2) + 1, x:x + w] == k)
                cx = x + (xs.mean() if len(xs) else w / 2)
                regions.append((cx, y + hw, hw * 1.6))
                stats[2] += 1

        for det in face.detect(rgb).detections:
            b = det.bounding_box
            fx, fy = int(b.origin_x + b.width / 2), int(b.origin_y + b.height / 2)
            if 0 <= fx < W and 0 <= fy < H and fg[fy, fx]:
                regions.append((fx, fy, 1.3 * max(b.width, b.height)))
                stats[1] += 1

        if regions:
            last, hold = regions, 15
        elif hold > 0:
            regions, hold = last, hold - 1
        mask = np.zeros((H, W), np.uint8)
        for cx, cy, rad in regions:
            cv2.circle(mask, (int(cx), int(cy)), int(rad), 255, -1)
        if mask.any():
            small = cv2.resize(f, (W // 24, H // 24), interpolation=cv2.INTER_AREA)
            pix = cv2.GaussianBlur(cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST), (31, 31), 0)
            m = cv2.GaussianBlur(mask, (21, 21), 0)[..., None] / 255.0
            f = (f * (1 - m) + pix * m).astype(np.uint8)
        out.write(f)
    out.release()
    print(f"{a.src}: {len(frames)} frames, pose {stats[0]}, face {stats[1]}, motion-head {stats[2]}")


if __name__ == "__main__":
    main()
