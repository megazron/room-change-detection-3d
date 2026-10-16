"""Render a synthetic before/after room pair with known changes (MuJoCo, headless).

Writes <out>/before.mp4, <out>/after.mp4 and <out>/gt.json (camera poses per video
frame + ground-truth changes). Two scenes (--scene a|b), each with one moved, one added and one removed object;
the two videos of a scene use different camera paths.

    python tools/make_synthetic.py --scene a --out data/synthetic
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _pick_gl():
    """Use MUJOCO_GL if set, else the first headless backend that can render on this machine."""
    if "MUJOCO_GL" in os.environ:
        return
    probe = "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); mujoco.Renderer(m, 8, 8).render()"
    for gl in ("egl", "osmesa", "glfw"):
        env = dict(os.environ, MUJOCO_GL=gl)
        if subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True).returncode == 0:
            os.environ["MUJOCO_GL"] = gl
            return
    sys.exit("error: no working MuJoCo GL backend (tried egl, osmesa, glfw). "
             "On Ubuntu: sudo apt install libosmesa6 (then MUJOCO_GL=osmesa) or libegl1.")


_pick_gl()
import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

W, H = 960, 540
FOVY = 60.0


def noise_texture(path, seed, base, n_shapes=900, size=1024, amp=90):
    """Random high-frequency texture (blobs, rectangles, lines) so SIFT finds features."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = base
    for _ in range(n_shapes):
        col = tuple(int(c) for c in np.clip(np.array(base) + rng.integers(-amp, amp, 3), 0, 255))
        kind = rng.integers(3)
        x, y = rng.integers(0, size, 2)
        if kind == 0:
            cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 25)), col, -1)
        elif kind == 1:
            w, h = rng.integers(4, 50, 2)
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), col, -1)
        else:
            x2, y2 = rng.integers(0, size, 2)
            cv2.line(img, (int(x), int(y)), (int(x2), int(y2)), col, int(rng.integers(1, 4)))
    cv2.imwrite(str(path), img)


def box_texture(path, seed):
    """Cardboard look: brown with tape strip and printed marks."""
    rng = np.random.default_rng(seed)
    img = np.full((512, 512, 3), (70, 130, 185), np.uint8)  # BGR brown
    img = np.clip(img.astype(int) + rng.integers(-12, 12, img.shape), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (230, 0), (282, 511), (150, 190, 215), -1)
    for _ in range(30):
        x, y = rng.integers(20, 490, 2)
        cv2.putText(img, "FRAGILE"[: rng.integers(3, 8)], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (30, 50, 80), 2)
    cv2.imwrite(str(path), img)


def _chair(cx, cy, yaw):
    # seat, back, 4 legs; body frame on the floor
    return f'''
    <body name="chair" pos="{cx} {cy} 0" euler="0 0 {yaw}">
      <geom type="box" size="0.24 0.24 0.03" pos="0 0 0.45" material="wood"/>
      <geom type="box" size="0.03 0.24 0.25" pos="-0.21 0 0.73" material="wood"/>
      <geom type="cylinder" size="0.025 0.22" pos="0.2 0.2 0.22" material="metal"/>
      <geom type="cylinder" size="0.025 0.22" pos="0.2 -0.2 0.22" material="metal"/>
      <geom type="cylinder" size="0.025 0.22" pos="-0.2 0.2 0.22" material="metal"/>
      <geom type="cylinder" size="0.025 0.22" pos="-0.2 -0.2 0.22" material="metal"/>
    </body>'''


def scene_xml(texdir, state):
    t = texdir
    obj = []
    if state.get("chair"):
        obj.append(_chair(*state["chair"]))
    if state.get("bin"):
        bx, by = state["bin"]
        obj.append(f'<geom name="bin" type="cylinder" size="0.18 0.22" pos="{bx} {by} 0.22" material="bin"/>')
    if state.get("ball"):
        bx, by = state["ball"]
        obj.append(f'<geom name="ball" type="sphere" size="0.2" pos="{bx} {by} 0.2" material="ball"/>')
    if state.get("box"):
        bx, by = state["box"]
        obj.append(f'<geom name="cbox" type="box" size="0.25 0.2 0.2" pos="{bx} {by} 0.2" '
                   f'euler="0 0 0.3" material="cardboard"/>')
    return f'''
<mujoco>
  <visual><global fovy="{FOVY}" offwidth="{W}" offheight="{H}"/><quality shadowsize="2048"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.35 0.35 0.35"/></visual>
  <asset>
    <texture name="floor" type="2d" file="{t}/floor.png"/>
    <texture name="wallA" type="cube" file="{t}/wallA.png"/>
    <texture name="wallB" type="cube" file="{t}/wallB.png"/>
    <texture name="wallC" type="cube" file="{t}/wallC.png"/>
    <texture name="wallD" type="cube" file="{t}/wallD.png"/>
    <texture name="wood" type="cube" file="{t}/wood.png"/>
    <texture name="cardboard" type="cube" file="{t}/cardboard.png"/>
    <texture name="ball" type="cube" builtin="checker" rgb1="0.95 0.45 0.05" rgb2="0.15 0.1 0.05" width="64" height="64"/>
    <texture name="shelf" type="cube" file="{t}/shelf.png"/>
    <material name="floor" texture="floor" texrepeat="1 1"/>
    <material name="wallA" texture="wallA"/><material name="wallB" texture="wallB"/>
    <material name="wallC" texture="wallC"/><material name="wallD" texture="wallD"/>
    <material name="wood" texture="wood"/><material name="cardboard" texture="cardboard"/>
    <material name="ball" texture="ball"/><material name="shelf" texture="shelf"/>
    <material name="metal" rgba="0.3 0.3 0.32 1"/>
    <texture name="bin" type="cube" builtin="gradient" rgb1="0.1 0.35 0.8" rgb2="0.05 0.15 0.4" width="64" height="64"/>
    <material name="bin" texture="bin"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <geom type="plane" size="3 2.5 0.1" material="floor"/>
    <geom type="box" size="3 0.05 1.4" pos="0 2.55 1.4" material="wallA"/>
    <geom type="box" size="3 0.05 1.4" pos="0 -2.55 1.4" material="wallB"/>
    <geom type="box" size="0.05 2.5 1.4" pos="3.05 0 1.4" material="wallC"/>
    <geom type="box" size="0.05 2.5 1.4" pos="-3.05 0 1.4" material="wallD"/>
    <!-- static furniture -->
    <geom type="box" size="0.6 0.4 0.03" pos="0.9 0.6 0.74" material="wood"/>
    <geom type="cylinder" size="0.03 0.36" pos="1.4 0.9 0.36" material="metal"/>
    <geom type="cylinder" size="0.03 0.36" pos="0.4 0.9 0.36" material="metal"/>
    <geom type="cylinder" size="0.03 0.36" pos="1.4 0.3 0.36" material="metal"/>
    <geom type="cylinder" size="0.03 0.36" pos="0.4 0.3 0.36" material="metal"/>
    <geom type="box" size="0.2 0.7 0.9" pos="-2.75 0.8 0.9" material="shelf"/>
    {"".join(obj)}
  </worldbody>
</mujoco>'''


# Two scenes with different changes and camera paths (azimuth range in degrees, height).
SCENES = {
    "a": {
        "before": {"chair": (-0.9, 0.7, 0.0), "ball": (1.3, -1.0)},
        "after": {"chair": (-0.5, -1.1, 0.6), "box": (0.1, -0.2)},
        "paths": ((-160, 10, 1.45), (-50, 120, 1.6)),
        "changes": [
            {"type": "moved", "label": "chair", "from": [-0.9, 0.7, 0.45], "to": [-0.5, -1.1, 0.45]},
            {"type": "removed", "label": "ball", "centroid": [1.3, -1.0, 0.2]},
            {"type": "added", "label": "box", "centroid": [0.1, -0.2, 0.2]},
        ],
    },
    "b": {
        "before": {"chair": (-0.9, 0.7, 0.0), "box": (1.2, -1.1), "ball": (-1.4, -1.0)},
        "after": {"box": (-0.1, 1.3), "ball": (-1.4, -1.0), "bin": (1.1, -0.3)},
        "paths": ((-140, 30, 1.5), (-10, 160, 1.35)),
        "changes": [
            {"type": "moved", "label": "box", "from": [1.2, -1.1, 0.2], "to": [-0.1, 1.3, 0.2]},
            {"type": "removed", "label": "chair", "centroid": [-0.9, 0.7, 0.45]},
            {"type": "added", "label": "bin", "centroid": [1.1, -0.3, 0.22]},
        ],
    },
}


def camera_path(n, a0, a1, height, seed):
    """Camera walks an ellipse near the walls, looking at the room centre with jitter."""
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n):
        s = i / (n - 1)
        a = np.deg2rad(a0 + (a1 - a0) * s)
        pos = np.array([2.3 * np.cos(a), 1.8 * np.sin(a), height + 0.08 * np.sin(6 * s)])
        look = np.array([0.6 * np.sin(3 * s), 0.4 * np.cos(2 * s), 0.35]) + rng.normal(0, 0.005, 3)
        poses.append((pos, look))
    return poses


def cam_matrices(pos, look):
    f = look - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])  # world -> camera (OpenCV)
    return R, -R @ pos


def render_video(state, poses, texdir, mp4, fps):
    m = mujoco.MjModel.from_xml_string(scene_xml(texdir, state))
    dat = mujoco.MjData(m)
    mujoco.mj_forward(m, dat)
    ren = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    proc = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(mp4)],
        stdin=subprocess.PIPE)
    out = []
    for pos, look in poses:
        v = look - pos
        dist = np.linalg.norm(v)
        cam.lookat[:] = look
        cam.distance = dist
        cam.azimuth = np.rad2deg(np.arctan2(v[1], v[0]))
        cam.elevation = np.rad2deg(np.arcsin(v[2] / dist))
        ren.update_scene(dat, cam)
        proc.stdin.write(ren.render().tobytes())
        R, t = cam_matrices(pos, look)
        out.append({"R": R.tolist(), "t": t.tolist()})
    proc.stdin.close()
    proc.wait()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scene", choices=sorted(SCENES), default="a")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        noise_texture(f"{td}/floor.png", 1, (120, 140, 150), 2500, 2048, amp=45)
        noise_texture(f"{td}/wallA.png", 2, (200, 200, 190))
        noise_texture(f"{td}/wallB.png", 3, (170, 190, 210))
        noise_texture(f"{td}/wallC.png", 4, (190, 170, 200))
        noise_texture(f"{td}/wallD.png", 5, (180, 210, 180))
        noise_texture(f"{td}/wood.png", 6, (60, 100, 150), 300, 512, amp=25)
        noise_texture(f"{td}/shelf.png", 7, (90, 90, 110), 300, 512, amp=40)
        box_texture(f"{td}/cardboard.png", 8)
        n = int(a.seconds * a.fps)
        fy = H / 2 / np.tan(np.deg2rad(FOVY) / 2)
        K = [[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]]
        sc = SCENES[a.scene]
        gt = {"K": K, "width": W, "height": H, "scene": a.scene, "changes": sc["changes"]}
        (p0, p1) = sc["paths"]
        gt["before"] = render_video(sc["before"], camera_path(n, *p0, 0), td, out / "before.mp4", a.fps)
        gt["after"] = render_video(sc["after"], camera_path(n, *p1, 1), td, out / "after.mp4", a.fps)
    (out / "gt.json").write_text(json.dumps(gt))
    print("wrote", out)


if __name__ == "__main__":
    main()
