"""Joint structure-from-motion over the frames of both recordings.

Both videos go into one COLMAP model. The unchanged parts of the room give
matches across the two recordings, so they land in a common frame for free;
moved objects only contribute outlier matches that RANSAC rejects.
"""
import shutil
from pathlib import Path

import numpy as np
import pycolmap

from .frames import InputError


def run_sfm(image_dir, work_dir, max_features=4096, log=print):
    pycolmap.set_random_seed(0)
    image_dir, work_dir = Path(image_dir), Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    db = work_dir / "database.db"

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = max_features
    log("  sfm: extracting SIFT features")
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.PER_FOLDER,
                              reader_options=reader, extraction_options=ext)
    log("  sfm: exhaustive matching")
    pycolmap.match_exhaustive(db)
    log("  sfm: incremental mapping")
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = 6
    opts.random_seed = 0
    recs = pycolmap.incremental_mapping(db, image_dir, work_dir / "sparse", options=opts)
    if not recs:
        raise InputError("SfM failed to build any model. Record slower, with more texture and overlap.")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    return rec


def export_cameras(rec):
    """Per registered image: K, R (world->cam), t, and its observed 3D points."""
    cams = {}
    pts = {pid: p.xyz for pid, p in rec.points3D.items()}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        cam = img.camera
        pose = img.cam_from_world()
        R = np.asarray(pose.rotation.matrix())
        t = np.asarray(pose.translation)
        obs = [p.point3D_id for p in img.points2D if p.has_point3D()]
        cams[img.name] = {
            "K": np.asarray(cam.calibration_matrix()),
            "R": R, "t": t,
            "width": cam.width, "height": cam.height,
            "points": np.array([pts[i] for i in obs]) if obs else np.zeros((0, 3)),
        }
    return cams
