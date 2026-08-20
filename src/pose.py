"""MediaPipe Holistic pose extraction: video -> per-frame keypoint array.

Uses the Tasks API (mediapipe >= 1.0). The legacy ``mp.solutions.holistic``
module was removed, so the model bundle is downloaded once and cached instead
of being shipped inside the package.
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# MediaPipe's C++ core writes these straight to file descriptor 2 before glog is
# initialised, so the environment variables above cannot suppress them. They are
# emitted once per landmarker and say nothing actionable. Anything not matching
# this list is passed through untouched, so real errors still surface.
_BENIGN_LOG_PATTERNS = (
    "inference_feedback_manager",
    "landmark_projection_calculator",
    "Created TensorFlow Lite XNNPACK delegate",
    "Logging before InitGoogle()",
    "feedback tensors",
)


@contextlib.contextmanager
def quiet_native_logs():
    """Swallow MediaPipe's benign C++ chatter, re-emitting anything else."""
    saved = os.dup(2)
    tmp = tempfile.TemporaryFile()
    try:
        os.dup2(tmp.fileno(), 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        tmp.seek(0)
        text = tmp.read().decode("utf-8", "replace")
        tmp.close()
        kept = [ln for ln in text.splitlines()
                if ln.strip() and not any(p in ln for p in _BENIGN_LOG_PATTERNS)]
        if kept:
            sys.stderr.write(chr(10).join(kept) + chr(10))

import numpy as np

from .config import ARTIFACT_DIR, MIN_VISIBILITY, N_HAND, TARGET_FPS, UPPER_BODY_IDX

N_POINTS = len(UPPER_BODY_IDX) + 2 * N_HAND  # body + left hand + right hand

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")
MODEL_PATH = ARTIFACT_DIR / "models" / "holistic_landmarker.task"

_LANDMARKER = None
_CLOCK_MS = 0  # VIDEO mode demands timestamps that never go backwards


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the holistic model bundle on first use."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading holistic model -> {path}")
    tmp = path.with_suffix(".tmp")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(path)
    return path


def _landmarker():
    """Lazily build a single landmarker (model load is expensive)."""
    global _LANDMARKER
    if _LANDMARKER is None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (HolisticLandmarker,
                                                   HolisticLandmarkerOptions,
                                                   RunningMode)

        with quiet_native_logs():
            _LANDMARKER = HolisticLandmarker.create_from_options(
                HolisticLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(ensure_model())),
                    running_mode=RunningMode.VIDEO,
                    min_pose_detection_confidence=0.5,
                    min_pose_landmarks_confidence=0.5,
                    min_hand_landmarks_confidence=0.5,
                )
            )
    return _LANDMARKER


def _body_block(landmarks):
    pts = np.zeros((len(UPPER_BODY_IDX), 3), np.float32)
    ok = np.zeros(len(UPPER_BODY_IDX), bool)
    if not landmarks:
        return pts, ok
    for out_i, src_i in enumerate(UPPER_BODY_IDX):
        p = landmarks[src_i]
        pts[out_i] = (p.x, p.y, p.z)
        vis = p.visibility if p.visibility is not None else 1.0
        ok[out_i] = vis >= MIN_VISIBILITY
    return pts, ok


def _hand_block(landmarks):
    pts = np.zeros((N_HAND, 3), np.float32)
    ok = np.zeros(N_HAND, bool)
    if not landmarks:
        return pts, ok
    for i, p in enumerate(landmarks[:N_HAND]):
        pts[i] = (p.x, p.y, p.z)
    ok[:] = True
    return pts, ok


def extract_video(path, target_fps: float = TARGET_FPS):
    """Return (xyz[T,N_POINTS,3], mask[T,N_POINTS], fps) for one video.

    Frames are resampled to ``target_fps`` so clustering is not biased by the
    source frame rate, which varies across the INCLUDE dataset.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / target_fps)))
    eff_fps = src_fps / step

    lm = _landmarker()

    # The landmarker is shared across videos and its VIDEO-mode clock is global,
    # so each new video continues past the previous one instead of restarting.
    global _CLOCK_MS
    base_ms = _CLOCK_MS + 1000

    xyz, mask = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts_ms = base_ms + int(idx / src_fps * 1000)
            _CLOCK_MS = max(_CLOCK_MS, ts_ms)
            with quiet_native_logs():
                res = lm.detect_for_video(image, ts_ms)
            b, bm = _body_block(res.pose_landmarks)
            lh, lhm = _hand_block(res.left_hand_landmarks)
            rh, rhm = _hand_block(res.right_hand_landmarks)
            xyz.append(np.concatenate([b, lh, rh], 0))
            mask.append(np.concatenate([bm, lhm, rhm], 0))
        idx += 1
    cap.release()

    if not xyz:
        raise RuntimeError(f"no frames decoded: {path}")
    return np.stack(xyz), np.stack(mask), float(eff_fps)


def close():
    global _LANDMARKER
    if _LANDMARKER is not None:
        _LANDMARKER.close()
        _LANDMARKER = None
_CLOCK_MS = 0  # VIDEO mode demands timestamps that never go backwards
