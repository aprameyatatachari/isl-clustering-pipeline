"""Human-readable names for clusters, derived from the average pose."""
from __future__ import annotations

import numpy as np

from .features import (EPS, L_ELB, L_SHO, L_WRI, NOSE, R_ELB, R_SHO, R_WRI)

# Height zones, calibrated against real INCLUDE poses. Units are shoulder
# widths above the shoulder line (y grows downward in the normalized frame), so
# hanging arms measure about -2.7, hands at the chest about -0.9, and hands at
# the chin about -0.2.
HANGING = -1.8          # below this the arm is simply down at the side
_ZONES = [(-0.60, "WAIST"), (-0.10, "CHEST"), (0.50, "FACE"), (9e9, "ABOVE_HEAD")]
TOGETHER_GAP = 1.5      # measured hand separation when both hands work together


def _zone(y_up: float) -> str:
    for thr, name in _ZONES:
        if y_up < thr:
            return name
    return "ABOVE_HEAD"


def _raised(n, wri) -> bool:
    """Wrist is meaningfully lifted, not just hanging at the signer's side."""
    return (-n[wri, 1]) > HANGING


def describe(n_frame: np.ndarray) -> dict:
    """Structured description of one normalized average pose."""
    n = n_frame
    lh_up, rh_up = -n[L_WRI, 1], -n[R_WRI, 1]
    both = _raised(n, L_WRI) and _raised(n, R_WRI)
    left, right = _raised(n, L_WRI), _raised(n, R_WRI)
    hands_gap = float(np.linalg.norm(n[L_WRI] - n[R_WRI]))

    if both:
        side, y = "BOTH", max(lh_up, rh_up)
    elif left:
        side, y = "LEFT", lh_up
    elif right:
        side, y = "RIGHT", rh_up
    else:
        side, y = "REST", max(lh_up, rh_up)

    def flex(sho, elb, wri):
        v1, v2 = n[sho] - n[elb], n[wri] - n[elb]
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + EPS))
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    return {
        "side": side,
        "zone": _zone(float(y)),
        "hands_gap": round(hands_gap, 3),
        "contact": "TOGETHER" if hands_gap < TOGETHER_GAP else "APART",
        "left_elbow_deg": round(flex(L_SHO, L_ELB, L_WRI), 1),
        "right_elbow_deg": round(flex(R_SHO, R_ELB, R_WRI), 1),
        "left_wrist_height": round(float(lh_up), 3),
        "right_wrist_height": round(float(rh_up), 3),
    }


def is_idle(desc: dict) -> bool:
    """Both arms hanging down: the neutral pose every clip starts and ends in."""
    return (desc["side"] == "REST"
            and desc["left_wrist_height"] < HANGING
            and desc["right_wrist_height"] < HANGING)


def name_clusters(centroid_frames: np.ndarray, idle_id: int) -> dict:
    """Map cluster id -> unique uppercase word. ``centroid_frames`` is [K,N,3]."""
    names, used = {}, {}
    for k, frame in enumerate(centroid_frames):
        if k == idle_id:
            names[k] = "IDLE"
            continue
        d = describe(frame)
        base = f"{d['side']}_{d['zone']}" if d["side"] != "REST" else "REST_LOW"
        if d["side"] == "BOTH":
            base += f"_{d['contact']}"
        used[base] = used.get(base, 0) + 1
        names[k] = base if used[base] == 1 else f"{base}_{used[base]}"
    return names
