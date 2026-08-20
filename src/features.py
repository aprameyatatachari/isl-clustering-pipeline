"""Normalization + feature vectors + joint rotations for avatar retargeting."""
from __future__ import annotations

import numpy as np

from .config import N_HAND, UPPER_BODY_IDX

# Indices inside the stored keypoint array
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 0, 1, 2, 3, 4, 5
L_HIP, R_HIP, NOSE = 12, 13, 14
B = len(UPPER_BODY_IDX)
LH = slice(B, B + N_HAND)
RH = slice(B + N_HAND, B + 2 * N_HAND)
EPS = 1e-6


def normalize(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Center on mid-shoulder, scale by shoulder width, fill missing hands.

    Works on a single frame [N,3] or a batch [T,N,3]; returns the same shape.
    Making the pose translation- and scale-invariant is what lets frames from
    different signers and camera distances land in the same cluster.
    """
    single = xyz.ndim == 2
    if single:
        xyz, mask = xyz[None], mask[None]
    out = xyz.astype(np.float32).copy()

    # A hand that was not detected collapses onto its wrist, so the descriptor
    # degrades smoothly instead of jumping to the origin.
    for hs, wri in ((LH, L_WRI), (RH, R_WRI)):
        missing = ~mask[:, hs].any(1)
        out[missing, hs] = out[missing, wri][:, None, :]

    center = 0.5 * (out[:, L_SHO] + out[:, R_SHO])
    scale = np.linalg.norm(out[:, L_SHO] - out[:, R_SHO], axis=-1)
    scale = np.maximum(scale, EPS)[:, None, None]
    out = (out - center[:, None, :]) / scale
    return out[0] if single else out


def _angle(a, b, c):
    """Interior angle at b, in degrees, for point triples (batched)."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=-1) + EPS
    n2 = np.linalg.norm(v2, axis=-1) + EPS
    cos = np.clip((v1 * v2).sum(-1) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def angle_features(n: np.ndarray) -> np.ndarray:
    """Rotation-ish scalars that clustering benefits from (batched [T,N,3])."""
    feats = [
        _angle(n[:, L_SHO], n[:, L_ELB], n[:, L_WRI]),   # left elbow flexion
        _angle(n[:, R_SHO], n[:, R_ELB], n[:, R_WRI]),   # right elbow flexion
        _angle(n[:, R_SHO], n[:, L_SHO], n[:, L_ELB]),   # left shoulder abduct
        _angle(n[:, L_SHO], n[:, R_SHO], n[:, R_ELB]),   # right shoulder abduct
    ]
    # Wrist height relative to the nose separates "hands up" from "hands down".
    feats.append((n[:, NOSE, 1] - n[:, L_WRI, 1]) * 45.0)
    feats.append((n[:, NOSE, 1] - n[:, R_WRI, 1]) * 45.0)
    # Distance between the two hands separates one-handed vs two-handed signs.
    feats.append(np.linalg.norm(n[:, L_WRI] - n[:, R_WRI], axis=-1) * 45.0)
    return np.stack(feats, -1).astype(np.float32)


def descriptor(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Full feature matrix [T, D] used for KMeans."""
    n = normalize(xyz, mask)
    flat = n.reshape(len(n), -1)
    return np.concatenate([flat, angle_features(n) / 45.0], 1).astype(np.float32)


def joint_rotations(n_frame: np.ndarray) -> dict:
    """Target rotations for an avatar rig, from one normalized frame.

    Angles are in degrees in a body-local frame: +x is the signer's left, +y is
    up, +z is toward the camera. MediaPipe reports depth the other way round
    (more negative means closer to the camera) and its image y grows downward,
    so both axes are flipped here. Bone directions are also emitted as unit
    vectors so a rig can consume whichever form it prefers.
    """
    f = n_frame

    def bone(a, b):
        v = f[b] - f[a]
        v = np.array([v[0], -v[1], -v[2]], np.float32)
        return v / (np.linalg.norm(v) + EPS)

    out = {}
    for side, sho, elb, wri, hand in (
        ("left", L_SHO, L_ELB, L_WRI, LH),
        ("right", R_SHO, R_ELB, R_WRI, RH),
    ):
        upper, fore = bone(sho, elb), bone(elb, wri)
        out[f"{side}_shoulder"] = {
            "pitch": float(np.degrees(np.arcsin(np.clip(upper[1], -1, 1)))),
            "yaw": float(np.degrees(np.arctan2(upper[2], upper[0]))),
            "dir": [round(float(v), 4) for v in upper],
        }
        out[f"{side}_elbow"] = {
            "flexion": float(180.0 - _angle(f[sho][None], f[elb][None], f[wri][None])[0]),
            "dir": [round(float(v), 4) for v in fore],
        }
        palm = f[hand][9] - f[hand][0]
        palm = palm / (np.linalg.norm(palm) + EPS)
        out[f"{side}_wrist"] = {
            "dir": [round(float(v), 4) for v in (palm[0], -palm[1], -palm[2])]
        }
    return out
