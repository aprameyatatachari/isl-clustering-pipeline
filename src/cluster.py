"""KMeans over all dataset frames -> named preset poses + target rotations."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import (ARTIFACT_DIR, CACHE_DIR, CLUSTER_NAMES_PATH, KMEANS_K_RANGE,
                     KMEANS_PATH, KMEANS_SEED, MAX_FRAMES_FOR_FIT, POSE_MAP_PATH,
                     SCALER_PATH)
from .extract import load_cache
from .features import descriptor, joint_rotations, normalize
from .naming import describe, is_idle, name_clusters

EDGE_FRAC = 0.12  # fraction of a clip, at each end, treated as "start/end"


def build_matrix(cache_dir: Path = CACHE_DIR):
    """Stack every cached frame into X[D], plus per-frame bookkeeping."""
    feats, norms, labels, edge = [], [], [], []
    for xyz, mask, _fps, label, _src in load_cache(cache_dir):
        feats.append(descriptor(xyz, mask))
        norms.append(normalize(xyz, mask))
        t = len(xyz)
        e = max(1, int(t * EDGE_FRAC))
        flag = np.zeros(t, bool)
        flag[:e] = True
        flag[-e:] = True
        edge.append(flag)
        labels += [label] * t
    if not feats:
        raise RuntimeError("no cached keypoints; run `extract` first")
    return (np.concatenate(feats), np.concatenate(norms),
            np.array(labels), np.concatenate(edge))


def pick_k(Xs: np.ndarray, k_range=KMEANS_K_RANGE, sample: int = 8000):
    """Silhouette sweep on a subsample; step through the range coarsely."""
    rng = np.random.default_rng(KMEANS_SEED)
    idx = rng.choice(len(Xs), min(sample, len(Xs)), replace=False)
    sub = Xs[idx]
    lo, hi = k_range
    best, best_k = -2.0, lo
    for k in range(lo, hi + 1, max(1, (hi - lo) // 8)):
        km = MiniBatchKMeans(k, random_state=KMEANS_SEED, n_init=3, batch_size=2048)
        s = silhouette_score(sub, km.fit_predict(sub), sample_size=min(4000, len(sub)),
                             random_state=KMEANS_SEED)
        if s > best:
            best, best_k = s, k
    return best_k, best


def fit(cache_dir: Path = CACHE_DIR, k: int | None = None):
    X, N, labels, edge = build_matrix(cache_dir)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    if len(Xs) > MAX_FRAMES_FOR_FIT:
        rng = np.random.default_rng(KMEANS_SEED)
        fit_idx = rng.choice(len(Xs), MAX_FRAMES_FOR_FIT, replace=False)
    else:
        fit_idx = np.arange(len(Xs))

    score = None
    if k is None:
        k, score = pick_k(Xs[fit_idx])

    km = MiniBatchKMeans(k, random_state=KMEANS_SEED, n_init=10, batch_size=4096)
    km.fit(Xs[fit_idx])
    assign = km.predict(Xs)

    # Average pose per cluster, in normalized space, so it can drive an avatar.
    avg = np.stack([N[assign == c].mean(0) if (assign == c).any() else np.zeros(N.shape[1:])
                    for c in range(k)])

    edge_rate = np.array([edge[assign == c].mean() if (assign == c).any() else 0.0
                          for c in range(k)])
    descs = [describe(a) for a in avg]
    idle_candidates = [c for c in range(k) if is_idle(descs[c])]
    pool = idle_candidates or list(range(k))
    idle_id = int(max(pool, key=lambda c: edge_rate[c]))

    names = name_clusters(avg, idle_id)

    pose_map = {}
    for c in range(k):
        sel = assign == c
        top = {}
        for lab in labels[sel][:20000]:
            top[lab] = top.get(lab, 0) + 1
        pose_map[names[c]] = {
            "cluster_id": c,
            "frames": int(sel.sum()),
            "share": round(float(sel.mean()), 5),
            "edge_rate": round(float(edge_rate[c]), 4),
            "is_idle": c == idle_id,
            "description": descs[c],
            "target_rotations": joint_rotations(avg[c]),
            "average_pose": np.round(avg[c], 4).tolist(),
            "top_signs": sorted(top.items(), key=lambda kv: -kv[1])[:8],
        }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, KMEANS_PATH)
    joblib.dump(scaler, SCALER_PATH)
    POSE_MAP_PATH.write_text(json.dumps(pose_map, indent=2))
    CLUSTER_NAMES_PATH.write_text(json.dumps(
        {"idle_id": idle_id, "names": {str(c): names[c] for c in range(k)}}, indent=2))
    return {"k": k, "silhouette": score, "frames": int(len(Xs)), "idle": names[idle_id]}


def load():
    """Load the fitted model + naming table for inference."""
    km = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = json.loads(CLUSTER_NAMES_PATH.read_text())
    names = {int(c): n for c, n in meta["names"].items()}
    return km, scaler, names, int(meta["idle_id"])
