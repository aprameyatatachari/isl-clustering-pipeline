"""Video -> array of pose words with timestamps, bracketed by IDLE."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import MIN_SEGMENT_FRAMES, SMOOTH_WINDOW
from .cluster import load
from .features import descriptor
from .pose import extract_video

IDLE = "IDLE"


def _median_smooth(ids: np.ndarray, w: int = SMOOTH_WINDOW) -> np.ndarray:
    """Kill single-frame flicker between clusters."""
    if w < 3 or len(ids) < w:
        return ids
    half = w // 2
    pad = np.pad(ids, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, w)
    return np.median(win, axis=1).astype(int)


def _runs(ids: np.ndarray):
    """Run-length encode into [(cluster_id, start_frame, end_frame_exclusive)]."""
    out, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            out.append((int(ids[start]), start, i))
            start = i
    return out


def _drop_short(runs, min_len: int = MIN_SEGMENT_FRAMES, idle_id: int | None = None):
    """Absorb sub-threshold runs into the longer neighbour, then re-merge.

    An idle run at either end is always kept however short it is: it anchors the
    timestamps and the first/last-word contract, and a one frame idle at the
    start is common because the signer starts moving almost immediately.
    """
    if len(runs) <= 1:
        return runs

    def anchored(i):
        return idle_id is not None and runs[i][0] == idle_id and i in (0, len(runs) - 1)

    kept = [r for i, r in enumerate(runs)
            if r[2] - r[1] >= min_len or anchored(i)]
    if not kept:
        kept = [max(runs, key=lambda r: r[2] - r[1])]
    merged = []
    for cid, s, e in kept:
        if merged and merged[-1][0] == cid:
            merged[-1] = (cid, merged[-1][1], e)
        else:
            merged.append((cid, s, e))
    # Stretch the surviving runs so they still tile the whole clip.
    total_end = runs[-1][2]
    out = []
    for i, (cid, s, e) in enumerate(merged):
        s = merged[i - 1][2] if i else 0
        e = merged[i + 1][1] if i + 1 < len(merged) else total_end
        out.append((cid, s, e))
    return out


def encode_arrays(xyz, mask, fps: float, source: str, keep_all: bool = False) -> dict:
    """Turn already extracted keypoints into a pose word sequence."""
    km, scaler, names, idle_id = load()
    ids = km.predict(scaler.transform(descriptor(xyz, mask)))
    ids = _median_smooth(ids)
    segs = _drop_short(_runs(ids), idle_id=idle_id)

    # Trim to the outermost idle segments; the sign happens between them. With
    # only one idle present, trim on that side alone -- trimming to a single
    # segment would throw the whole sign away.
    if not keep_all:
        idle_at = [i for i, s in enumerate(segs) if s[0] == idle_id]
        if len(idle_at) > 1:
            segs = segs[idle_at[0]: idle_at[-1] + 1]
        elif len(idle_at) == 1:
            i = idle_at[0]
            segs = segs[i:] if i < len(segs) / 2 else segs[: i + 1]

    # Guarantee the contract: first and last word are always IDLE.
    if not segs or segs[0][0] != idle_id:
        segs = [(idle_id, 0, segs[0][1] if segs else 1)] + segs
    if segs[-1][0] != idle_id:
        segs = segs + [(idle_id, segs[-1][2], segs[-1][2] + 1)]

    t0 = segs[0][1] / fps  # timestamps are relative to the first idle pose
    words = [{
        "word": names[cid],
        "cluster_id": cid,
        "start": round(s / fps - t0, 3),
        "end": round(e / fps - t0, 3),
        "duration": round((e - s) / fps, 3),
    } for cid, s, e in segs]

    return {
        "video": str(source),
        "fps": round(fps, 3),
        "frames": int(len(ids)),
        "sequence": [w["word"] for w in words],
        "words": words,
    }


def encode_video(path: Path, keep_all: bool = False) -> dict:
    """Run one video through the pipeline and return its pose word sequence."""
    xyz, mask, fps = extract_video(path)
    return encode_arrays(xyz, mask, fps, str(path), keep_all)


def encode_cached(word: str, cache_dir: Path | None = None, limit: int = 3):
    """Encode cached clips for one sign label, no video file needed.

    Pose extraction already ran over the dataset, so a word can be replayed from
    the keypoint cache even after `auto` deleted the source videos.
    """
    from .config import CACHE_DIR
    from .extract import load_cache

    want = word.strip().lower()
    out = []
    for xyz, mask, fps, label, source in load_cache(cache_dir or CACHE_DIR):
        if label != want:
            continue
        out.append(encode_arrays(xyz, mask, fps, source))
        if len(out) >= limit:
            break
    return out


def cached_words(cache_dir: Path | None = None) -> dict:
    """Sign label -> number of cached clips, for listing what can be demoed."""
    from .config import CACHE_DIR
    from .extract import load_cache

    counts: dict = {}
    for _xyz, _mask, _fps, label, _src in load_cache(cache_dir or CACHE_DIR):
        counts[label] = counts.get(label, 0) + 1
    return counts


def encode_many(paths, out_path: Path | None = None):
    results = [encode_video(p) for p in paths]
    if out_path:
        Path(out_path).write_text(json.dumps(results, indent=2))
    return results
