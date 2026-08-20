"""Batch pose extraction: dataset videos -> cached keypoint .npz files."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import pose
from .config import CACHE_DIR, DATA_DIR
from .dataset import find_videos


def cache_path(video: Path, root: Path, cache_dir: Path = CACHE_DIR) -> Path:
    rel = str(Path(video).resolve().relative_to(Path(root).resolve()))
    key = hashlib.sha1(rel.encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{Path(video).stem}_{key}.npz"


def extract_one(video: Path, out: Path, label: str = "unknown") -> Path:
    xyz, mask, fps = pose.extract_video(video)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first: a crash mid-write must not leave a half
    # written .npz that the clustering step would later fail to read.
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, xyz=xyz, mask=mask, fps=np.float32(fps),
        label=np.array(label), source=np.array(str(video)),
    )
    tmp.replace(out)
    return out


def extract_dataset(root: Path = DATA_DIR, cache_dir: Path = CACHE_DIR,
                    limit: int | None = None, overwrite: bool = False,
                    include50_only: bool = False,
                    categories: "list[str] | None" = None):
    """Run pose estimation over the whole dataset, skipping cached videos."""
    videos = find_videos(root, include50_only=include50_only, categories=categories)
    if limit:
        videos = videos[:limit]
    done, failed = 0, []
    for video, label in tqdm(videos, desc="pose"):
        out = cache_path(video, root, cache_dir)
        if out.exists() and not overwrite:
            done += 1
            continue
        try:
            extract_one(video, out, label)
            done += 1
        except Exception as exc:  # a few INCLUDE files are truncated
            failed.append((str(video), str(exc)))
    pose.close()
    return done, failed


def load_cache(cache_dir: Path = CACHE_DIR):
    """Yield (keypoints, mask, fps, label, source) for every cached video."""
    for f in sorted(Path(cache_dir).glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        try:
            d = np.load(f, allow_pickle=False)
            out = d["xyz"], d["mask"], float(d["fps"]), str(d["label"]), str(d["source"])
        except Exception:
            continue  # skip a cache file left corrupt by an interrupted run
        yield out
