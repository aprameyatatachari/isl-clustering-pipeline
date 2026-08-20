"""One-shot pipeline: walk every Zenodo archive, pose it, then cluster.

Archives are handled strictly one at a time -- download, unzip, pose-estimate,
delete the videos -- so peak disk usage stays around one archive (~2 GB) instead
of the full 57 GB. Progress is journalled, so re-running resumes where it
stopped after a crash, a full disk, or a dropped connection.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import pose
from .config import ARTIFACT_DIR, CACHE_DIR, DATA_DIR, DOWNLOAD_WORKERS
from .dataset import fetch_archive, label_from_path, load_metadata, select_archives
from .extract import cache_path, extract_one

STATE_PATH = ARTIFACT_DIR / "auto_state.json"

# If this many videos fail in a row without a single success, the problem is the
# environment (a broken mediapipe install, a missing model), not the videos --
# so stop instead of burning through 57 GB producing nothing.
ABORT_AFTER_CONSECUTIVE_FAILURES = 8


class PoseBackendError(RuntimeError):
    pass


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"done": [], "videos": 0, "failed": []}


def _save_state(state: dict):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _include50_names() -> set:
    meta = load_metadata()
    return {Path(v).name for v in meta.loc[meta["include_50"] == True, "video_path"]}


def _prune(paths, data_dir: Path):
    """Delete extracted videos and any directories they emptied out."""
    dirs = set()
    for p in paths:
        Path(p).unlink(missing_ok=True)
        dirs.add(Path(p).parent)
    for d in sorted(dirs, key=lambda x: -len(x.parts)):
        try:
            while d != data_dir and d.exists() and not any(d.iterdir()):
                d.rmdir()
                d = d.parent
        except OSError:
            pass


def run(data_dir: Path = DATA_DIR, cache_dir: Path = CACHE_DIR,
        categories: "list[str] | None" = None, max_gb: float | None = None,
        include50_only: bool = False, keep_videos: bool = False,
        limit_per_archive: int | None = None, do_cluster: bool = True,
        k: int | None = None, restart: bool = False,
        workers: int = DOWNLOAD_WORKERS):
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    archives = select_archives(categories, max_gb)

    state = {"done": [], "videos": 0, "failed": []} if restart else _load_state()
    streak = 0  # consecutive failures with no success in between
    keep_names = _include50_names() if include50_only else None
    pending = [a for a in archives if a["name"] not in state["done"]]

    print(f"{len(pending)}/{len(archives)} archive(s) left "
          f"({sum(a['size_mb'] for a in pending)/1000:.1f} GB to fetch)")

    for i, a in enumerate(pending, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(pending)}] {a['name']} ({a['size_mb']} MB)")
        try:
            paths = fetch_archive(a, data_dir, workers=workers)
        except Exception as exc:
            print(f"  download failed: {exc}")
            state["failed"].append([a["name"], f"download: {exc}"])
            _save_state(state)
            continue

        todo = paths if keep_names is None else [p for p in paths if p.name in keep_names]
        if limit_per_archive:
            todo = todo[:limit_per_archive]

        done = 0
        for v in todo:
            out = cache_path(v, data_dir, cache_dir)
            if out.exists():
                done += 1
                continue
            try:
                extract_one(v, out, label_from_path(v, data_dir))
                done += 1
                streak = 0
            except Exception as exc:  # a handful of INCLUDE files are truncated
                state["failed"].append([str(v), str(exc)])
                streak += 1
                if state["videos"] + done == 0 and streak >= ABORT_AFTER_CONSECUTIVE_FAILURES:
                    _save_state(state)
                    raise PoseBackendError(
                        f"{streak} videos failed in a row with no successes; "
                        "the pose backend is broken, not the data. "
                        f"last error: {exc}"
                    ) from exc
            if done and done % 25 == 0:
                print(f"  posed {done}/{len(todo)}")

        if not keep_videos:
            _prune(paths, data_dir)

        state["done"].append(a["name"])
        state["videos"] += done
        _save_state(state)
        free = shutil.disk_usage(data_dir.parent).free / 1e9
        print(f"  posed {done}/{len(todo)} videos in {time.time()-t0:.0f}s "
              f"| total {state['videos']} | disk free {free:.1f} GB")

    pose.close()
    print(f"\nall archives done: {state['videos']} videos posed, "
          f"{len(state['failed'])} failures")

    if do_cluster:
        from .cluster import fit
        print("clustering...")
        print(json.dumps(fit(cache_dir, k), indent=2))
    return state
