"""INCLUDE dataset: Hugging Face metadata + Zenodo video archives.

The HF repo ai4bharat/INCLUDE ships metadata parquet files only (HF does not
host the videos). The actual .MOV files live in Zenodo record 4010759 as ~46
category zips, 57 GB in total, so downloads are per-category and resumable.
"""
from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
from pathlib import Path

from tqdm import tqdm

from .config import (ARTIFACT_DIR, DATA_DIR, DOWNLOAD_CHUNK_MB, DOWNLOAD_WORKERS,
                     VIDEO_EXTS)

HF_REPO = "ai4bharat/INCLUDE"
ZENODO_API = "https://zenodo.org/api/records/4010759"
METADATA_PATH = ARTIFACT_DIR / "include_metadata.csv"

_NUM_PREFIX = re.compile(r"^\d+[\.\-_ ]+")


# --------------------------------------------------------------- metadata ---
def fetch_metadata(repo_id: str = HF_REPO) -> "list[dict]":
    """Download the HF parquet splits and flatten them into one row list."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    rows = []
    for split in ("train", "val", "test"):
        p = hf_hub_download(repo_id, f"data/{split}-00000-of-00001.parquet",
                            repo_type="dataset")
        df = pd.read_parquet(p)
        df["split"] = split
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_PATH, index=False)
    return df


def load_metadata():
    import pandas as pd

    if not METADATA_PATH.exists():
        return fetch_metadata()
    return pd.read_csv(METADATA_PATH)


# --------------------------------------------------------------- archives ---
def list_archives() -> "list[dict]":
    """Zenodo archive list: [{name, url, size_mb, category}]."""
    with urllib.request.urlopen(ZENODO_API) as r:
        rec = json.load(r)
    out = []
    for f in rec["files"]:
        name = f["key"]
        if not name.lower().endswith(".zip"):
            continue  # the record also carries the upstream download helper script
        out.append({
            "name": name,
            "url": f["links"]["self"],
            "size_mb": round(f["size"] / 1e6),
            "category": re.sub(r"_\d+of\d+\.zip$", "", name).replace(".zip", ""),
        })
    return sorted(out, key=lambda d: d["name"])


def _remote_size(url: str):
    """(size_bytes, supports_ranges) for a remote file.

    Zenodo serves ranges but does not advertise Accept-Ranges on HEAD, so the
    capability is probed with a one byte range request rather than trusted from
    the header -- otherwise every download silently falls back to one slow
    stream.
    """
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        size = int(r.headers.get("Content-Length", 0))

    probe = urllib.request.Request(url)
    probe.add_header("Range", "bytes=0-0")
    try:
        with urllib.request.urlopen(probe) as r:
            ranged = r.status == 206
    except urllib.error.HTTPError:
        ranged = False
    return size, ranged


def _download_stream(url: str, dest: Path, resume_from: int = 0):
    """Plain single-connection download, resuming from ``resume_from``."""
    req = urllib.request.Request(url)
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            return dest
        raise
    total = int(resp.headers.get("Content-Length", 0)) + resume_from
    mode = "ab" if resp.status == 206 else "wb"
    start = resume_from if resp.status == 206 else 0
    with open(dest, mode) as fh, tqdm(total=total, initial=start, unit="B",
                                      unit_scale=True, desc=dest.name) as bar:
        while chunk := resp.read(1 << 18):
            fh.write(chunk)
            bar.update(len(chunk))
    return dest


def _download_file(url: str, dest: Path, workers: int = DOWNLOAD_WORKERS,
                   chunk_mb: int = DOWNLOAD_CHUNK_MB):
    """Download ``url`` using several parallel range requests.

    Zenodo throttles each connection to well under 1 MB/s but does not cap the
    total, so fetching disjoint byte ranges concurrently is several times
    faster. Completed chunks are journalled next to the file, which makes an
    interrupted download resume without refetching what it already has.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    size, ranged = _remote_size(url)

    if not size or not ranged or workers < 2:
        have = dest.stat().st_size if dest.exists() else 0
        return _download_stream(url, dest, have)

    if dest.exists() and dest.stat().st_size == size:
        return dest

    chunk = chunk_mb * 1024 * 1024
    n_chunks = (size + chunk - 1) // chunk
    journal = dest.with_suffix(dest.suffix + ".progress")

    done = set()
    if journal.exists() and dest.exists():
        try:
            state = json.loads(journal.read_text())
            if state.get("size") == size and state.get("chunk") == chunk:
                done = set(state["done"])
        except (ValueError, KeyError):
            done = set()

    # Preallocate so every worker can seek straight to its own offset.
    if not dest.exists() or dest.stat().st_size != size:
        with open(dest, "wb") as fh:
            fh.truncate(size)
        done = set()

    todo = [i for i in range(n_chunks) if i not in done]
    lock = threading.Lock()

    def fetch(i: int):
        s = i * chunk
        e = min(s + chunk, size) - 1
        req = urllib.request.Request(url)
        req.add_header("Range", f"bytes={s}-{e}")
        with urllib.request.urlopen(req) as resp, open(dest, "r+b") as fh:
            fh.seek(s)
            got = 0
            while data := resp.read(1 << 18):
                fh.write(data)
                got += len(data)
                with lock:
                    bar.update(len(data))
        return i, got

    with tqdm(total=size, initial=len(done) * chunk, unit="B", unit_scale=True,
              desc=dest.name) as bar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, i): i for i in todo}
            try:
                for fut in as_completed(futures):
                    i, _ = fut.result()
                    done.add(i)
                    journal.write_text(json.dumps(
                        {"size": size, "chunk": chunk, "done": sorted(done)}))
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise

    journal.unlink(missing_ok=True)
    return dest


def _unzip(zip_path: Path, dest: Path):
    """Extract the videos out of one archive; returns the extracted paths."""
    out_paths = []
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if Path(m).suffix.lower() in VIDEO_EXTS]
        for m in tqdm(members, desc=f"unzip {zip_path.name}", leave=False):
            target = dest / m
            out_paths.append(target)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as out:
                out.write(src.read())
    return out_paths


def select_archives(categories: "list[str] | None" = None, max_gb: float | None = None):
    """Archive list narrowed by category name and/or a rough size budget."""
    archives = list_archives()
    if categories:
        want = {c.lower() for c in categories}
        archives = [a for a in archives if a["category"].lower() in want
                    or a["name"].lower().startswith(tuple(want))]
        if not archives:
            raise SystemExit("no archives matched --categories; run `python cli.py archives`")
    if max_gb:
        picked, acc = [], 0.0
        for a in archives:
            if acc + a["size_mb"] / 1000 > max_gb and picked:
                break
            picked.append(a)
            acc += a["size_mb"] / 1000
        archives = picked
    return archives


def fetch_archive(archive: dict, dest: Path = DATA_DIR, zip_dir: Path | None = None,
                  keep_zip: bool = False, workers: int = DOWNLOAD_WORKERS):
    """Download + unzip a single archive. Returns the extracted video paths."""
    dest = Path(dest)
    zip_dir = Path(zip_dir) if zip_dir else dest / "_zips"
    zp = _download_file(archive["url"], zip_dir / archive["name"], workers=workers)
    paths = _unzip(zp, dest)
    if not keep_zip:
        zp.unlink(missing_ok=True)
    return paths


def download(dest: Path = DATA_DIR, categories: "list[str] | None" = None,
             max_gb: float | None = None, keep_zips: bool = False,
             zip_dir: Path | None = None, workers: int = DOWNLOAD_WORKERS):
    """Download and unzip Zenodo archives into ``dest``.

    ``categories`` filters by name prefix (e.g. ["Animals", "Greetings"]);
    ``max_gb`` stops once that much has been queued, which is the practical way
    to grab a workable subset instead of the full 57 GB.
    """
    dest = Path(dest)
    zip_dir = Path(zip_dir) if zip_dir else dest / "_zips"
    archives = select_archives(categories, max_gb)

    print(f"{len(archives)} archive(s), "
          f"{sum(a['size_mb'] for a in archives)/1000:.1f} GB")

    videos = 0
    for a in archives:
        zp = _download_file(a["url"], zip_dir / a["name"], workers=workers)
        videos += len(_unzip(zp, dest))
        if not keep_zips:
            zp.unlink(missing_ok=True)
    if not keep_zips and zip_dir.exists() and not any(zip_dir.iterdir()):
        zip_dir.rmdir()
    return dest, videos


# ------------------------------------------------------------ local files ---
def label_from_path(path: Path, root: Path) -> str:
    """INCLUDE stores one folder per sign, e.g. Animals/4. Bird/MVI_4156.MOV."""
    rel = Path(path).relative_to(root)
    parent = rel.parent.name if rel.parent.name else "unknown"
    return _NUM_PREFIX.sub("", parent).strip().lower() or "unknown"


def find_videos(root: Path = DATA_DIR, include50_only: bool = False,
                categories: "list[str] | None" = None):
    """Every video under ``root``, sorted, with its inferred sign label."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset dir not found: {root}")
    vids = [p for p in sorted(root.rglob("*"))
            if p.suffix.lower() in VIDEO_EXTS and "_zips" not in p.parts]

    if categories:
        want = {c.lower() for c in categories}
        vids = [p for p in vids
                if any(part.lower() in want for part in p.relative_to(root).parts)]

    if include50_only:
        meta = load_metadata()
        keep = {Path(v).name for v in meta.loc[meta["include_50"] == True, "video_path"]}
        vids = [p for p in vids if p.name in keep]

    return [(p, label_from_path(p, root)) for p in vids]
