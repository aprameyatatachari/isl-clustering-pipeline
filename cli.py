#!/usr/bin/env python
"""ISL pose-clustering pipeline CLI: download -> extract -> cluster -> encode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import (ARTIFACT_DIR, CACHE_DIR, DATA_DIR, DOWNLOAD_WORKERS,
                        POSE_MAP_PATH, VIDEO_EXTS)
from src.viewer import VIEWER_PATH


def _csv(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def cmd_archives(a):
    from src.dataset import list_archives
    arcs = list_archives()
    for x in arcs:
        print(f"{x['name']:<34} {x['size_mb']:>6} MB   category={x['category']}")
    print(f"total {sum(x['size_mb'] for x in arcs)/1000:.1f} GB in {len(arcs)} archives")


def cmd_metadata(a):
    from src.dataset import METADATA_PATH, fetch_metadata
    df = fetch_metadata()
    print(f"{len(df)} rows, {df['label'].nunique()} signs, "
          f"{int(df['include_50'].sum())} in INCLUDE-50 -> {METADATA_PATH}")


def cmd_download(a):
    from src.dataset import download
    dest, n = download(Path(a.out), _csv(a.categories), a.max_gb, a.keep_zips,
                       workers=a.workers)
    print(f"{n} videos -> {dest}")


def cmd_auto(a):
    from src.auto import PoseBackendError, run
    try:
        run(Path(a.data), Path(a.cache), _csv(a.categories), a.max_gb, a.include50,
            a.keep_videos, a.limit_per_archive, not a.no_cluster, a.k, a.restart,
            a.workers)
    except PoseBackendError as exc:
        raise SystemExit("aborted: " + str(exc))


def cmd_extract(a):
    from src.extract import extract_dataset
    done, failed = extract_dataset(Path(a.data), Path(a.cache), a.limit, a.overwrite,
                                   a.include50, _csv(a.categories))
    print(f"extracted {done} videos, {len(failed)} failed")
    for f, e in failed[:10]:
        print("  fail:", f, "|", e)


def cmd_cluster(a):
    from src.cluster import fit
    info = fit(Path(a.cache), a.k)
    print(json.dumps(info, indent=2))
    print("pose map ->", POSE_MAP_PATH)


def cmd_names(a):
    m = json.loads(POSE_MAP_PATH.read_text())
    for name, v in sorted(m.items(), key=lambda kv: -kv[1]["frames"]):
        signs = ", ".join(s for s, _ in v["top_signs"][:4])
        print(f"{name:<28} id={v['cluster_id']:<3} frames={v['frames']:<7} signs: {signs}")


def cmd_signs(a):
    from src.encode import cached_words
    counts = cached_words(Path(a.cache) if a.cache else None)
    if not counts:
        raise SystemExit("no cached keypoints; run `extract` or `auto` first")
    for word, n in sorted(counts.items()):
        print(f"{word:<28} {n} clip(s)")
    print(f"{len(counts)} signs cached")


def cmd_demo(a):
    from src.encode import cached_words, encode_cached
    from src.viewer import build

    clips = encode_cached(a.word, Path(a.cache) if a.cache else None, a.n)
    if not clips:
        near = [w for w in cached_words(Path(a.cache) if a.cache else None)
                if a.word.strip().lower() in w]
        hint = f" Did you mean: {', '.join(sorted(near)[:5])}?" if near else ""
        raise SystemExit(f"no cached clips for '{a.word}'."
                         f" Run `python cli.py signs` to list what is available.{hint}")

    tmp = ARTIFACT_DIR / f"demo_{a.word.strip().lower().replace(' ', '_')}.json"
    tmp.write_text(json.dumps(clips, indent=2))
    vrm = _resolve_vrm(a.vrm)
    out = build(sequences=[tmp], vrm=vrm)
    for c in clips:
        print(Path(c["video"]).name, "->", " ".join(c["sequence"]))
    _open_viewer(out, vrm, a.serve)


def _resolve_vrm(arg):
    """--vrm if given, otherwise whatever .vrm is sitting in data/."""
    if arg:
        return Path(arg)
    from src.vrm import default_model
    found = default_model()
    if found:
        print("using avatar:", found)
    return found


def _open_viewer(out, vrm, serve):
    """Report where the viewer is, and serve it when a .vrm needs an origin."""
    print("viewer ->", out)
    if serve or vrm:
        from src.vrm import serve as serve_dir
        if vrm and not serve:
            print("a .vrm must be fetched over http, so serving it locally")
        serve_dir(Path(out).parent, page=Path(out).name)
    else:
        print("open it in a browser (no server needed)")


def cmd_viewer(a):
    from src.viewer import build
    vrm = _resolve_vrm(a.vrm)
    out = build(Path(a.out) if a.out else VIEWER_PATH, sequences=a.sequence, vrm=vrm)
    _open_viewer(out, vrm, a.serve)


def cmd_encode(a):
    from src.encode import encode_many
    p = Path(a.video)
    vids = ([q for q in sorted(p.rglob('*')) if q.suffix.lower() in VIDEO_EXTS]
            if p.is_dir() else [p])
    res = encode_many(vids, Path(a.out) if a.out else None)
    for r in res:
        print(Path(r["video"]).name, "->", " ".join(r["sequence"]))
        for w in r["words"]:
            print(f"   {w['start']:>7.2f}s  {w['word']}")
    if a.out:
        print("saved ->", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("archives", help="list Zenodo video archives + sizes").set_defaults(func=cmd_archives)
    sub.add_parser("metadata", help="fetch INCLUDE metadata from Hugging Face").set_defaults(func=cmd_metadata)

    d = sub.add_parser("download", help="fetch INCLUDE videos from Zenodo")
    d.add_argument("--out", default=str(DATA_DIR))
    d.add_argument("--categories", default=None,
                   help="comma-separated, e.g. Animals,Greetings (default: all 57 GB)")
    d.add_argument("--max-gb", type=float, default=None, dest="max_gb",
                   help="stop after roughly this many GB of archives")
    d.add_argument("--keep-zips", action="store_true")
    d.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS,
                   help="parallel download connections (Zenodo throttles each one)")
    d.set_defaults(func=cmd_download)

    u = sub.add_parser("auto", help="download+pose every archive one by one, then cluster")
    u.add_argument("--data", default=str(DATA_DIR))
    u.add_argument("--cache", default=str(CACHE_DIR))
    u.add_argument("--categories", default=None, help="comma-separated subset (default: all)")
    u.add_argument("--max-gb", type=float, default=None, dest="max_gb")
    u.add_argument("--include50", action="store_true", help="only INCLUDE-50 videos")
    u.add_argument("--keep-videos", action="store_true", help="do not delete videos after posing")
    u.add_argument("--limit-per-archive", type=int, default=None, dest="limit_per_archive")
    u.add_argument("--no-cluster", action="store_true", help="skip the clustering step")
    u.add_argument("--k", type=int, default=None)
    u.add_argument("--restart", action="store_true", help="ignore the resume journal")
    u.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS,
                   help="parallel download connections (Zenodo throttles each one)")
    u.set_defaults(func=cmd_auto)

    e = sub.add_parser("extract", help="run pose estimation over the dataset")
    e.add_argument("--data", default=str(DATA_DIR))
    e.add_argument("--cache", default=str(CACHE_DIR))
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("--include50", action="store_true",
                   help="only videos in the INCLUDE-50 subset (needs `metadata`)")
    e.add_argument("--categories", default=None, help="comma-separated category filter")
    e.set_defaults(func=cmd_extract)

    c = sub.add_parser("cluster", help="KMeans over all frames, build pose map")
    c.add_argument("--cache", default=str(CACHE_DIR))
    c.add_argument("--k", type=int, default=None, help="omit to auto-pick k")
    c.set_defaults(func=cmd_cluster)

    n = sub.add_parser("names", help="list cluster names and their top signs")
    n.set_defaults(func=cmd_names)

    g = sub.add_parser("signs", help="list sign labels available in the keypoint cache")
    g.add_argument("--cache", default=None)
    g.set_defaults(func=cmd_signs)

    m = sub.add_parser("demo", help="replay a sign word on the avatar, from the cache")
    m.add_argument("word", help="a sign label, e.g. happy (see `cli.py signs`)")
    m.add_argument("-n", type=int, default=3, help="how many clips of that word")
    m.add_argument("--cache", default=None)
    m.add_argument("--vrm", default=None,
                   help="path to a .vrm avatar (default: a .vrm found in data/)")
    m.add_argument("--serve", action="store_true", help="serve the viewer over http")
    m.set_defaults(func=cmd_demo)

    v = sub.add_parser("viewer", help="build an HTML viewer to preview poses on an avatar")
    v.add_argument("--out", default=None)
    v.add_argument("--sequence", action="append", default=None,
                   help="an `encode --out` json file to make playable; repeatable")
    v.add_argument("--vrm", default=None,
                   help="path to a .vrm avatar (default: a .vrm found in data/)")
    v.add_argument("--serve", action="store_true", help="serve the viewer over http")
    v.set_defaults(func=cmd_viewer)

    x = sub.add_parser("encode", help="video (or folder) -> pose word array")
    x.add_argument("video")
    x.add_argument("--out", default=None)
    x.set_defaults(func=cmd_encode)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
