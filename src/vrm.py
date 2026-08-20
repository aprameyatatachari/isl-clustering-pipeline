"""Vendored three.js assets for previewing poses on a .vrm avatar.

three r147 is the last release that still ships UMD builds and `examples/js`,
which keeps the viewer free of ES modules. The files are downloaded once into
artifacts/vendor/ so the viewer keeps working offline afterwards.
"""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from .config import ARTIFACT_DIR, DATA_DIR

THREE = "0.147.0"
THREE_VRM = "2.1.1"
VENDOR_DIR = ARTIFACT_DIR / "vendor"

ASSETS = {
    "three.min.js": f"https://unpkg.com/three@{THREE}/build/three.min.js",
    "GLTFLoader.js": f"https://unpkg.com/three@{THREE}/examples/js/loaders/GLTFLoader.js",
    "OrbitControls.js": f"https://unpkg.com/three@{THREE}/examples/js/controls/OrbitControls.js",
    "three-vrm.js": f"https://unpkg.com/@pixiv/three-vrm@{THREE_VRM}/lib/three-vrm.js",
}


def default_model() -> Path | None:
    """A .vrm dropped in data/ is picked up without needing --vrm."""
    named = DATA_DIR / "avatar.vrm"
    if named.exists():
        return named
    found = sorted(DATA_DIR.glob("*.vrm"))
    return found[0] if found else None


def vendor(dest: Path = VENDOR_DIR) -> Path:
    """Download the three.js + three-vrm bundle once."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in ASSETS.items():
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"fetching {name}")
        tmp = target.with_suffix(target.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)
    return dest


def stage_model(vrm_path: Path, dest_dir: Path = ARTIFACT_DIR) -> str:
    """Copy the user's .vrm next to the viewer and return its relative name."""
    vrm_path = Path(vrm_path)
    if not vrm_path.exists():
        raise SystemExit(f"vrm not found: {vrm_path}")
    if vrm_path.suffix.lower() != ".vrm":
        raise SystemExit(f"expected a .vrm file, got: {vrm_path.name}")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "avatar.vrm"
    if vrm_path.resolve() != out.resolve():
        shutil.copyfile(vrm_path, out)
    return out.name


def serve(directory: Path = ARTIFACT_DIR, port: int = 8000, page: str = "viewer.html"):
    """Serve the viewer over HTTP.

    A .vrm has to be fetched by the page, and browsers block fetches from
    file:// URLs, so the viewer needs a real origin to load the avatar.
    """
    import functools
    import http.server
    import socketserver

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/{page}"
        print(f"serving {directory} at {url}")
        print("press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
