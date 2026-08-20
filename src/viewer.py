"""Build a self-contained HTML viewer for the pose map.

The pose map and any encoded sequences are inlined into the page, so the result
is a single file that opens straight from disk with no server involved.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import ARTIFACT_DIR, POSE_MAP_PATH

TEMPLATE = Path(__file__).with_name("viewer_template.html")
VRM_SCRIPT = Path(__file__).with_name("viewer_vrm.js")
VIEWER_PATH = ARTIFACT_DIR / "viewer.html"

# three.js is vendored rather than inlined: the bundle is ~1.5 MB and would
# bloat every rebuild of the page.
_VRM_HEAD = """<script src="vendor/three.min.js"></script>
<script src="vendor/GLTFLoader.js"></script>
<script src="vendor/OrbitControls.js"></script>
<script src="vendor/three-vrm.js"></script>
<script>window.VRM_FILE = "__VRM_FILE__";</script>
<script>__VRM_JS__</script>"""


def _load_sequences(paths) -> list:
    """Read `encode --out` files into [{name, words}] entries for playback."""
    out = []
    for p in paths or []:
        data = json.loads(Path(p).read_text())
        for entry in (data if isinstance(data, list) else [data]):
            out.append({
                "name": Path(entry["video"]).name,
                "words": entry["words"],
            })
    return out


def build(out: Path = VIEWER_PATH, pose_map_path: Path = POSE_MAP_PATH,
          sequences=None, vrm: Path | None = None) -> Path:
    pose_map_path = Path(pose_map_path)
    if not pose_map_path.exists():
        raise SystemExit(f"{pose_map_path} not found; run `python cli.py cluster` first")

    pose_map = json.loads(pose_map_path.read_text())
    seqs = _load_sequences(sequences)

    # Only keep words the pose map can actually draw, so playback cannot hit an
    # undefined lookup if a sequence was encoded against an older pose map.
    for s in seqs:
        unknown = {w["word"] for w in s["words"]} - set(pose_map)
        if unknown:
            raise SystemExit(f"{s['name']} uses words missing from the pose map: "
                             f"{sorted(unknown)}; re-run `encode` after clustering")

    vrm_head = ""
    if vrm:
        from .vrm import stage_model, vendor

        out_dir = Path(out).parent
        vendor(out_dir / "vendor")
        vrm_head = (_VRM_HEAD
                    .replace("__VRM_FILE__", stage_model(vrm, out_dir))
                    .replace("__VRM_JS__", VRM_SCRIPT.read_text(encoding="utf-8")))

    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__POSE_MAP__", json.dumps(pose_map, separators=(",", ":")))
            .replace("__SEQUENCES__", json.dumps(seqs, separators=(",", ":")))
            .replace("__VRM_SCRIPTS__", vrm_head))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
