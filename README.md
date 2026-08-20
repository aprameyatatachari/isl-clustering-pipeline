# ISL pose clustering pipeline

Pose-estimate every INCLUDE video, KMeans the frames into preset poses, name each
cluster, and encode any video into a timestamped array of pose words.

## Install

Use a virtualenv -- a broken package in a global site-packages breaks mediapipe
in ways that surface as unrelated errors.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

The holistic model bundle (~130 MB) downloads itself into `artifacts/models/`
on the first run.

## Run

Hugging Face hosts only INCLUDE metadata; the videos come from Zenodo record
4010759 (44 zips, 57 GB total), so download by category.

### Automated (recommended)

One command walks every Zenodo archive in turn -- download, unzip, pose, delete
videos -- then clusters. Peak disk stays around one archive (~2 GB), and it
resumes from its journal if interrupted.

```bash
python cli.py auto                       # all 44 archives, then cluster
python cli.py auto --include50           # only the 958-video INCLUDE-50 subset
python cli.py auto --max-gb 10 --k 32    # small subset, fixed cluster count
python cli.py encode path/to/video.mp4
```

`--keep-videos` keeps the .MOV files, `--restart` ignores the resume journal
(`artifacts/auto_state.json`), `--no-cluster` stops after pose extraction.

### Manual steps

```bash
python cli.py archives                           # list Zenodo zips + sizes
python cli.py metadata                           # HF metadata -> artifacts/include_metadata.csv
python cli.py download --categories Animals,Greetings   # or --max-gb 10 for a subset
python cli.py extract                            # pose estimation -> artifacts/keypoints/
python cli.py cluster                            # KMeans + naming -> artifacts/pose_map.json
python cli.py names                              # list cluster names + top signs
python cli.py encode path/to/video.mp4 --out out.json
```

`download` is resumable and unzips into `data/`, deleting each zip after use
(`--keep-zips` to keep them). Omit `--categories`/`--max-gb` to pull all 57 GB.

Zenodo throttles each connection to under 1 MB/s but does not cap the total, so
archives are fetched as 16 parallel byte ranges (~7 MB/s measured, vs 0.8 MB/s
on a single stream). Tune with `--workers N`; completed 16 MB chunks are
journalled, so an interrupted download resumes without refetching them.

`--k N` on `cluster` fixes the cluster count (default: auto via silhouette).
`extract` takes `--limit N`, `--categories Animals,Greetings`, and `--include50`
(the 958-video INCLUDE-50 subset; run `metadata` first).
`encode` also accepts a folder and encodes every video inside it.

## Viewer

Preview any pose word on a skeleton avatar, and play an encoded video back as an
animation:

```bash
python cli.py encode data/Adjectives/5.\ Beautiful/MVI_9569.MOV --out out.json
python cli.py viewer --sequence out.json
```

Writes `artifacts/viewer.html`. Click a word to pose the avatar and read its
target rotations; use the timeline to scrub or play a sequence. `--sequence` is
repeatable, and can be omitted to browse poses only.

To replay a sign straight from the keypoint cache, without needing the source
video (which `auto` deletes after posing):

```bash
python cli.py signs          # sign labels available in the cache
python cli.py demo happy     # encodes 3 clips of that sign and opens the viewer
```

### 3D avatar (.vrm)

Drop a `.vrm` into `data/` (named `avatar.vrm`, or any single .vrm) and it is
picked up automatically; `--vrm path/to/model.vrm` overrides. Pose keypoints are
retargeted onto the model's humanoid arm and wrist bones, and the skeleton view
stays available behind a toggle.

```bash
python cli.py demo happy              # uses data/avatar.vrm if present
python cli.py viewer --sequence out.json --vrm mymodel.vrm
```

three.js and three-vrm are downloaded once into `artifacts/vendor/`. A .vrm has
to be fetched by the page and browsers block that on `file://` URLs, so these
commands start a local server and print its URL -- press Ctrl+C to stop it.
Without a .vrm the viewer is a single file that opens straight from disk.

## Output

`artifacts/pose_map.json` — per cluster: name, average pose, `target_rotations`
(shoulder pitch/yaw, elbow flexion, wrist direction), frame count, top signs.

`encode` returns `sequence` (words) plus `words` with `start`/`end`/`duration`
in seconds, relative to the first IDLE pose. First and last word are always IDLE.
