"""Central config for the ISL pose-clustering pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "artifacts" / "keypoints"
ARTIFACT_DIR = ROOT / "artifacts"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}

# Frame sampling
TARGET_FPS = 10.0          # resample every video to this rate before extraction
MIN_VISIBILITY = 0.3       # landmark confidence below this is treated as missing

# MediaPipe landmark counts
N_POSE = 33
N_HAND = 21

# Upper-body pose landmarks we keep (face/legs carry no ISL signal)
UPPER_BODY_IDX = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0]

# Download: Zenodo throttles each connection to well under 1 MB/s, so archives
# are fetched as parallel byte ranges instead of one stream.
DOWNLOAD_WORKERS = 16
DOWNLOAD_CHUNK_MB = 16

# Clustering
KMEANS_K_RANGE = (12, 60)  # inclusive search range when k is auto-selected
KMEANS_SEED = 0
MAX_FRAMES_FOR_FIT = 200_000

# Encoding (video -> word array)
SMOOTH_WINDOW = 3          # median filter width over cluster ids, in frames
MIN_SEGMENT_FRAMES = 2     # drop shorter runs as noise
IDLE_PAD_SECONDS = 0.0     # synthetic idle duration if a video lacks a real one

# Artifact paths
KMEANS_PATH = ARTIFACT_DIR / "kmeans.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
POSE_MAP_PATH = ARTIFACT_DIR / "pose_map.json"
CLUSTER_NAMES_PATH = ARTIFACT_DIR / "cluster_names.json"
