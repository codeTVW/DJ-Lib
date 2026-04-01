from __future__ import annotations

try:
    from app_config import get_config_value
except ModuleNotFoundError:  # pragma: no cover - import fallback for project-root imports
    from src.app_config import get_config_value

ENERGY_LOW_MAX = float(get_config_value("energy", "low_max", 0.40))
ENERGY_MID_MAX = float(get_config_value("energy", "mid_max", 0.75))
ENERGY_TRANSITION_MAX = float(get_config_value("energy", "transition_max", 0.50))
ENERGY_WARMUP_MAX = float(get_config_value("energy", "warmup_max", 0.45))
ENERGY_PEAK_TARGET = float(get_config_value("energy", "peak_target", 0.85))
ENERGY_BUILDUP_TARGET = float(get_config_value("energy", "buildup_target", 0.62))
ENERGY_CLOSING_TARGET = float(get_config_value("energy", "closing_target", 0.55))
ENERGY_CLUB_WARMUP_TARGET = float(get_config_value("energy", "club_warmup_target", 0.40))
ENERGY_AFTERHOURS_TARGET = float(get_config_value("energy", "afterhours_target", 0.35))
ENERGY_WARMUP_TARGET = float(get_config_value("energy", "warmup_target", 0.30))

BPM_WARMUP_MAX = float(get_config_value("bpm", "warmup_max", 110.0))
BPM_CLUB_WARMUP_MAX = float(get_config_value("bpm", "club_warmup_max", 120.0))
BPM_BUILDUP_MIN = float(get_config_value("bpm", "buildup_min", 120.0))
BPM_BUILDUP_MAX = float(get_config_value("bpm", "buildup_max", 128.0))
BPM_PEAKTIME_MIN = float(get_config_value("bpm", "peaktime_min", 128.0))
BPM_AFTERHOURS_MAX = float(get_config_value("bpm", "afterhours_max", 130.0))
BPM_WARMUP_TRAJECTORY_MIN = float(get_config_value("bpm", "warmup_trajectory_min", 100.0))
BPM_WARMUP_TRAJECTORY_MAX = float(get_config_value("bpm", "warmup_trajectory_max", 118.0))

BPM_TARGET_CLUB_WARMUP = float(get_config_value("bpm", "target_club_warmup", 110.0))
BPM_TARGET_BUILDUP = float(get_config_value("bpm", "target_buildup", 124.0))
BPM_TARGET_PEAKTIME = float(get_config_value("bpm", "target_peaktime", 132.0))
BPM_TARGET_CLOSING = float(get_config_value("bpm", "target_closing", 120.0))
BPM_TARGET_AFTERHOURS = float(get_config_value("bpm", "target_afterhours", 125.0))
BPM_TARGET_WARMUP = float(get_config_value("bpm", "target_warmup", 109.0))

VOCAL_TOOLS_MIN = float(get_config_value("vocals", "tools_min", 0.65))

SIMILARITY_SAFE_MIN = float(get_config_value("similarity", "safe_min", 0.85))
SIMILARITY_SAFE_MAX_BPM_DIFF = float(get_config_value("similarity", "safe_max_bpm_diff", 4.0))
SIMILARITY_CREATIVE_MIN = float(get_config_value("similarity", "creative_min", 0.65))
SIMILARITY_CREATIVE_MAX = float(get_config_value("similarity", "creative_max", 0.85))
SIMILARITY_CREATIVE_MAX_BPM_DIFF = float(
    get_config_value("similarity", "creative_max_bpm_diff", 8.0)
)

UNUSUAL_BPM_MIN = float(get_config_value("bpm", "unusual_min", 60.0))
UNUSUAL_BPM_MAX = float(get_config_value("bpm", "unusual_max", 220.0))
BPM_HISTOGRAM_MIN = int(get_config_value("bpm", "histogram_min", 60))
BPM_HISTOGRAM_MAX = int(get_config_value("bpm", "histogram_max", 220))
BPM_HISTOGRAM_BIN_SIZE = int(get_config_value("bpm", "histogram_bin_size", 10))

CLUSTER_RECALC_THRESHOLD = float(get_config_value("clustering", "recalc_threshold", 0.10))
UMAP_N_COMPONENTS = int(get_config_value("clustering", "umap_n_components", 2))
UMAP_N_NEIGHBORS = int(get_config_value("clustering", "umap_n_neighbors", 15))
UMAP_MIN_DIST = float(get_config_value("clustering", "umap_min_dist", 0.1))
UMAP_RANDOM_STATE = int(get_config_value("clustering", "umap_random_state", 42))
HDBSCAN_MIN_CLUSTER_SIZE = int(get_config_value("clustering", "hdbscan_min_cluster_size", 10))
HDBSCAN_MIN_SAMPLES = int(get_config_value("clustering", "hdbscan_min_samples", 5))
