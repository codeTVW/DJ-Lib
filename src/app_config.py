from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
USER_CONFIG_PATH = Path.home() / ".djlibrary" / "config.toml"

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "energy": {
        "low_max": 0.40,
        "mid_max": 0.75,
        "transition_max": 0.50,
        "warmup_max": 0.45,
        "peak_target": 0.85,
        "buildup_target": 0.62,
        "closing_target": 0.55,
        "club_warmup_target": 0.40,
        "afterhours_target": 0.35,
        "warmup_target": 0.30,
    },
    "bpm": {
        "warmup_max": 110.0,
        "club_warmup_max": 120.0,
        "buildup_min": 120.0,
        "buildup_max": 128.0,
        "peaktime_min": 128.0,
        "afterhours_max": 130.0,
        "warmup_trajectory_min": 100.0,
        "warmup_trajectory_max": 118.0,
        "target_club_warmup": 110.0,
        "target_buildup": 124.0,
        "target_peaktime": 132.0,
        "target_closing": 120.0,
        "target_afterhours": 125.0,
        "target_warmup": 109.0,
        "unusual_min": 60.0,
        "unusual_max": 220.0,
        "histogram_min": 60,
        "histogram_max": 220,
        "histogram_bin_size": 10,
    },
    "vocals": {
        "tools_min": 0.65,
    },
    "similarity": {
        "safe_min": 0.85,
        "safe_max_bpm_diff": 4.0,
        "creative_min": 0.65,
        "creative_max": 0.85,
        "creative_max_bpm_diff": 8.0,
    },
    "clustering": {
        "recalc_threshold": 0.10,
        "umap_n_components": 2,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "umap_random_state": 42,
        "hdbscan_min_cluster_size": 10,
        "hdbscan_min_samples": 5,
    },
}


def _merge_config(
    defaults: dict[str, dict[str, Any]],
    overrides: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    merged = {
        section: dict(values)
        for section, values in defaults.items()
    }
    for section, section_values in overrides.items():
        if section not in merged or not isinstance(section_values, dict):
            continue
        for key, value in section_values.items():
            merged[section][key] = value
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def load_app_config() -> dict[str, dict[str, Any]]:
    config = _merge_config(DEFAULT_CONFIG, _read_toml(DEFAULT_CONFIG_PATH))
    if USER_CONFIG_PATH.exists():
        config = _merge_config(config, _read_toml(USER_CONFIG_PATH))
    return config


def get_config_value(section: str, key: str, default: Any) -> Any:
    config = load_app_config()
    return config.get(section, {}).get(key, default)
