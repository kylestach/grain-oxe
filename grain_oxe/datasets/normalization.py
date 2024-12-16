import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Sequence, Tuple

import jax
import numpy as np


@dataclass
class Stats:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    p01: np.ndarray
    p05: np.ndarray
    p95: np.ndarray
    p99: np.ndarray


class NormalizationType(Enum):
    NONE = "none"

    MEAN_STD = "mean_std"
    SCALE_MEAN_STD = "scale_mean_std"
    MIN_MAX = "min_max"
    SCALE_MIN_MAX = "scale_min_max"
    PERCENTILE = "percentile"
    SCALE_PERCENTILE = "scale_percentile"


def _get_bounds(
    stat: Stats, normalization_type: NormalizationType
) -> Tuple[np.ndarray, np.ndarray]:
    if normalization_type == NormalizationType.MEAN_STD:
        upper = stat.mean + stat.std
        lower = stat.mean - stat.std
    elif normalization_type == NormalizationType.SCALE_MEAN_STD:
        upper = np.sqrt(stat.std**2 + stat.mean**2)
        lower = -upper
    elif normalization_type == NormalizationType.MIN_MAX:
        upper = stat.max
        lower = stat.min
    elif normalization_type == NormalizationType.SCALE_MIN_MAX:
        upper = (stat.max - stat.min) / 2
        lower = -upper
    elif normalization_type == NormalizationType.PERCENTILE:
        upper = stat.p99
        lower = stat.p01
    elif normalization_type == NormalizationType.SCALE_PERCENTILE:
        upper = (stat.p99 - stat.p01) / 2
        lower = -upper
    else:
        raise ValueError(f"Invalid normalization type: {normalization_type}")

    return lower, upper


def _normalize(
    data: np.ndarray, stat: Stats, normalization_type: NormalizationType
) -> np.ndarray:
    if (
        stat is None
        or normalization_type == NormalizationType.NONE
        or normalization_type is None
    ):
        return data
    lower, upper = _get_bounds(stat, normalization_type)
    return (data - lower) / (upper - lower)


def _unnormalize(
    data: np.ndarray, stat: Stats, normalization_type: NormalizationType
) -> np.ndarray:
    if stat is None:
        return data
    lower, upper = _get_bounds(stat, normalization_type)
    return data * (upper - lower) + lower


def _call_on_tree(func, data, stats, normalization_type):
    if not isinstance(data, dict):
        return func(data, stats, normalization_type)

    result = {}

    for key, value in data.items():
        if isinstance(stats, dict):
            child_stats = stats.get(key, None)
        else:
            child_stats = stats

        if isinstance(normalization_type, dict):
            child_normalization_type = normalization_type.get(key, None)
        else:
            child_normalization_type = normalization_type

        result[key] = _call_on_tree(func, value, child_stats, child_normalization_type)

    return result


class Normalizer:
    def __init__(self, stats: Dict[str, Stats], normalization_type: NormalizationType):
        self.stats = stats
        self.normalization_type = normalization_type

    @classmethod
    def load(
        cls, stats_dir: str, normalization_type: NormalizationType
    ) -> "Normalizer":
        # Load all .json files in the stats_dir
        stats = {}
        for file in os.listdir(stats_dir):
            if file.endswith(".json"):
                with open(os.path.join(stats_dir, file), "r") as f:
                    f = json.load(f)
                    stats[file[:-5]] = jax.tree.map(
                        lambda x: Stats(**x),
                        jax.tree.map(
                            np.asarray,
                            f,
                            is_leaf=lambda x: isinstance(x, list),
                        ),
                        is_leaf=lambda x: isinstance(x, dict)
                        and set(x.keys())
                        == {"mean", "std", "min", "max", "p01", "p05", "p95", "p99"},
                    )
        return cls(stats, normalization_type)

    def save(self, stats_dir: str):
        for dataset_name, stat in self.stats.items():
            stat = jax.tree.map(asdict, stat, is_leaf=lambda x: isinstance(x, Stats))
            with open(os.path.join(stats_dir, f"{dataset_name}.json"), "w") as f:
                json.dump(jax.tree.map(lambda x: x.tolist(), stat), f)

    def _get_stat_at_key(
        self, dataset_name: str, key: str | Sequence[str] | None = None
    ) -> Stats:
        stats = self.stats[dataset_name]
        if key is None:
            return stats
        elif isinstance(key, str):
            return getattr(stats, key)
        elif isinstance(key, Sequence[str]):
            for k in key:
                stats = getattr(stats, k)
            return stats
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

    def normalize(
        self,
        dataset_name: str,
        data: Dict[str, Any],
        key: str | Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        stats = self._get_stat_at_key(dataset_name, key)
        return _call_on_tree(
            _normalize,
            data,
            stats,
            self.normalization_type,
        )

    def unnormalize(
        self,
        dataset_name: str,
        data: Dict[str, Any],
        key: str | Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        stats = self._get_stat_at_key(dataset_name, key)
        return _call_on_tree(
            _unnormalize,
            data,
            stats,
            self.normalization_type,
        )
